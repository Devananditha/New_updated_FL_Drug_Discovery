"""Coordinator database helpers for idempotent update processing."""

import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

_DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "ledger" / "ledger.db"

DEFAULT_MODEL_VERSION = "v1.0.0"
DEFAULT_ROUND_ID = 1
SYSTEM_CLIENT_ID = "SYSTEM"

LEDGER_EVENT_QUERY_STARTED = "query_started"
LEDGER_EVENT_CLIENT_RESPONDED = "client_responded"
LEDGER_EVENT_UPDATE_UPLOADED = "update_uploaded"
LEDGER_EVENT_UPDATE_COMMITTED = "update_committed"
LEDGER_EVENT_DUPLICATE_IGNORED = "duplicate_ignored"
LEDGER_EVENT_CLIENT_TIMEOUT = "client_timeout"
LEDGER_EVENT_CLIENT_RECOVERED = "client_recovered"

_LEDGER_COLUMNS: tuple[tuple[str, str], ...] = (
    ("request_id", "TEXT"),
    ("response_id", "TEXT"),
    ("round_id", "INTEGER DEFAULT 1"),
    ("checkpoint_path", "TEXT"),
    ("model_version", "TEXT"),
    ("evidence_hash", "TEXT"),
)


def _resolve_db_path(db_path: str) -> str:
    """Resolve a database path relative to the project ledger directory.

    Args:
        db_path: Caller-provided database path or the shorthand ``ledger.db``.

    Returns:
        Absolute filesystem path to the SQLite database file.
    """
    if db_path == "ledger.db":
        return str(_DEFAULT_DB_PATH)
    return db_path


def default_checkpoint_path(client_id: str) -> str:
    """Return the canonical on-disk checkpoint path for a federated client."""
    if client_id == SYSTEM_CLIENT_ID:
        return ""
    slug = client_id.strip().lower().replace(" ", "_")
    return f"checkpoints/{slug}.pt"


def init_ledger_db(db_path: str = "ledger.db") -> None:
    """Create or migrate the checkpoint ledger to the extended schema."""
    resolved_path = _resolve_db_path(db_path)
    Path(resolved_path).parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(resolved_path, timeout=30.0) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS checkpoint_ledger (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                update_id TEXT UNIQUE,
                query_id TEXT,
                client_id TEXT,
                status TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                request_id TEXT,
                response_id TEXT,
                round_id INTEGER DEFAULT 1,
                checkpoint_path TEXT,
                model_version TEXT,
                evidence_hash TEXT
            )
            """
        )

        existing_columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(checkpoint_ledger)").fetchall()
        }
        for column_name, column_type in _LEDGER_COLUMNS:
            if column_name not in existing_columns:
                conn.execute(
                    f"ALTER TABLE checkpoint_ledger ADD COLUMN {column_name} {column_type}"
                )


def check_if_duplicate(update_id: str, db_path: str = "ledger.db") -> bool:
    """Check whether an update has already been committed to the ledger.

    Args:
        update_id: Unique client response identifier.
        db_path: SQLite database path or ``ledger.db`` shorthand.

    Returns:
        True if a committed row with the same ``update_id`` already exists.
    """
    resolved_path = _resolve_db_path(db_path)

    with sqlite3.connect(resolved_path, timeout=30.0) as conn:
        cursor = conn.execute(
            """
            SELECT update_id
            FROM checkpoint_ledger
            WHERE update_id = ?
              AND status = 'update_committed'
            """,
            (update_id,),
        )
        row = cursor.fetchone()

    return row is not None


def log_to_ledger(
    query_id: str,
    client_id: str,
    update_id: str,
    status: str,
    db_path: str = "ledger.db",
    *,
    request_id: str | None = None,
    response_id: str | None = None,
    round_id: int = DEFAULT_ROUND_ID,
    checkpoint_path: str | None = None,
    model_version: str | None = None,
    evidence_hash: str | None = None,
) -> None:
    """Append a checkpoint event to the SQLite ledger.

    Duplicate replay events are stored as separate audit rows so the original
    committed update remains immutable.

    Args:
        query_id: Federated query identifier assigned by the coordinator.
        client_id: Responding client identifier.
        update_id: Unique client update or response identifier.
        status: Ledger status such as ``update_committed`` or ``duplicate_ignored``.
        db_path: SQLite database path or ``ledger.db`` shorthand.
        request_id: Trace id for the parallel coordinator-to-client call.
        response_id: Client-issued response identifier.
        round_id: Federated iteration counter for this retrieval round.
        checkpoint_path: Archived local checkpoint location for the client.
        model_version: PyTorch weight schema version tag.
        evidence_hash: Integrity digest (typically the client ``batch_hash``).

    Returns:
        None
    """
    resolved_path = _resolve_db_path(db_path)
    timestamp = datetime.now().isoformat()
    ledger_update_id = update_id

    if status == "duplicate_ignored":
        ledger_update_id = f"{update_id}::duplicate_ignored::{query_id}"

    resolved_request_id = request_id or str(uuid.uuid4())
    resolved_response_id = response_id or str(uuid.uuid4())
    if checkpoint_path is None:
        resolved_checkpoint_path = default_checkpoint_path(client_id)
    else:
        resolved_checkpoint_path = checkpoint_path
    resolved_model_version = model_version or DEFAULT_MODEL_VERSION
    resolved_evidence_hash = evidence_hash or ""

    with sqlite3.connect(resolved_path, timeout=30.0) as conn:
        try:
            conn.execute(
                """
                INSERT INTO checkpoint_ledger (
                    query_id,
                    client_id,
                    update_id,
                    status,
                    timestamp,
                    request_id,
                    response_id,
                    round_id,
                    checkpoint_path,
                    model_version,
                    evidence_hash
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    query_id,
                    client_id,
                    ledger_update_id,
                    status,
                    timestamp,
                    resolved_request_id,
                    resolved_response_id,
                    round_id,
                    resolved_checkpoint_path,
                    resolved_model_version,
                    resolved_evidence_hash,
                ),
            )
        except sqlite3.IntegrityError:
            return


def get_audit_trail(limit: int = 100, db_path: str = "ledger.db") -> list[dict]:
    """Fetch recent checkpoint ledger rows for the audit dashboard."""
    resolved_path = _resolve_db_path(db_path)

    with sqlite3.connect(resolved_path, timeout=30.0) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            """
            SELECT
                id,
                query_id,
                client_id,
                update_id,
                status,
                timestamp,
                request_id,
                response_id,
                round_id,
                checkpoint_path,
                model_version,
                evidence_hash
            FROM checkpoint_ledger
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(row) for row in cursor.fetchall()]


def get_latest_client_checkpoint(client_name: str, db_path: str = "ledger.db") -> dict | None:
    """Fetch the most recent committed checkpoint for one federated client."""
    resolved_path = _resolve_db_path(db_path)

    with sqlite3.connect(resolved_path, timeout=30.0) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            """
            SELECT
                query_id,
                update_id,
                timestamp,
                checkpoint_path,
                model_version,
                evidence_hash,
                round_id
            FROM checkpoint_ledger
            WHERE client_id = ?
              AND status = 'update_committed'
            ORDER BY timestamp DESC, id DESC
            LIMIT 1
            """,
            (client_name,),
        )
        row = cursor.fetchone()

    if row is None:
        return None

    return dict(row)

"""
June 12 recovery simulation: stale Client_1 payload replay after restart.

Prerequisites:
  - Coordinator running on http://localhost:8000
  - Port 8001 free (stop the real Client_1 before running)
  - ledger/ledger.db contains at least one row with status 'update_committed'
"""

import json
import sqlite3
import sys
import threading
import time
from pathlib import Path

import requests
import uvicorn
from fastapi import FastAPI

PROJECT_ROOT = Path(__file__).resolve().parent
LEDGER_DB = PROJECT_ROOT / "ledger" / "ledger.db"
COORDINATOR_URL = "http://localhost:8000/global_retrieve"
MOCK_CLIENT_PORT = 8001
TEST_DRUG_ID = "CID000003488"
DUMMY_TARGETS = [9999, 8888]

mock_app = FastAPI()
STALE_PAYLOAD: dict = {}


@mock_app.get("/retrieve")
def mock_retrieve(drug_id: str):
    return STALE_PAYLOAD


def fetch_committed_update_id() -> str:
    if not LEDGER_DB.exists():
        print(f"Error: ledger database not found at {LEDGER_DB}")
        sys.exit(1)

    with sqlite3.connect(LEDGER_DB) as conn:
        row = conn.execute(
            """
            SELECT update_id
            FROM checkpoint_ledger
            WHERE status = 'update_committed'
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()

    if not row:
        print(
            "Error: no update_committed rows in ledger. "
            "Run a successful /global_retrieve first, then retry."
        )
        sys.exit(1)

    return row[0]


def start_mock_client_server():
    config = uvicorn.Config(
        mock_app,
        host="0.0.0.0",
        port=MOCK_CLIENT_PORT,
        log_level="warning",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    time.sleep(1.5)
    return server


def evidence_contains_dummy_targets(evidence_paths: list) -> bool:
    for entry in evidence_paths:
        path = entry.get("path", "")
        for target in DUMMY_TARGETS:
            if str(target) in path:
                return True
    return False


def main():
    update_id = fetch_committed_update_id()

    global STALE_PAYLOAD
    STALE_PAYLOAD = {
        "client_id": "Client_1",
        "drug_id": TEST_DRUG_ID,
        "targets": DUMMY_TARGETS,
        "status": "success",
        "update_id": update_id,
    }

    print("=" * 60)
    print("Recovery Simulation")
    print("=" * 60)
    print(f"\n[1] Loaded committed update_id from ledger:\n    {update_id}")
    print(f"\n[2] Crafted stale Client_1 payload (restart replay):")
    print(json.dumps(STALE_PAYLOAD, indent=2))

    print(f"\n[3] Starting mock Client_1 on port {MOCK_CLIENT_PORT}...")
    print("    (Ensure the real Client_1 is stopped.)")
    start_mock_client_server()

    print(f"\n[4] Triggering Coordinator pull: GET {COORDINATOR_URL}")
    try:
        response = requests.get(
            COORDINATOR_URL,
            params={"drug_id": TEST_DRUG_ID},
            timeout=30,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"Error: Coordinator request failed — {exc}")
        print("    Is the Coordinator running on port 8000?")
        sys.exit(1)

    result = response.json()

    print("\n[5] Coordinator response summary:")
    print(f"    query_id:              {result.get('query_id')}")
    print(f"    completeness_score:    {result.get('completeness_score')}")
    print(f"    available_clients:     {result.get('available_clients')}")
    print(f"    missing_clients:       {result.get('missing_clients')}")
    print(f"    evidence_paths_count:  {result.get('evidence_paths_count')}")

    client1_paths = [
        p for p in result.get("evidence_paths", []) if p.get("client_id") == "Client_1"
    ]
    dummy_leaked = evidence_contains_dummy_targets(result.get("evidence_paths", []))

    with sqlite3.connect(LEDGER_DB) as conn:
        ledger_row = conn.execute(
            "SELECT status FROM checkpoint_ledger WHERE update_id = ?",
            (update_id,),
        ).fetchone()

    print(f"\n[6] Ledger status for replayed update_id: {ledger_row[0] if ledger_row else 'not found'}")

    print("\n[7] Recovery verification:")
    print(f"    Client_1 evidence paths in response: {len(client1_paths)}")
    print(f"    Dummy targets ({DUMMY_TARGETS}) in evidence_paths: {dummy_leaked}")

    if len(client1_paths) == 0 and not dummy_leaked:
        print("\n    RESULT: PASS — stale duplicate payload rejected; dummy targets excluded.")
    else:
        print("\n    RESULT: FAIL — duplicate evidence was not fully rejected.")
        sys.exit(1)

    print("\n[8] Full Coordinator JSON:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

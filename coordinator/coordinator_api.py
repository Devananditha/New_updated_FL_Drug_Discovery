"""Central coordinator for federated drug-target graph retrieval."""

import asyncio
import sys
import uuid
from pathlib import Path

import httpx
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from coordinator.coordinator_db import (
    DEFAULT_MODEL_VERSION,
    LEDGER_EVENT_CLIENT_RECOVERED,
    LEDGER_EVENT_CLIENT_RESPONDED,
    LEDGER_EVENT_CLIENT_TIMEOUT,
    LEDGER_EVENT_DUPLICATE_IGNORED,
    LEDGER_EVENT_QUERY_STARTED,
    LEDGER_EVENT_UPDATE_COMMITTED,
    LEDGER_EVENT_UPDATE_UPLOADED,
    SYSTEM_CLIENT_ID,
    check_if_duplicate,
    default_checkpoint_path,
    get_audit_trail,
    get_latest_client_checkpoint,
    init_ledger_db,
    log_to_ledger,
)

app = FastAPI()

LEDGER_DB_PATH = str(PROJECT_ROOT / "ledger" / "ledger.db")
AUDIT_HTML_PATH = PROJECT_ROOT / "audit.html"

CLIENT_URLS = [
    "http://localhost:8001",
    "http://localhost:8002",
    "http://localhost:8003",
]

UNBATCHED_GAS_PER_TARGET = 10
BATCHED_GAS_PER_CLIENT = 50

_FEDERATED_ROUND_ID = 0

init_ledger_db(LEDGER_DB_PATH)


def _next_round_id() -> int:
    """Increment and return the federated retrieval round counter."""
    global _FEDERATED_ROUND_ID
    _FEDERATED_ROUND_ID += 1
    return _FEDERATED_ROUND_ID


async def _record_ledger_event(
    query_id: str,
    client_id: str,
    ledger_update_id: str,
    status: str,
    round_id: int,
    **ledger_fields,
) -> None:
    """Append one taxonomy event to the checkpoint ledger on a worker thread."""
    ledger_fields.pop("round_id", None)
    await asyncio.to_thread(
        log_to_ledger,
        query_id,
        client_id,
        ledger_update_id,
        status,
        LEDGER_DB_PATH,
        round_id=round_id,
        **ledger_fields,
    )


def _ledger_fields_from_response(response: dict, round_id: int) -> dict:
    """Map a client or synthetic coordinator response to extended ledger columns."""
    client_id = response.get("client_id", "unknown")
    batch_hash = response.get("batch_hash") or response.get("evidence_hash") or ""

    return {
        "request_id": response.get("request_id"),
        "response_id": response.get("response_id") or response.get("update_id"),
        "checkpoint_path": response.get("checkpoint_path") or default_checkpoint_path(client_id),
        "model_version": response.get("model_version", DEFAULT_MODEL_VERSION),
        "evidence_hash": batch_hash,
    }


def calculate_gas_optimization_metrics(
    all_targets_found: list,
    successful_clients_count: int,
    verified_batch_hashes: list[str],
) -> dict:
    """Estimate theoretical gas savings from batching vs per-target transactions."""
    simulated_unbatched_gas = len(all_targets_found) * UNBATCHED_GAS_PER_TARGET
    actual_batched_gas = successful_clients_count * BATCHED_GAS_PER_CLIENT

    if simulated_unbatched_gas == 0:
        gas_saved_percentage = 0.0
    else:
        gas_saved_percentage = round(
            ((simulated_unbatched_gas - actual_batched_gas) / simulated_unbatched_gas) * 100,
            2,
        )

    return {
        "simulated_unbatched_gas": simulated_unbatched_gas,
        "actual_batched_gas": actual_batched_gas,
        "gas_saved_percentage": gas_saved_percentage,
        "verified_batch_hashes": verified_batch_hashes,
    }


def aggregate_models(client_weights_list: list[dict]) -> dict:
    """Average client model weights using the FedAvg algorithm."""
    if not client_weights_list:
        return {}

    aggregated: dict[str, list] = {}
    layer_names = client_weights_list[0].keys()

    for layer_name in layer_names:
        stacked = torch.stack(
            [torch.tensor(weights[layer_name], dtype=torch.float32) for weights in client_weights_list]
        )
        aggregated[layer_name] = stacked.mean(dim=0).tolist()

    return aggregated


async def fetch_client_data(client: str, url: str, drug_id: str, timeout: float = 2.0) -> dict:
    """Fetch drug-target evidence from one lab client with a strict timeout.

    Args:
        client: Logical client identifier (for example, ``Client_1``).
        url: Base URL of the client service.
        drug_id: Drug identifier to query.
        timeout: Maximum seconds to wait for the client response.

    Returns:
        Parsed JSON response from the client, or a synthetic failure payload when
        the client is unreachable or exceeds the timeout.
    """
    target_url = f"{url}/retrieve?drug_id={drug_id}"
    request_id = str(uuid.uuid4())

    try:
        async with httpx.AsyncClient() as http_client:
            response = await http_client.get(target_url, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            payload["request_id"] = request_id
            return payload
    except (httpx.TimeoutException, httpx.RequestError, httpx.HTTPStatusError):
        return {
            "client_id": client,
            "drug_id": drug_id,
            "targets": [],
            "status": "failed_or_timeout",
            "request_id": request_id,
            "response_id": str(uuid.uuid4()),
            "checkpoint_path": default_checkpoint_path(client),
            "model_version": DEFAULT_MODEL_VERSION,
            "batch_hash": "",
        }


@app.get("/global_retrieve")
async def global_retrieve(drug_id: str, mode: str = "aware") -> dict:
    """Query all lab clients in parallel and return a partial federated answer.

    The coordinator aggregates evidence from available clients, records ledger
    events for exactly-once recovery, and reports completeness when some clients
    fail or time out.

    Args:
        drug_id: Drug identifier shared across all client queries.

    Returns:
        Federated retrieval payload containing query metadata, completeness score,
        available and missing clients, evidence paths, and raw client responses.
    """
    query_id = str(uuid.uuid4())
    round_id = _next_round_id()

    await _record_ledger_event(
        query_id,
        SYSTEM_CLIENT_ID,
        f"{query_id}::{LEDGER_EVENT_QUERY_STARTED}",
        LEDGER_EVENT_QUERY_STARTED,
        round_id,
        request_id=str(uuid.uuid4()),
        response_id="",
        checkpoint_path="",
        model_version=DEFAULT_MODEL_VERSION,
        evidence_hash="",
    )

    tasks = [
        fetch_client_data(f"Client_{index}", url, drug_id)
        for index, url in enumerate(CLIENT_URLS, start=1)
    ]
    raw_responses = await asyncio.gather(*tasks)

    available_clients = []
    missing_clients = []
    for response in raw_responses:
        status = response.get("status")
        client_id = response.get("client_id")
        if status in ("success", "not_found"):
            available_clients.append(client_id)
        elif status == "failed_or_timeout":
            missing_clients.append(client_id)

    if mode == "unaware" and len(missing_clients) > 0:
        raise HTTPException(
            status_code=500,
            detail=f"Federated query stalled: Missing required client nodes: {missing_clients}",
        )

    completeness_score = f"{len(available_clients)}/{len(CLIENT_URLS)}"

    evidence_paths = []
    client_weights_list = []
    all_targets_found = []
    verified_batch_hashes = []
    client_confidences = []
    successful_clients_count = 0
    for response in raw_responses:
        update_id = response.get("update_id")
        client_id = response.get("client_id", "unknown")
        status = response.get("status", "unknown")

        ledger_fields = _ledger_fields_from_response(response, round_id)

        if status == "failed_or_timeout" or not update_id:
            timeout_ledger_id = f"{query_id}::{client_id}::{LEDGER_EVENT_CLIENT_TIMEOUT}"
            await _record_ledger_event(
                query_id,
                client_id,
                timeout_ledger_id,
                LEDGER_EVENT_CLIENT_TIMEOUT,
                round_id,
                **ledger_fields,
            )
            continue

        await _record_ledger_event(
            query_id,
            client_id,
            f"{query_id}::{client_id}::{LEDGER_EVENT_CLIENT_RESPONDED}",
            LEDGER_EVENT_CLIENT_RESPONDED,
            round_id,
            **ledger_fields,
        )

        is_duplicate = await asyncio.to_thread(
            check_if_duplicate, update_id, LEDGER_DB_PATH
        )

        if is_duplicate:
            await _record_ledger_event(
                query_id,
                client_id,
                update_id,
                LEDGER_EVENT_DUPLICATE_IGNORED,
                round_id,
                **ledger_fields,
            )
            continue

        model_weights = response.get("model_weights")
        if model_weights:
            await _record_ledger_event(
                query_id,
                client_id,
                f"{query_id}::{client_id}::{LEDGER_EVENT_UPDATE_UPLOADED}::{update_id}",
                LEDGER_EVENT_UPDATE_UPLOADED,
                round_id,
                **ledger_fields,
            )

        await _record_ledger_event(
            query_id,
            client_id,
            update_id,
            LEDGER_EVENT_UPDATE_COMMITTED,
            round_id,
            **ledger_fields,
        )

        if response.get("status") in ("success", "not_found"):
            client_confidences.append(float(response.get("local_confidence", 0.0)))

        if response.get("status") == "success":
            successful_clients_count += 1
            targets = response.get("targets", [])
            all_targets_found.extend(targets)

            batch_hash = response.get("batch_hash")
            if batch_hash:
                verified_batch_hashes.append(batch_hash)

            if model_weights:
                client_weights_list.append(model_weights)

            for target in targets:
                evidence_paths.append(
                    {"client_id": client_id, "path": f"{drug_id} -> {target}"}
                )

    global_confidence = (
        round(sum(client_confidences) / len(client_confidences), 2)
        if client_confidences
        else 0.0
    )
    global_aggregated_model = aggregate_models(client_weights_list)
    gas_optimization_metrics = calculate_gas_optimization_metrics(
        all_targets_found,
        successful_clients_count,
        verified_batch_hashes,
    )

    return {
        "query": drug_id,
        "query_id": query_id,
        "round_id": round_id,
        "completeness_score": completeness_score,
        "retrieval_confidence_score": f"{int(global_confidence * 100)}%",
        "available_clients": available_clients,
        "missing_clients": missing_clients,
        "evidence_paths_count": len(evidence_paths),
        "evidence_paths": evidence_paths,
        "global_aggregated_model": global_aggregated_model,
        "gas_optimization_metrics": gas_optimization_metrics,
        "raw_responses": raw_responses,
    }


@app.get("/audit_data")
async def audit_data(limit: int = 100) -> list[dict]:
    """Return recent checkpoint ledger rows as JSON."""
    return await asyncio.to_thread(get_audit_trail, limit, LEDGER_DB_PATH)


@app.get("/client_checkpoint/{client_name}")
async def client_checkpoint(client_name: str) -> dict:
    """Return the latest committed checkpoint known for a federated client."""
    checkpoint = await asyncio.to_thread(
        get_latest_client_checkpoint,
        client_name,
        LEDGER_DB_PATH,
    )

    if checkpoint is None:
        return {"status": "clean", "last_update_id": None}

    recovery_query_id = checkpoint.get("query_id") or str(uuid.uuid4())
    recovery_round_id = checkpoint.get("round_id") or 1
    recovery_fields = {
        "request_id": str(uuid.uuid4()),
        "response_id": checkpoint["update_id"],
        "checkpoint_path": checkpoint.get("checkpoint_path")
        or default_checkpoint_path(client_name),
        "model_version": checkpoint.get("model_version") or DEFAULT_MODEL_VERSION,
        "evidence_hash": checkpoint.get("evidence_hash") or "",
    }
    await _record_ledger_event(
        recovery_query_id,
        client_name,
        f"{client_name}::{LEDGER_EVENT_CLIENT_RECOVERED}::{uuid.uuid4()}",
        LEDGER_EVENT_CLIENT_RECOVERED,
        recovery_round_id,
        **recovery_fields,
    )

    return {
        "status": "found",
        "last_update_id": checkpoint["update_id"],
        "timestamp": checkpoint["timestamp"],
        "checkpoint_path": checkpoint.get("checkpoint_path"),
        "model_version": checkpoint.get("model_version"),
        "evidence_hash": checkpoint.get("evidence_hash"),
        "round_id": checkpoint.get("round_id"),
        "ledger_event": LEDGER_EVENT_CLIENT_RECOVERED,
    }


@app.get("/audit", response_class=HTMLResponse)
async def audit_dashboard() -> HTMLResponse:
    """Serve the read-only audit ledger dashboard."""
    html_content = AUDIT_HTML_PATH.read_text(encoding="utf-8")
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

"""Central coordinator for federated drug-target graph retrieval."""

import asyncio
import sys
import uuid
from pathlib import Path

import httpx
import torch
import uvicorn
from fastapi import FastAPI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from coordinator.coordinator_db import check_if_duplicate, log_to_ledger

app = FastAPI()

LEDGER_DB_PATH = str(PROJECT_ROOT / "ledger" / "ledger.db")

CLIENT_URLS = [
    "http://localhost:8001",
    "http://localhost:8002",
    "http://localhost:8003",
]


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

    try:
        async with httpx.AsyncClient() as http_client:
            response = await http_client.get(target_url, timeout=timeout)
            response.raise_for_status()
            return response.json()
    except (httpx.TimeoutException, httpx.RequestError, httpx.HTTPStatusError):
        return {
            "client_id": client,
            "drug_id": drug_id,
            "targets": [],
            "status": "failed_or_timeout",
        }


@app.get("/global_retrieve")
async def global_retrieve(drug_id: str) -> dict:
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

    completeness_score = f"{len(available_clients)}/{len(CLIENT_URLS)}"

    evidence_paths = []
    client_weights_list = []
    for response in raw_responses:
        update_id = response.get("update_id")
        client_id = response.get("client_id", "unknown")
        status = response.get("status", "unknown")

        if not update_id:
            timeout_update_id = f"{query_id}_{client_id}_{status}"
            await asyncio.to_thread(
                log_to_ledger,
                query_id,
                client_id,
                timeout_update_id,
                status,
                LEDGER_DB_PATH,
            )
            continue

        is_duplicate = await asyncio.to_thread(
            check_if_duplicate, update_id, LEDGER_DB_PATH
        )

        if is_duplicate:
            await asyncio.to_thread(
                log_to_ledger,
                query_id,
                client_id,
                update_id,
                "duplicate_ignored",
                LEDGER_DB_PATH,
            )
            continue

        await asyncio.to_thread(
            log_to_ledger,
            query_id,
            client_id,
            update_id,
            "update_committed",
            LEDGER_DB_PATH,
        )

        if response.get("status") == "success":
            model_weights = response.get("model_weights")
            if model_weights:
                client_weights_list.append(model_weights)

            for target in response.get("targets", []):
                evidence_paths.append(
                    {"client_id": client_id, "path": f"{drug_id} -> {target}"}
                )

    global_aggregated_model = aggregate_models(client_weights_list)

    return {
        "query": drug_id,
        "query_id": query_id,
        "completeness_score": completeness_score,
        "available_clients": available_clients,
        "missing_clients": missing_clients,
        "evidence_paths_count": len(evidence_paths),
        "evidence_paths": evidence_paths,
        "global_aggregated_model": global_aggregated_model,
        "raw_responses": raw_responses,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

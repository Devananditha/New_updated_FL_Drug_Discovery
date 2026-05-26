"""Central coordinator for federated drug-target graph retrieval."""

import asyncio
import sys
import uuid
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from ledger.ledger_manager import log_update

app = FastAPI()

CLIENT_URLS = [
    "http://localhost:8001",
    "http://localhost:8002",
    "http://localhost:8003",
]


async def fetch_client_data(client, url, drug_id, timeout=2.0):
    """Fetch targets from one lab client with a strict timeout."""
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
async def global_retrieve(drug_id: str):
    """Query all lab clients in parallel and aggregate successful targets."""
    query_id = str(uuid.uuid4())

    tasks = [
        fetch_client_data(f"Client_{index}", url, drug_id)
        for index, url in enumerate(CLIENT_URLS, start=1)
    ]
    raw_responses = await asyncio.gather(*tasks)

    for response in raw_responses:
        update_id = f"{query_id}_{response['client_id']}"
        log_update(
            update_id=update_id,
            query_id=query_id,
            client_id=response["client_id"],
            status=response["status"],
        )

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
    for response in raw_responses:
        if response.get("status") == "success":
            for target in response.get("targets", []):
                evidence_paths.append(
                    {"client_id": response["client_id"], "path": f"{drug_id} -> {target}"}
                )

    return {
        "query": drug_id,
        "query_id": query_id,
        "completeness_score": completeness_score,
        "available_clients": available_clients,
        "missing_clients": missing_clients,
        "evidence_paths_count": len(evidence_paths),
        "evidence_paths": evidence_paths,
        "raw_responses": raw_responses,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

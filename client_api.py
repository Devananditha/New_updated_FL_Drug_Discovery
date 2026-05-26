"""Reusable FastAPI client for local drug-target graph retrieval."""

import argparse
import uuid

import networkx as nx
import uvicorn
from fastapi import FastAPI, HTTPException

app = FastAPI()

G = None
CLIENT_NAME = "Unknown"


@app.get("/retrieve")
def retrieve(drug_id: str) -> dict:
    """Return local graph neighbors for a queried drug ID.

    Args:
        drug_id: Drug or compound identifier present in the local graph partition.

    Returns:
        JSON payload with client metadata, matched targets, status, and a unique
        ``update_id`` used by the coordinator for exactly-once ledger tracking.
    """
    if G is None:
        raise HTTPException(status_code=503, detail="Client graph is not loaded")

    if drug_id not in G:
        payload = {
            "client_id": CLIENT_NAME,
            "drug_id": drug_id,
            "targets": [],
            "status": "not_found",
        }
        payload["update_id"] = str(uuid.uuid4())
        return payload

    targets = list(G.neighbors(drug_id))
    payload = {
        "client_id": CLIENT_NAME,
        "drug_id": drug_id,
        "targets": targets,
        "status": "success",
    }
    payload["update_id"] = str(uuid.uuid4())
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a federated graph client API.")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--name", type=str, default="Client_1")
    args = parser.parse_args()

    G = nx.read_graphml(args.file)
    CLIENT_NAME = args.name

    uvicorn.run(app, host="0.0.0.0", port=args.port)

"""Reusable FastAPI client for local drug-target graph retrieval."""

import argparse
import hashlib
import uuid

import networkx as nx
import torch
import torch.nn as nn
import uvicorn
from fastapi import FastAPI, HTTPException

app = FastAPI()

G = None
CLIENT_NAME = "Unknown"


class LinkPredictor(nn.Module):
    """Lightweight MLP for simulated federated link prediction."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def state_dict_to_lists(state_dict: dict) -> dict:
    return {key: tensor.detach().cpu().tolist() for key, tensor in state_dict.items()}


def compute_batch_hash(targets: list) -> str:
    """Build an SHA-256 digest over batched targets (rollup-style integrity root)."""
    targets_string = ",".join(str(target) for target in targets)
    return hashlib.sha256(targets_string.encode("utf-8")).hexdigest()


def train_local_model(drug_id: str) -> dict:
    """Run a dummy 1-epoch local training loop and return serializable weights."""
    seed = abs(hash(drug_id)) % (2**32)
    torch.manual_seed(seed)

    model = LinkPredictor()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()

    features = torch.randn(32, 16)
    labels = torch.randint(0, 2, (32, 1)).float()

    model.train()
    optimizer.zero_grad()
    predictions = model(features)
    loss = criterion(predictions, labels)
    loss.backward()
    optimizer.step()

    return state_dict_to_lists(model.state_dict())


@app.get("/retrieve")
def retrieve(drug_id: str) -> dict:
    """Return local graph neighbors for a queried drug ID."""
    if G is None:
        raise HTTPException(status_code=503, detail="Client graph is not loaded")

    model_weights = train_local_model(drug_id)

    if drug_id not in G:
        targets: list = []
        payload = {
            "client_id": CLIENT_NAME,
            "drug_id": drug_id,
            "targets": targets,
            "status": "not_found",
            "model_weights": model_weights,
            "batch_hash": compute_batch_hash(targets),
        }
        payload["update_id"] = str(uuid.uuid4())
        return payload

    targets = list(G.neighbors(drug_id))
    payload = {
        "client_id": CLIENT_NAME,
        "drug_id": drug_id,
        "targets": targets,
        "status": "success",
        "model_weights": model_weights,
        "batch_hash": compute_batch_hash(targets),
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

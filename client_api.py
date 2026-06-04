"""Reusable FastAPI client for local drug-target graph retrieval."""

import argparse
from contextlib import asynccontextmanager
import hashlib
import random
import uuid

import networkx as nx
import requests
import torch
import torch.nn as nn
import uvicorn
from fastapi import FastAPI, HTTPException
from sklearn.metrics import (
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

G = None
CLIENT_NAME = "Unknown"
COORDINATOR_BASE_URL = "http://localhost:8000"
MODEL_VERSION = "v1.0.0"

NODE_TO_IDX: dict = {}
NUM_NODES = 0
EMBEDDING_DIM = 64
GLOBAL_EMBEDDING_VOCAB_SIZE = 100_000

POSITIVE_TRAIN_EDGES = 256
HOLDOUT_EDGES = 64
TRAINING_EPOCHS = 2
TOP_K = 50

REGRESSION_HIGH_AFFINITY = (5.0, 10.0)
REGRESSION_LOW_AFFINITY = (0.0, 3.0)
REGRESSION_SCORE_THRESHOLD = 5.0


def load_client_graph(graph_path: str) -> None:
    """Load the client graph and build the node-to-index mapping for embeddings."""
    global G, NODE_TO_IDX, NUM_NODES
    G = nx.read_graphml(graph_path)
    NODE_TO_IDX = {node: index for index, node in enumerate(G.nodes())}
    NUM_NODES = len(NODE_TO_IDX)


class ClientRuntimeState:
    """In-memory state restored from the coordinator checkpoint ledger."""

    def __init__(self) -> None:
        self.last_valid_checkpoint: str | None = None


CLIENT_STATE = ClientRuntimeState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup checkpoint recovery without deprecated FastAPI event hooks."""
    resume_from_checkpoint()
    yield


app = FastAPI(lifespan=lifespan)


class LinkPredictor(nn.Module):
    """Embedding-based MLP for link classification or DeepDTA-style affinity regression."""

    def __init__(
        self,
        embedding_dim: int = EMBEDDING_DIM,
        task_type: str = "classification",
    ) -> None:
        super().__init__()
        self.task_type = task_type
        self.embeddings = nn.Embedding(
            num_embeddings=GLOBAL_EMBEDDING_VOCAB_SIZE,
            embedding_dim=embedding_dim,
        )
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, drug_idx: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
        drug_emb = self.embeddings(drug_idx.long())
        target_emb = self.embeddings(target_idx.long())
        combined = torch.cat([drug_emb, target_emb], dim=-1)
        return self.mlp(combined)

    def predict_score(self, drug_idx: int, target_idx: int) -> float:
        """Return a classification probability or raw affinity prediction."""
        self.eval()
        with torch.no_grad():
            drug_tensor = torch.tensor([drug_idx], dtype=torch.long)
            target_tensor = torch.tensor([target_idx], dtype=torch.long)
            logit = self.forward(drug_tensor, target_tensor)
            if self.task_type == "regression":
                return float(logit.item())
            return float(torch.sigmoid(logit).item())


def state_dict_to_lists(state_dict: dict) -> dict:
    return {key: tensor.detach().cpu().tolist() for key, tensor in state_dict.items()}


def summarize_state_dict(state_dict: dict) -> dict:
    """Return layer shapes only — safe for browser/curl demos."""
    summary: dict = {}
    for layer_name, tensor in state_dict.items():
        shape = list(tensor.shape)
        summary[layer_name] = {"shape": shape, "num_params": int(tensor.numel())}
    return summary


def client_checkpoint_path() -> str:
    """Return the canonical checkpoint archive path for this client node."""
    slug = CLIENT_NAME.strip().lower().replace(" ", "_")
    return f"checkpoints/{slug}.pt"


def compute_batch_hash(targets: list) -> str:
    """Build an SHA-256 digest over batched targets (rollup-style integrity root)."""
    targets_string = ",".join(str(target) for target in targets)
    return hashlib.sha256(targets_string.encode("utf-8")).hexdigest()


def calculate_local_confidence(evidence_paths_count: int) -> float:
    """Estimate local retrieval confidence from the amount of graph evidence."""
    if evidence_paths_count == 0:
        return 0.0

    return round(min(0.99, 0.50 + (evidence_paths_count * 0.005)), 2)


def _sample_negative_edges(sample_count: int, rng: random.Random) -> list[tuple]:
    """Sample random node pairs that do not share an edge in the graph."""
    nodes = list(NODE_TO_IDX.keys())
    negative_edges: list[tuple] = []
    attempts = 0
    max_attempts = sample_count * 20

    while len(negative_edges) < sample_count and attempts < max_attempts:
        attempts += 1
        source = rng.choice(nodes)
        destination = rng.choice(nodes)
        if source == destination or G.has_edge(source, destination):
            continue
        negative_edges.append((source, destination))

    return negative_edges


def _empty_metrics(task_type: str) -> dict:
    """Return zeroed metrics for the active training task."""
    if task_type == "regression":
        return {"mse": 0.0, "r2": 0.0}
    return {
        "precision": 0.0,
        "recall": 0.0,
        "f1_score": 0.0,
        "top_50_precision": 0.0,
    }


def _affinity_labels(edge_count: int, is_positive: bool, task_type: str, rng: random.Random) -> list[float]:
    """Build binary or mock DeepDTA affinity labels for an edge batch."""
    if task_type == "regression":
        low, high = REGRESSION_HIGH_AFFINITY if is_positive else REGRESSION_LOW_AFFINITY
        return [rng.uniform(low, high) for _ in range(edge_count)]
    label_value = 1.0 if is_positive else 0.0
    return [label_value] * edge_count


def _edges_to_index_tensors(edge_pairs: list[tuple]) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert node-name edge pairs to drug/target index tensors for the model."""
    drug_indices = [NODE_TO_IDX[source] for source, _ in edge_pairs]
    target_indices = [NODE_TO_IDX[destination] for _, destination in edge_pairs]
    return (
        torch.tensor(drug_indices, dtype=torch.long),
        torch.tensor(target_indices, dtype=torch.long),
    )


def _evaluate_model(
    model: LinkPredictor,
    holdout_positive: list[tuple],
    holdout_negative: list[tuple],
    task_type: str,
    rng: random.Random,
) -> dict:
    """Compute classification or DeepDTA regression metrics on a holdout edge set."""
    evaluation_edges = holdout_positive + holdout_negative
    if not evaluation_edges:
        return _empty_metrics(task_type)

    y_true = _affinity_labels(len(holdout_positive), True, task_type, rng) + _affinity_labels(
        len(holdout_negative), False, task_type, rng
    )
    drug_tensor, target_tensor = _edges_to_index_tensors(evaluation_edges)

    model.eval()
    with torch.no_grad():
        logits = model(drug_tensor, target_tensor).squeeze(-1)
        if task_type == "regression":
            y_pred = logits.tolist()
        else:
            y_pred = torch.sigmoid(logits).tolist()

    if isinstance(y_pred, float):
        y_pred = [y_pred]

    if task_type == "regression":
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return {"mse": round(float(mse), 3), "r2": round(float(r2), 3)}

    y_pred_binary = [1.0 if probability >= 0.5 else 0.0 for probability in y_pred]

    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)

    ranked = sorted(zip(y_pred, y_true), key=lambda pair: pair[0], reverse=True)
    top_k = ranked[: min(TOP_K, len(ranked))]
    if top_k:
        p_at_k = sum(label for _, label in top_k) / len(top_k)
    else:
        p_at_k = 0.0

    return {
        "precision": round(float(precision), 3),
        "recall": round(float(recall), 3),
        "f1_score": round(float(f1), 3),
        "top_50_precision": round(float(p_at_k), 3),
    }


def train_model(drug_id: str, task_type: str = "classification") -> tuple[LinkPredictor, dict]:
    """Train the link predictor on real graph edges and return it with metrics."""
    if task_type not in ("classification", "regression"):
        raise ValueError(f"Unsupported task_type: {task_type}")

    seed = abs(hash(drug_id)) % (2**32)
    torch.manual_seed(seed)
    rng = random.Random(seed)

    model = LinkPredictor(task_type=task_type)

    all_edges = list(G.edges())
    if not all_edges or NUM_NODES == 0:
        return model, _empty_metrics(task_type)

    rng.shuffle(all_edges)
    positive_train = all_edges[:POSITIVE_TRAIN_EDGES]
    holdout_positive = all_edges[POSITIVE_TRAIN_EDGES : POSITIVE_TRAIN_EDGES + HOLDOUT_EDGES]

    negative_pool = _sample_negative_edges(len(positive_train) + HOLDOUT_EDGES, rng)
    negative_train = negative_pool[: len(positive_train)]
    holdout_negative = negative_pool[len(positive_train) :]

    train_edges = positive_train + negative_train
    train_labels = _affinity_labels(len(positive_train), True, task_type, rng) + _affinity_labels(
        len(negative_train), False, task_type, rng
    )
    if not train_edges:
        return model, _empty_metrics(task_type)

    drug_tensor, target_tensor = _edges_to_index_tensors(train_edges)
    label_tensor = torch.tensor(train_labels, dtype=torch.float32)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    if task_type == "regression":
        criterion = nn.MSELoss()
        loss_name = "MSE"
    else:
        criterion = nn.BCEWithLogitsLoss()
        loss_name = "BCE"

    print(
        f"[ML Training] {CLIENT_NAME} | task={task_type} | drug={drug_id} | nodes={NUM_NODES} | "
        f"train_edges={len(train_edges)} (pos={len(positive_train)}, neg={len(negative_train)})"
    )

    model.train()
    for epoch in range(TRAINING_EPOCHS):
        optimizer.zero_grad()
        logits = model(drug_tensor, target_tensor).squeeze(-1)
        loss = criterion(logits, label_tensor)
        loss.backward()
        optimizer.step()
        print(
            f"[ML Training] {CLIENT_NAME} epoch {epoch + 1}/{TRAINING_EPOCHS} "
            f"{loss_name} loss={loss.item():.4f}"
        )

    metrics = _evaluate_model(model, holdout_positive, holdout_negative, task_type, rng)
    print(f"[ML Training] {CLIENT_NAME} holdout metrics: {metrics}")
    return model, metrics


def score_candidate_paths(
    local_model: LinkPredictor,
    drug_id: str,
    candidate_targets: list,
) -> list[dict]:
    """Rank candidate graph edges by the local link-prediction model score."""
    scored_paths = []
    local_model.eval()

    drug_idx = NODE_TO_IDX.get(drug_id, 0)
    score_threshold = (
        REGRESSION_SCORE_THRESHOLD
        if local_model.task_type == "regression"
        else 0.5
    )

    with torch.no_grad():
        for target in candidate_targets:
            target_idx = NODE_TO_IDX.get(target, 0)
            score = local_model.predict_score(drug_idx, target_idx)
            if score > score_threshold:
                scored_paths.append(
                    {
                        "path": f"{drug_id} -> {target}",
                        "ml_score": round(score, 3),
                    }
                )

    scored_paths.sort(key=lambda item: item["ml_score"], reverse=True)
    return scored_paths[:50]


def resume_from_checkpoint() -> None:
    """Synchronize this client with its latest committed coordinator checkpoint."""
    checkpoint_url = f"{COORDINATOR_BASE_URL}/client_checkpoint/{CLIENT_NAME}"

    try:
        response = requests.get(checkpoint_url, timeout=5)
        response.raise_for_status()
        checkpoint = response.json()
    except requests.RequestException as exc:
        CLIENT_STATE.last_valid_checkpoint = None
        print(
            f"[Checkpoint Recovery] Unable to contact coordinator for node {CLIENT_NAME}: "
            f"{exc}. Initializing without a recovered checkpoint."
        )
        return

    last_update_id = checkpoint.get("last_update_id")
    if last_update_id:
        CLIENT_STATE.last_valid_checkpoint = last_update_id
        print(
            f"[Checkpoint Recovery] Resuming node {CLIENT_NAME}. "
            f"Last verified committed update: {last_update_id}"
        )
    else:
        CLIENT_STATE.last_valid_checkpoint = None
        print(
            f"[Checkpoint Recovery] No historical checkpoint found for node {CLIENT_NAME}. "
            "Initializing fresh state."
        )


def _fast_graph_retrieve(drug_id: str, task_type: str) -> tuple[str, list[dict], dict]:
    """Graph neighbor lookup only — used for load-test / demo speed (no ML training)."""
    if drug_id not in G:
        return "not_found", [], _empty_metrics(task_type)

    neighbors = list(G.neighbors(drug_id))[:TOP_K]
    scored_paths = [
        {"path": f"{drug_id} -> {target}", "ml_score": 0.72}
        for target in neighbors
    ]
    demo_metrics = (
        {"mse": 0.0, "r2": 0.0}
        if task_type == "regression"
        else {
            "precision": 0.6,
            "recall": 0.6,
            "f1_score": 0.6,
            "top_50_precision": 0.64,
        }
    )
    return "success", scored_paths, demo_metrics


def _build_retrieve_payload(
    *,
    drug_id: str,
    task_type: str,
    status: str,
    scored_paths: list,
    metrics: dict,
    state_dict: dict,
    include_weights: bool,
) -> dict:
    """Assemble client JSON; omit full tensors unless the coordinator requested them."""
    response_id = str(uuid.uuid4())
    payload = {
        "client_id": CLIENT_NAME,
        "drug_id": drug_id,
        "task_type": task_type,
        "targets": scored_paths,
        "status": status,
        "batch_hash": compute_batch_hash(scored_paths),
        "local_confidence": calculate_local_confidence(len(scored_paths)),
        "metrics": metrics,
        "response_id": response_id,
        "checkpoint_path": client_checkpoint_path(),
        "model_version": MODEL_VERSION,
        "update_id": str(uuid.uuid4()),
    }
    if include_weights:
        payload["model_weights"] = state_dict_to_lists(state_dict)
    else:
        payload["model_weights_summary"] = summarize_state_dict(state_dict)
    return payload


@app.get("/retrieve")
def retrieve(
    drug_id: str,
    task_type: str = "classification",
    include_weights: bool = False,
    fast: bool = False,
) -> dict:
    """Return local graph neighbors for a queried drug ID."""
    if task_type not in ("classification", "regression"):
        raise HTTPException(
            status_code=400,
            detail="task_type must be 'classification' or 'regression'",
        )
    if G is None:
        raise HTTPException(status_code=503, detail="Client graph is not loaded")

    if fast:
        print(f"[Retrieve] {CLIENT_NAME} fast lookup for {drug_id} (skipping ML training)")
        status, scored_paths, metrics = _fast_graph_retrieve(drug_id, task_type)
        state_dict = LinkPredictor(task_type=task_type).state_dict()
        return _build_retrieve_payload(
            drug_id=drug_id,
            task_type=task_type,
            status=status,
            scored_paths=scored_paths,
            metrics=metrics,
            state_dict=state_dict,
            include_weights=False,
        )

    if CLIENT_STATE.last_valid_checkpoint:
        print(
            f"[Checkpoint Recovery] Node {CLIENT_NAME} serving {drug_id} "
            f"on recovered baseline {CLIENT_STATE.last_valid_checkpoint}."
        )
    else:
        print(
            f"[Checkpoint Recovery] Node {CLIENT_NAME} serving {drug_id} "
            "without a recovered baseline checkpoint."
        )

    print(
        f"[Retrieve] {CLIENT_NAME} {task_type} training started for {drug_id}"
    )
    local_model, metrics = train_model(drug_id, task_type=task_type)
    state_dict = local_model.state_dict()
    print(f"[Retrieve] {CLIENT_NAME} training complete for {drug_id}")

    if drug_id not in G:
        return _build_retrieve_payload(
            drug_id=drug_id,
            task_type=task_type,
            status="not_found",
            scored_paths=[],
            metrics=metrics,
            state_dict=state_dict,
            include_weights=include_weights,
        )

    targets = list(G.neighbors(drug_id))
    scored_paths = score_candidate_paths(local_model, drug_id, targets)
    return _build_retrieve_payload(
        drug_id=drug_id,
        task_type=task_type,
        status="success",
        scored_paths=scored_paths,
        metrics=metrics,
        state_dict=state_dict,
        include_weights=include_weights,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a federated graph client API.")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--name", type=str, default="Client_1")
    args = parser.parse_args()

    load_client_graph(args.file)
    CLIENT_NAME = args.name

    uvicorn.run(app, host="0.0.0.0", port=args.port)

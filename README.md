# Failure-Aware Federated Graph-RAG for Drug-Target Retrieval with Exactly-Once Ledger Recovery

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST-green.svg)](https://fastapi.tiangolo.com/)
[![SQLite](https://img.shields.io/badge/SQLite-Exactly--Once%20Ledger-lightgrey.svg)](https://www.sqlite.org/)
[![Redis](https://img.shields.io/badge/Redis-Optional%20Stream%20Mirror-red.svg)](https://redis.io/)

Federated biomedical retrieval prototype that:
- keeps graph partitions local on each client,
- continues answering under partial client failure,
- enforces exactly-once update commitment with a checkpoint ledger,
- exposes runtime metrics for confidence, latency, failure-state, and ledger storage overhead.

---

## Table of Contents

1. [Core Novelty](#core-novelty)
2. [Architecture](#architecture)
3. [Runtime Flow](#runtime-flow)
4. [Technology Stack](#technology-stack)
5. [Setup](#setup)
6. [Runbook (From Scratch)](#runbook-from-scratch)
7. [Evaluation and Experiments](#evaluation-and-experiments)
8. [API Reference](#api-reference)
9. [Project Structure](#project-structure)
10. [Troubleshooting](#troubleshooting)

---

## Core Novelty

This project addresses two practical reliability issues in federated retrieval:

| Reliability Problem | Implementation |
|---|---|
| Client timeout/crash during query | Async parallel retrieval + degraded but valid response (`completeness_score`, `missing_clients`) |
| Stale update replay after reboot | Exactly-once duplicate detection using committed `update_id`; duplicates logged as `duplicate_ignored` and skipped |

### Why this is different
- Uses lightweight **SQLite append-only ledger** (not blockchain consensus) for exactly-once semantics.
- Adds a rich event taxonomy for full auditability.
- Mirrors ledger events to Redis Streams optionally to prove backend-agnostic design.

---

## Architecture

```mermaid
flowchart LR
    U[Researcher / Evaluator] -->|GET /global_retrieve| C
    U -->|GET /audit| C

    subgraph CO[Coordinator API :8000]
      C[coordinator_api.py]
      O1[Parallel Query Orchestrator]
      O2[Duplicate Check (Exactly-Once)]
      O3[FedAvg Aggregation]
      O4[Performance Metrics: latency, system_state, ledger_size_kb]
      C --> O1 --> O2 --> O3 --> O4
    end

    C -->|async /retrieve| C1
    C -->|async /retrieve| C2
    C -->|async /retrieve| C3

    subgraph CL[Lab Clients]
      C1[Client_1 :8001\nGraphML partition + PyTorch]
      C2[Client_2 :8002\nGraphML partition + PyTorch]
      C3[Client_3 :8003\nGraphML partition + PyTorch]
    end

    C1 -->|targets, ml_score, update_id, metrics| C
    C2 -->|targets, ml_score, update_id, metrics| C
    C3 -->|targets, ml_score, update_id, metrics| C

    C -->|append lifecycle events| SQL[(SQLite ledger.db\nsource of truth)]
    C -. optional mirror .-> REDIS[(Redis Stream\nfederated:ledger:stream)]

    SQL --> AUD[/audit_data + /audit dashboard/]
```

---

## Runtime Flow

1. `GET /global_retrieve?drug_id=...` hits coordinator.
2. Coordinator generates `query_id`, `round_id`, logs `query_started`.
3. Coordinator sends parallel `/retrieve` requests to all clients.
4. Per response:
   - logs `client_responded`,
   - runs `check_if_duplicate(update_id)`,
   - if duplicate: logs `duplicate_ignored` and skips commit,
   - else logs `update_uploaded` and `update_committed`.
5. Aggregates evidence + model weights, computes confidence and gas metrics.
6. Computes `performance_metrics`:
   - `total_latency_ms`
   - `system_state` (`no_failure` / `failure`)
   - `ledger_size_kb`
7. `/audit` displays lifecycle rows and live ledger storage overhead.

### Event Taxonomy
- `query_started`
- `client_responded`
- `update_uploaded`
- `update_committed`
- `duplicate_ignored`
- `client_timeout`
- `client_recovered`

---

## Technology Stack

| Layer | Tools |
|---|---|
| API server | FastAPI, Uvicorn |
| Graph retrieval | NetworkX |
| Local model scoring | PyTorch |
| Link-prediction metrics | scikit-learn |
| Async coordinator calls | httpx |
| Exactly-once ledger | SQLite |
| Optional stream backend | redis-py + Redis Streams |
| Analysis scripts | pandas, matplotlib, numpy |

---

## Setup

All commands assume project root.

### 1) Create and activate virtual environment

```bash
python -m venv venv
```

Windows (PowerShell):

```bash
.\venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Build graph partitions

```bash
python download_data.py
python graph_builder.py
python partition_data.py
```

Expected files:
- `data/client_1_graph.graphml`
- `data/client_2_graph.graphml`
- `data/client_3_graph.graphml`

### 4) Initialize ledger

```bash
python ledger/ledger_manager.py
```

---

## Runbook (From Scratch)

Open four terminals (venv active), then run:

Terminal 1:

```bash
python client_api.py --port 8001 --file data/client_1_graph.graphml --name Client_1
```

Terminal 2:

```bash
python client_api.py --port 8002 --file data/client_2_graph.graphml --name Client_2
```

Terminal 3:

```bash
python client_api.py --port 8003 --file data/client_3_graph.graphml --name Client_3
```

Terminal 4:

```bash
python coordinator/coordinator_api.py
```

Test query:

```text
http://localhost:8000/global_retrieve?drug_id=CID000000271
```

Audit dashboard:

```text
http://localhost:8000/audit
```

---

## Evaluation and Experiments

## 1) Failure vs no-failure latency (Task 14)

- Healthy run: all 3 clients up -> `completeness_score = 3/3` and `system_state = no_failure`.
- Failure run: stop one client -> `<3/3` and `system_state = failure`.
- Compare `performance_metrics.total_latency_ms` across both.

## 2) Duplicate replay experiment (Task 16)

```bash
python experiment_duplicate_rate.py
```

Prints:
- analytical no-ledger baseline duplicate rate,
- exactly-once interception rate,
- number of `duplicate_ignored` rows.

## 3) Storage overhead experiment (Task 18)

- `/audit` shows: `Ledger Storage Overhead: ... KB/MB`
- `/global_retrieve` returns: `performance_metrics.ledger_size_kb`

## 4) Optional Redis mirror validation

If Redis is running before coordinator startup:

```bash
redis-cli XLEN federated:ledger:stream
redis-cli XREVRANGE federated:ledger:stream + - COUNT 5
```

SQLite remains source of truth even when Redis is unavailable.

---

## API Reference

### `GET /retrieve?drug_id={drug_id}` (client)

Key fields:
- `status`: `success` / `not_found`
- `targets` (optional `ml_score`)
- `update_id`, `batch_hash`
- `metrics`: `precision`, `recall`, `f1_score`, `top_50_precision`
- `local_confidence`
- `model_weights`

### `GET /global_retrieve?drug_id={drug_id}&mode={aware|unaware}` (coordinator)

Key fields:
- `query_id`, `round_id`
- `completeness_score`, `available_clients`, `missing_clients`
- `evidence_paths`, `evidence_paths_count`
- `client_link_prediction_metrics`
- `federated_link_prediction_metrics`
- `gas_optimization_metrics`
- `performance_metrics`:
  - `total_latency_ms`
  - `system_state`
  - `ledger_size_kb`
- `raw_responses` (sanitized)

### `GET /audit_data?limit=100`
Returns latest ledger rows as JSON.

### `GET /audit`
Returns dashboard HTML with lifecycle rows and storage overhead badge.

### `GET /client_checkpoint/{client_name}`
Returns latest committed checkpoint metadata and logs `client_recovered` when applicable.

---

## Project Structure

```text
FL_DRUG_DISCOVERY/
├── client_api.py
├── coordinator/
│   ├── coordinator_api.py
│   └── coordinator_db.py
├── ledger/
│   ├── ledger_manager.py
│   └── ledger.db
├── data/
│   ├── ChG-TargetDecagon_targets.csv.gz
│   ├── client_1_graph.graphml
│   ├── client_2_graph.graphml
│   └── client_3_graph.graphml
├── audit.html
├── experiment_duplicate_rate.py
├── evaluate_baselines.py
├── data_collection.py
├── visualize_metrics.py
├── load_tester.py
├── simulate_recovery.py
├── requirements.txt
└── README.md
```

---

## Troubleshooting

| Issue | Resolution |
|---|---|
| `Errno 10048` port already in use | Find PID: `netstat -ano | findstr :800X`; kill: `taskkill /PID <pid> /F` |
| New fields missing in API/dashboard | Restart coordinator after code updates |
| Pull blocked by local changes | `git stash`; pull; `git stash pop` |
| Redis unavailable | Expected fallback: SQLite-only mode remains functional |
| Unexpected `2/3` completeness | Ensure all three clients are running and reachable |

---

## License

Academic research prototype for internship and educational use.

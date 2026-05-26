# Failure-Aware Federated Graph-RAG for Drug-Target Interaction Retrieval with Exactly-Once Checkpoint Recovery

A federated biomedical graph retrieval prototype that continues answering drug-target queries when simulated lab clients crash, reports retrieval completeness, and enforces exactly-once recovery using a lightweight append-only SQLite ledger.

## Core Novelty

Traditional federated retrieval systems often assume all clients are always available. This project introduces a failure-aware federated Graph-RAG layer that returns partial but transparent answers when clients fail, and uses an append-only SQLite checkpoint ledger to prevent duplicate client updates after recovery. The design avoids heavy blockchain consensus while still providing auditability and exactly-once update handling.

## System Architecture

- **Simulated Lab Clients** — Three independent FastAPI services, each holding a private BioSNAP graph partition on ports `8001`, `8002`, and `8003`.
- **Federated Coordinator** — Central service on port `8000` that broadcasts queries, applies strict timeouts, aggregates evidence paths, and computes completeness scores.
- **Append-Only Ledger** — SQLite database (`ledger/ledger.db`) that records query and client update events for recovery and duplicate detection.
- **Chaos / Load Tester** — Automated fault-injection script that kills and restarts Client 2 while querying the coordinator to evaluate degraded-mode behavior.

## Prerequisites & Setup

### 1. Create and activate a virtual environment

```bash
python -m venv venv
```

**Windows (PowerShell):**

```bash
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install fastapi uvicorn networkx pandas requests aiohttp psutil matplotlib numpy httpx
```

Or install from the project requirements file:

```bash
pip install -r requirements.txt
```

### 3. Download and partition the dataset

```bash
python download_data.py
python graph_builder.py
python partition_data.py
```

This creates:

- `data/ChG-TargetDecagon_targets.csv.gz`
- `data/client_1_graph.graphml`
- `data/client_2_graph.graphml`
- `data/client_3_graph.graphml`

### 4. Initialize the ledger database

```bash
python ledger/ledger_manager.py
```

## How to Run the System

Open **four separate terminals** from the project root with the virtual environment activated.

### 1. Start the 3 clients

**Terminal 1 — Client 1:**

```bash
python client_api.py --port 8001 --file data/client_1_graph.graphml --name Client_1
```

**Terminal 2 — Client 2:**

```bash
python client_api.py --port 8002 --file data/client_2_graph.graphml --name Client_2
```

**Terminal 3 — Client 3:**

```bash
python client_api.py --port 8003 --file data/client_3_graph.graphml --name Client_3
```

### 2. Start the coordinator

**Terminal 4:**

```bash
python coordinator/coordinator_api.py
```

### 3. Test a federated query (optional)

Open in a browser or API client:

```text
http://localhost:8000/global_retrieve?drug_id=CID000000271
```

Expected fields include `completeness_score`, `available_clients`, `missing_clients`, and `evidence_paths`.

### 4. Run the chaos load tester

```bash
python load_tester.py
```

For a shorter evaluation run:

```bash
python load_tester.py --queries 20
```

### 5. Generate final metrics and charts

```bash
python data_collection.py
python visualize_metrics.py
```

This prints ledger recovery metrics and saves:

- `reliability_chart.png`
- `recovery_chart.png`

## Project Structure

| Path | Description |
|------|-------------|
| `client_api.py` | Reusable FastAPI client for local graph retrieval |
| `coordinator/coordinator_api.py` | Federated coordinator with timeout and aggregation logic |
| `coordinator/coordinator_db.py` | SQLite ledger helpers for duplicate detection |
| `ledger/ledger_manager.py` | Ledger schema initialization utilities |
| `load_tester.py` | Fault-injection and load-testing script |
| `data_collection.py` | Extracts exactly-once recovery metrics from the ledger |
| `visualize_metrics.py` | Generates presentation-ready evaluation charts |
| `simulate_recovery.py` | Stale client payload replay recovery test |

## Dataset

BioSNAP / Stanford Drug-Target Interaction Network (ChG-Target Decagon):

https://snap.stanford.edu/biodata/datasets/10015/10015-ChG-TargetDecagon.html

## License

Academic research prototype for internship submission.

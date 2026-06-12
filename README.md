# Decentralized P2P Federated Drug Discovery

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Async%20REST-green.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-LinkPredictor-orange.svg)](https://pytorch.org/)
[![Architecture](https://img.shields.io/badge/Architecture-P2P%20Decentralized-purple.svg)]()
[![DHT](https://img.shields.io/badge/Routing-Kademlia%20XOR%20DHT-red.svg)]()
[![CRDT](https://img.shields.io/badge/Ledger-CRDT%20LWW--Map-yellow.svg)]()

A fully decentralized, peer-to-peer Federated Learning system for drug-target interaction discovery.  
Each peer holds a private graph partition and collaboratively trains a neural network — **with no central coordinator, no single point of failure**.

> **Course:** Decentralized Federated Learning  
> **Team:** Shiven Patro · Devananditha Vimalkumar  
> **Repository:** https://github.com/Alien0427/New_updated_FL_Drug_Discovery

---

## Table of Contents

1. [What This System Does](#what-this-system-does)
2. [Architecture](#architecture)
3. [Requirements Fulfillment](#requirements-fulfillment)
4. [How a Query Works (Step by Step)](#how-a-query-works-step-by-step)
5. [Setup](#setup)
6. [Running the System](#running-the-system)
7. [API Reference](#api-reference)
8. [Project Structure](#project-structure)
9. [Test Suite](#test-suite)
10. [Troubleshooting](#troubleshooting)

---

## What This System Does

This project implements a **Peer-to-Peer Federated Learning network** for biomedical drug discovery.

**The problem:** Drug-target interaction data is distributed across multiple research institutions. Each institution has private data they cannot share. But they want to collaboratively discover which drugs bind to which proteins, and build a better AI model together.

**Our solution:** A fully decentralized swarm where:
- Every peer holds its own **private graph partition** (drug-target interaction network)
- Every peer trains its own local **PyTorch neural network** (LinkPredictor)
- Peers collaborate via **DHT-routed (Kademlia) federated queries**
- Model weights are aggregated via **FedAvg** — raw data never leaves any peer
- A **CRDT-based distributed log** (replacing the old SQLite ledger) keeps all peers consistent
- If any peer crashes — the system **continues working** via fault-tolerant fallback routing

There is **no coordinator, no master node, no single point of failure**.

---

## Architecture

```
                  ┌──────────────────────────────────────────────────────┐
                  │              Decentralized P2P Overlay                │
                  │                                                        │
          ┌───────┴──────┐    DHT (Kademlia XOR)   ┌───────────────────┐ │
          │   Peer_1     │◄──────────────────────►│     Peer_2        │ │
          │  port 8001   │                         │   port 8002       │ │
          │  Private     │   Gossip / CRDT Sync    │   Private         │ │
          │  Graph #1    │◄───────────────────────►│   Graph #2        │ │
          └──────┬───────┘                         └─────────┬─────────┘ │
                 │           ┌─────────────────┐             │           │
                 └──────────►│     Peer_3      │◄────────────┘           │
                             │   port 8003     │                          │
                 ┌──────────►│   Private       │◄────────────┐           │
                 │           │   Graph #3      │             │           │
                 │           └─────────────────┘             │           │
                 │                                            │           │
          ┌──────┴───────┐                        ┌──────────┴────────┐  │
          │   Peer_4     │◄──────────────────────►│   Peer_N ...      │  │
          │  port 8004   │    FedAvg weights       │   port 800N       │  │
          └──────────────┘                        └───────────────────┘  │
                  └──────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────┐
  │                        Query Lifecycle                              │
  │                                                                     │
  │  Any Peer initiates                                                 │
  │       │                                                             │
  │       ▼                                                             │
  │  DHT routes query to all active peers (XOR distance, TTL-bounded)  │
  │       │                                                             │
  │       ▼                                                             │
  │  Each peer: train local model → return weights                     │
  │       │                                                             │
  │       ▼                                                             │
  │  Initiator runs FedAvg → broadcasts global model back to all       │
  │       │                                                             │
  │       ▼                                                             │
  │  CRDT ledger event gossiped to all peers automatically             │
  └─────────────────────────────────────────────────────────────────────┘
```

---

## Requirements Fulfillment

| # | Requirement | Implementation |
|---|-------------|----------------|
| **1** | No dedicated coordinator. Any node can initiate `global_retrieve`. | `GET /global_retrieve` is registered on **every** peer. Query fired at `Peer_4` (last-joined) succeeds identically to `Peer_1`. |
| **2** | Every node is both a **client** (private graph, local model) and a **routing/aggregation peer**. | Each peer loads its own `.graphml`, trains its own `LinkPredictor` (PyTorch), forwards DHT queries, runs FedAvg, and broadcasts the global model back to participants (`/receive_global_model`). |
| **3** | Query propagates via **DHT (Kademlia) — PREFERRED** with bounded TTL fallback. | `NODE_ID` = SHA-256(name + port) → 256-bit integer. `calculate_xor_distance()` implements Kademlia. `POST /dht_retrieve` is the TTL-decremented forwarding hop. Dead peers are marked `OFFLINE`; routing continues via survivors. |
| **4** | Each node keeps a **partial view** of the network (membership list). | `KNOWN_PEERS` dict per peer. `heartbeat_loop` pings every 10 s and performs gossip-based peer discovery by pulling each active peer's routing table. |
| **5** | Exactly-once ledger replaced with **CRDT-based distributed log**. | In-memory `CRDT_LEDGER` (LWW-Map) on every peer. `merge_crdt_event()` applies Last-Writer-Wins. `crdt_gossip_loop` runs every 15 s, syncing ledgers automatically across all peers. |

---

## How a Query Works (Step by Step)

When any peer receives `GET /global_retrieve?drug_id=CID000000271`:

```
Step 1  →  Generate unique query_id (UUID)
Step 2  →  Write "query_started" event to own CRDT_LEDGER
Step 3  →  Identify all ACTIVE peers from KNOWN_PEERS membership list
Step 4  →  Route to each peer via POST /dht_retrieve
             └─ Kademlia XOR distance determines routing order
             └─ TTL decremented at each hop — prevents infinite loops
             └─ Dead peers detected → marked OFFLINE → fallback to next-closest
Step 5  →  Each peer runs /local_retrieve:
             └─ Loads private graph partition
             └─ Trains PyTorch LinkPredictor (2 epochs)
             └─ Evaluates on holdout set (precision, recall, F1)
             └─ Returns model weights + predictions
Step 6  →  FedAvg: element-wise average of all weight tensors, layer by layer
Step 7  →  Write "update_committed" event to CRDT_LEDGER
Step 8  →  Broadcast FedAvg model to all participants via POST /receive_global_model
             └─ Each peer evaluates global model vs local model
             └─ Logs before/after F1 comparison to own CRDT_LEDGER
Step 9  →  Return aggregated metrics + model summary to caller
Step 10 →  Background: crdt_gossip_loop propagates ledger events to all other peers
```

---

## Setup

### 1. Virtual Environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Data (already prepared — skip if graphs exist)

The graph partitions are already in `data/`. Only run these if you need to rebuild from scratch:

```powershell
python download_data.py
python graph_builder.py
python partition_data.py
```

Expected outputs: `data/client_1_graph.graphml`, `data/client_2_graph.graphml`, `data/client_3_graph.graphml`

---

## Running the System

### Option A — One Command (Recommended for demo)

```powershell
python launch_network.py --nodes 4
```

This spawns 4 peer processes and opens a live refreshing ASCII dashboard:

```
+=================================================================+
|    P2P DRUG DISCOVERY  --  NETWORK ORCHESTRATOR DASHBOARD      |
+=================================================================+

+----------+------+--------+-------------+-------------+------------------+
| Node     | Port | Status | Known Peers | CRDT Ledger | DHT ID           |
+----------+------+--------+-------------+-------------+------------------+
| Peer_1   | 8001 | [ALIVE]|     3       |      5      | a3f2d1c0b4e9...  |
| Peer_2   | 8002 | [ALIVE]|     3       |      5      | 7b8c910ef213...  |
| Peer_3   | 8003 | [ALIVE]|     3       |      5      | 2d4f6a8c0e1b...  |
| Peer_4   | 8004 | [ALIVE]|     3       |      5      | f1e3d5c7b9a0...  |
+----------+------+--------+-------------+-------------+------------------+
Alive: 4/4  |  Fully Meshed: 4/4  |  Total CRDT Events: 20
```

Watch the **Known Peers** column climb from 0 → 3 as the gossip protocol forms the full mesh (~15 seconds).

**Press Ctrl+C** to terminate all peers cleanly.

### Option B — Manual Startup (for individual testing)

```powershell
# Terminal 1 — Seed node (no bootstrap)
python peer_node.py --port 8001 --file data/client_1_graph.graphml --name Peer_1

# Terminal 2 — Bootstrap via Peer_1
python peer_node.py --port 8002 --file data/client_2_graph.graphml --name Peer_2 --bootstrap http://localhost:8001

# Terminal 3 — Bootstrap via Peer_1
python peer_node.py --port 8003 --file data/client_3_graph.graphml --name Peer_3 --bootstrap http://localhost:8001
```

### Firing a Federated Query

```powershell
# Fire from ANY peer — here we use Peer_4 (last-joined node) to prove Requirement 1
curl "http://localhost:8004/global_retrieve?drug_id=CID000000271"

# Inspect the CRDT distributed ledger on any peer
curl "http://localhost:8001/crdt_state"

# Inspect the membership list (partial network view)
curl "http://localhost:8002/peers"

# Check DHT identity (Kademlia node_id)
curl "http://localhost:8001/ping"
```

---

## API Reference

All endpoints are available on **every peer** (ports 8001–800N). There is no special coordinator port.

### Health & Discovery

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ping` | GET | Returns node name, port, DHT `node_id` (256-bit), status |
| `/peers` | GET | Returns `KNOWN_PEERS` membership list with status and last_seen |
| `/bootstrap` | POST | Join the network — returns known routing table |

### DHT Routing

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/closest_peers?target_id=<hex>` | GET | Returns k-nearest peers sorted by Kademlia XOR distance |
| `/dht_retrieve` | POST | Internal forwarding hop — decrements TTL and routes query |

### Federated Learning

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/local_retrieve?drug_id=<id>` | GET | Train local model on private graph, return link predictions + weights |
| `/global_retrieve?drug_id=<id>` | GET | **Main endpoint** — full federated query + FedAvg + model dissemination |
| `/receive_global_model` | POST | Receive FedAvg weights; evaluate global vs local; log to CRDT |

### CRDT Distributed Ledger

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/crdt_state` | GET | View full `CRDT_LEDGER` on this peer — all committed events |
| `/crdt_sync` | POST | Push/pull ledger exchange with another peer (LWW merge) |

### Sample `/global_retrieve` Response

```json
{
  "query_id": "3f8a1c2d-...",
  "initiator": "Peer_4",
  "available_peers": ["Peer_4", "Peer_1", "Peer_2", "Peer_3"],
  "federated_link_prediction_metrics": {
    "precision": 0.58,
    "recall": 0.68,
    "f1_score": 0.63,
    "top_50_precision": 0.66
  },
  "global_model_summary": {
    "embeddings.weight": {"shape": [100000, 64], "num_params": 6400000},
    "mlp.0.weight":      {"shape": [64, 128],    "num_params": 8192},
    "mlp.2.weight":      {"shape": [1, 64],      "num_params": 64}
  },
  "dissemination_targets": ["Peer_1", "Peer_2", "Peer_3"],
  "crdt_update_id": "9b2e4f1a-...",
  "crdt_ledger_size": 5
}
```

---

## Project Structure

```
New_updated_FL_Drug_Discovery/
│
├── peer_node.py            ← THE ENTIRE P2P SYSTEM (1,572 lines)
│                             Every peer runs this. No separate coordinator or client.
│
├── launch_network.py       ← Network orchestrator + live CLI dashboard
│                             One command spawns N peers and monitors them.
│
├── data/
│   ├── client_1_graph.graphml          ← Private graph partition — Peer_1
│   ├── client_2_graph.graphml          ← Private graph partition — Peer_2
│   ├── client_3_graph.graphml          ← Private graph partition — Peer_3
│   └── ChG-TargetDecagon_targets.csv.gz ← Raw drug-target dataset (BioSNAP/ChEMBL)
│
├── test_commit4.py         ← Tests: CRDT LWW ledger (Requirement 5)
├── test_commit5.py         ← Tests: Kademlia XOR DHT routing (Requirement 3)
├── test_commit6.py         ← Tests: FedAvg aggregation (Requirement 2)
├── test_commit7.py         ← Tests: DHT query propagation with TTL (Requirement 3)
├── test_commit8.py         ← Tests: Anti-entropy CRDT gossip (Requirement 5)
├── test_commit9.py         ← Tests: Fault tolerance + node recovery (Requirement 3) ★
├── test_commit10.py        ← Tests: Global model dissemination (Requirement 2) ★
├── test_commit11.py        ← Tests: Full grand finale — all requirements (★ Self-spawning)
│
├── check_requirements.py   ← Scans peer_node.py for all 22 requirement tokens
│
├── data_collection.py      ← Data pipeline: collect raw drug-target data
├── download_data.py        ← Data pipeline: download from BioSNAP/PubChem
├── graph_builder.py        ← Data pipeline: build NetworkX graph from CSV
├── partition_data.py       ← Data pipeline: split graph into 3 private partitions
│
├── DOCUMENTATION.md        ← Full technical project documentation for submission
├── requirements.txt        ← Python dependencies
├── SETUP.md                ← Setup guide
└── README.md               ← This file

★ = Self-spawning test (no manual peer startup required)
```

---

## Test Suite

Run the automated requirement verifier first:

```powershell
python check_requirements.py
# Expected: ALL PRESENT -- fully implemented (22/22 OK)
```

| Test File | Requirement | Self-Spawning | What it proves |
|-----------|-------------|:---:|----------------|
| `test_commit4.py` | Req 5 — CRDT | No | LWW merge, idempotency, cross-peer sync |
| `test_commit5.py` | Req 3 — DHT | No | Kademlia XOR routing, node_id, k-buckets |
| `test_commit6.py` | Req 2 — FedAvg | No | Local training, weight averaging, metrics |
| `test_commit7.py` | Req 3 — Propagation | No | TTL-bounded DHT query flooding |
| `test_commit8.py` | Req 5 — Gossip | No | Automatic ledger propagation (no manual sync) |
| `test_commit9.py` | Req 3 — Fault Tolerance | ✅ Yes | Kill peer mid-query → system continues; recovery |
| `test_commit10.py` | Req 2 — Dissemination | ✅ Yes | FedAvg model broadcast to all participants |
| `test_commit11.py` | All Requirements | ✅ Yes | Complete end-to-end: swarm → mesh → query → CRDT |

Tests marked **Self-Spawning** start their own peer processes — no manual terminal setup needed.

---

## Troubleshooting

| Issue | Resolution |
|-------|-----------|
| Port already in use | `netstat -ano \| findstr :8001` then `taskkill /PID <pid> /F` |
| "Known Peers = 0" on dashboard | Wait 10–15 s for gossip/heartbeat to run. Peers discover each other automatically. |
| Query returns only 1 peer | Other peers haven't completed gossip mesh yet. Wait and retry. |
| Test fails with `Connection refused` | Ports may still be bound from a previous run. Kill all Python processes and retry. |
| `torch` import error | Run `pip install -r requirements.txt` with venv activated |
| Dashboard not refreshing | Ensure all 4 peer processes started successfully (check for port conflicts) |

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Peer API Server | FastAPI + Uvicorn (async HTTP) |
| Neural Network | PyTorch — Embedding-based MLP (LinkPredictor) |
| Graph Processing | NetworkX — GraphML drug-target partitions |
| DHT Routing | Custom Kademlia XOR-distance (256-bit SHA-256 node IDs) |
| Distributed Log | In-memory CRDT LWW-Map with anti-entropy gossip |
| HTTP Client | httpx (async) |
| Metrics | scikit-learn (F1, precision, recall, MSE, R²) |
| Orchestration | Python subprocess.Popen + live ASCII dashboard |
| Dataset | BioSNAP / ChEMBL drug-target interactions |

---

## License

Academic research prototype — Decentralized Federated Learning course project.

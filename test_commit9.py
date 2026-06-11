"""Commit 9 full harness: Spawns 4 nodes, kills Peer_2, verifies self-healing.

This script manages all 4 peer processes internally so it can simulate
true mid-network churn (force-killing Peer_2 before heartbeat catches it).
"""
import os
import sys
import subprocess
import time
import httpx
import signal

PYTHON = sys.executable
SCRIPT = "peer_node.py"
VENV_PY = r".\venv\Scripts\python.exe"

BASE1 = "http://localhost:8001"
BASE2 = "http://localhost:8002"
BASE3 = "http://localhost:8003"
BASE4 = "http://localhost:8004"

processes = {}


def section(title):
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print("=" * 65)


def start_peer(port, name, graph, bootstrap=None):
    cmd = [VENV_PY, SCRIPT, "--port", str(port), "--file", graph, "--name", name]
    if bootstrap:
        cmd += ["--bootstrap", bootstrap]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    processes[name] = proc
    return proc


def kill_peer(name):
    proc = processes.get(name)
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
    print(f"  [ASSASSIN] {name} process terminated (PID={proc.pid})")


def kill_all():
    for name, proc in processes.items():
        if proc and proc.poll() is None:
            proc.terminate()


def wait_for_peer(base, name, retries=15):
    for _ in range(retries):
        try:
            r = httpx.get(f"{base}/ping", timeout=2.0)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


try:
    # ------------------------------------------------------------------
    # Boot all 4 nodes
    # ------------------------------------------------------------------
    section("Booting 4 Peer Nodes")
    start_peer(8001, "Peer_1", "data/client_1_graph.graphml")
    print("  Peer_1 starting on 8001...")
    time.sleep(3)

    start_peer(8002, "Peer_2", "data/client_2_graph.graphml", bootstrap=BASE1)
    print("  Peer_2 starting on 8002 (bootstrap -> Peer_1)...")
    time.sleep(2)

    start_peer(8003, "Peer_3", "data/client_3_graph.graphml", bootstrap=BASE1)
    print("  Peer_3 starting on 8003 (bootstrap -> Peer_1)...")
    time.sleep(2)

    # Use 4th graph if available, else reuse client_1
    graph4 = "data/client_4_graph.graphml"
    if not os.path.exists(graph4):
        graph4 = "data/client_1_graph.graphml"
    start_peer(8004, "Peer_4", graph4, bootstrap=BASE1)
    print("  Peer_4 starting on 8004 (bootstrap -> Peer_1)...")

    print("\n  Waiting for all 4 nodes to be ready...")
    for base, name in [(BASE1,"Peer_1"),(BASE2,"Peer_2"),(BASE3,"Peer_3"),(BASE4,"Peer_4")]:
        ok = wait_for_peer(base, name)
        assert ok, f"FAIL: {name} never came online"
        print(f"  {name}: alive")
    print("  -> All 4 nodes up  [PASS]")

    # ------------------------------------------------------------------
    # Wait for mesh formation
    # ------------------------------------------------------------------
    section("Waiting 18s for heartbeat/gossip to form full mesh")
    for i in range(18, 0, -1):
        print(f"  {i}s...", end=" ", flush=True)
        time.sleep(1)
    print()

    r = httpx.get(f"{BASE1}/peers")
    p1_peers = r.json()
    print(f"  Peer_1 knows: {list(p1_peers['peers'].keys())}")
    assert p1_peers["total_known"] >= 3, f"FAIL: mesh not formed, Peer_1 knows only {p1_peers['total_known']}"

    # Verify Peer_2 is currently ACTIVE in Peer_1's table
    p2_status = p1_peers["peers"].get("http://localhost:8002", {}).get("status")
    print(f"  Peer_2 current status in Peer_1's table: {p2_status}")
    assert p2_status == "active", f"FAIL: Peer_2 should be active before kill, got {p2_status}"
    print("  -> Full mesh confirmed, Peer_2 active  [PASS]")

    # ------------------------------------------------------------------
    # THE ASSASSIN: Kill Peer_2
    # ------------------------------------------------------------------
    section("THE ASSASSIN: Force-killing Peer_2")
    kill_peer("Peer_2")
    print("  Peer_2 is DEAD. Heartbeat has NOT fired yet (next cycle in ~10s).")
    print("  Firing global_retrieve from Peer_1 IMMEDIATELY...")

    # ------------------------------------------------------------------
    # Fire the query — must complete via fallback
    # ------------------------------------------------------------------
    section("TEST: global_retrieve must self-heal via fallback routing")
    start = time.time()
    r = httpx.get(f"{BASE1}/global_retrieve?drug_id=DB00001&ttl=2", timeout=90.0)
    elapsed = time.time() - start
    assert r.status_code == 200, f"FAIL: global_retrieve returned {r.status_code}"
    data = r.json()

    print(f"\n  Query completed in {elapsed:.2f}s")
    print(f"  completeness_score : {data.get('completeness_score')}")
    print(f"  available_peers    : {data.get('available_peers')}")
    print(f"  missing_peers      : {data.get('missing_peers')}")

    ap = data.get("available_peers", [])
    assert len(ap) >= 2, f"FAIL: Expected >=2 successful responses, got {ap}"
    assert "Peer_1" in ap, "FAIL: Peer_1 should always be in available_peers"
    print("  -> Query completed with >=2 peers despite Peer_2 being dead  [PASS]")

    # ------------------------------------------------------------------
    # Verify Peer_2 now OFFLINE in Peer_1's routing table
    # ------------------------------------------------------------------
    section("Verify: Peer_2 marked OFFLINE in Peer_1's routing table")
    time.sleep(1)
    r = httpx.get(f"{BASE1}/peers")
    updated = r.json()
    p2_entry = updated["peers"].get("http://localhost:8002", {})
    p2_new_status = p2_entry.get("status")
    print(f"  Peer_2 new status: {p2_new_status}")
    assert p2_new_status == "offline", f"FAIL: Expected 'offline', got '{p2_new_status}'"
    print("  -> Peer_2 immediately marked OFFLINE by Fault Tolerance layer  [PASS]")

    # ------------------------------------------------------------------
    # Federated metrics valid from surviving peers
    # ------------------------------------------------------------------
    section("Verify: Federated metrics valid from surviving peers")
    fm = data.get("federated_link_prediction_metrics", {})
    print(f"  federated metrics: {fm}")
    assert "precision" in fm, "FAIL: precision missing"
    assert 0.0 <= fm["precision"] <= 1.0, "FAIL: precision out of range"
    print("  -> Federated metrics valid  [PASS]")

    # ------------------------------------------------------------------
    # Global aggregated model still produced
    # ------------------------------------------------------------------
    section("Verify: FedAvg global model still produced")
    gam = data.get("global_aggregated_model", {})
    print(f"  Layers in aggregated model: {list(gam.keys())[:3]}")
    assert len(gam) > 0, "FAIL: global_aggregated_model is empty"
    print("  -> FedAvg model produced from surviving peers  [PASS]")

    print("\n" + "=" * 65)
    print("  ALL TESTS PASSED - Commit 9 Fault Tolerance verified")
    print("=" * 65)

finally:
    print("\n[Harness] Shutting down all peer processes...")
    kill_all()

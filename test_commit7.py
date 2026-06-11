"""Commit 7 test script: Dynamic Bootstrapping (Network Entry)."""
import json
import time
import httpx

BASE1 = "http://localhost:8001"
BASE2 = "http://localhost:8002"
BASE3 = "http://localhost:8003"


def section(title):
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print("=" * 65)


# ------------------------------------------------------------------
# TEST 1: All 3 nodes are alive (no manual add_peer called!)
# ------------------------------------------------------------------
section("TEST 1: All 3 nodes alive (no manual /add_peer)")
for base, name in [(BASE1, "Peer_1"), (BASE2, "Peer_2"), (BASE3, "Peer_3")]:
    r = httpx.get(f"{base}/ping")
    assert r.status_code == 200, f"FAIL: {name} not responding"
    print(f"  {name}: alive")
print("  -> All 3 nodes up  [PASS]")

# ------------------------------------------------------------------
# TEST 2: Peer_1 automatically discovered Peer_2 and Peer_3
#         (via /bootstrap calls made during their startup)
# ------------------------------------------------------------------
section("TEST 2: Peer_1 routing table populated by bootstrap")
print("Waiting 12s for one heartbeat/gossip cycle to propagate routing tables...")
for i in range(12, 0, -1):
    print(f"  {i}s...", end=" ", flush=True)
    time.sleep(1)
print()

r = httpx.get(f"{BASE1}/peers")
p1 = r.json()
print(json.dumps({**p1, "peers": {k: {"status": v["status"]} for k, v in p1["peers"].items()}}, indent=2))
assert p1["total_known"] >= 2, f"FAIL: Peer_1 should know >=2 peers, knows {p1['total_known']}"
assert "http://localhost:8002" in p1["peers"], "FAIL: Peer_2 not in Peer_1 routing table"
assert "http://localhost:8003" in p1["peers"], "FAIL: Peer_3 not in Peer_1 routing table"
print("  -> Peer_1 knows Peer_2 and Peer_3 via bootstrap  [PASS]")

# ------------------------------------------------------------------
# TEST 3: Peer_2 knows Peer_1 AND Peer_3 (pulled routing table)
# ------------------------------------------------------------------
section("TEST 3: Peer_2 pulled routing table — knows Peer_1 AND Peer_3")
r = httpx.get(f"{BASE2}/peers")
p2 = r.json()
print(json.dumps({**p2, "peers": {k: {"status": v["status"]} for k, v in p2["peers"].items()}}, indent=2))
assert p2["total_known"] >= 2, f"FAIL: Peer_2 should know >=2 peers, knows {p2['total_known']}"
assert "http://localhost:8001" in p2["peers"], "FAIL: Peer_1 not in Peer_2's routing table"
assert "http://localhost:8003" in p2["peers"], "FAIL: Peer_3 not in Peer_2's routing table"
print("  -> Peer_2 discovered Peer_1 AND Peer_3 from routing table merge  [PASS]")

# ------------------------------------------------------------------
# TEST 4: Peer_3 knows Peer_1 AND Peer_2
# ------------------------------------------------------------------
section("TEST 4: Peer_3 pulled routing table — knows Peer_1 AND Peer_2")
r = httpx.get(f"{BASE3}/peers")
p3 = r.json()
print(json.dumps({**p3, "peers": {k: {"status": v["status"]} for k, v in p3["peers"].items()}}, indent=2))
assert p3["total_known"] >= 2, f"FAIL: Peer_3 should know >=2 peers, knows {p3['total_known']}"
assert "http://localhost:8001" in p3["peers"], "FAIL: Peer_1 not in Peer_3's routing table"
assert "http://localhost:8002" in p3["peers"], "FAIL: Peer_2 not in Peer_3's routing table"
print("  -> Peer_3 discovered both peers from routing table merge  [PASS]")

# ------------------------------------------------------------------
# TEST 5: /bootstrap endpoint returns correct structure
# ------------------------------------------------------------------
section("TEST 5: /bootstrap endpoint structure")
r = httpx.post(f"{BASE1}/bootstrap?peer_url=http://localhost:9999")
data = r.json()
print(json.dumps({**data, "known_peers": f"<{len(data.get('known_peers', {}))} entries>"}, indent=2))
assert r.status_code == 200, f"FAIL: status {r.status_code}"
assert data.get("result") == "welcome", f"FAIL: result={data.get('result')}"
assert "introducer" in data, "FAIL: 'introducer' key missing"
assert "known_peers" in data, "FAIL: 'known_peers' key missing"
assert data["introducer"] == "Peer_1", f"FAIL: introducer={data['introducer']}"
print("  -> /bootstrap returns welcome + known_peers  [PASS]")

# ------------------------------------------------------------------
# TEST 6: Self-URL is never added to own KNOWN_PEERS
# ------------------------------------------------------------------
section("TEST 6: No self-registration in KNOWN_PEERS")
for base, name, self_url in [
    (BASE1, "Peer_1", "http://localhost:8001"),
    (BASE2, "Peer_2", "http://localhost:8002"),
    (BASE3, "Peer_3", "http://localhost:8003"),
]:
    r = httpx.get(f"{base}/peers")
    peers = r.json()["peers"]
    assert self_url not in peers, f"FAIL: {name} has itself in KNOWN_PEERS!"
    print(f"  {name}: self-URL not in routing table  [OK]")
print("  -> No node has itself in KNOWN_PEERS  [PASS]")

# ------------------------------------------------------------------
# TEST 7: Full FedAvg still works (no manual add_peer needed)
# ------------------------------------------------------------------
section("TEST 7: FedAvg global_retrieve works with bootstrap-connected network")
print("Waiting 12s for heartbeats to resolve node_ids...")
for i in range(12, 0, -1):
    print(f"  {i}s...", end=" ", flush=True)
    time.sleep(1)
print()

r = httpx.get(f"{BASE1}/global_retrieve?drug_id=DB00001&ttl=2", timeout=90.0)
assert r.status_code == 200, f"FAIL: global_retrieve status {r.status_code}"
data = r.json()
cs = data.get("completeness_score")
ap = data.get("available_peers", [])
print(f"  completeness_score: {cs}")
print(f"  available_peers: {ap}")
assert cs == "3/3", f"FAIL: completeness_score={cs}"
assert len(ap) == 3, f"FAIL: expected 3 peers, got {len(ap)}"
print("  -> Full FedAvg across 3 bootstrapped peers  [PASS]")

print("\n" + "=" * 65)
print("  ALL 7 TESTS PASSED - Commit 7 Bootstrap verified successfully")
print("=" * 65)

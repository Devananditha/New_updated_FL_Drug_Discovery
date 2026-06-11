"""Commit 4 test script: In-Memory CRDT LWW Log."""
import json
import time
import uuid

import httpx

BASE1 = "http://localhost:8001"
BASE2 = "http://localhost:8002"


def section(title):
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print("=" * 65)


# ------------------------------------------------------------------
# TEST 1: /crdt_state starts empty
# ------------------------------------------------------------------
section("TEST 1: /crdt_state is empty at boot")
r = httpx.get(f"{BASE1}/crdt_state")
data = r.json()
print(json.dumps(data, indent=2))
assert r.status_code == 200, f"FAIL: status {r.status_code}"
assert data["ledger_size"] == 0, f"FAIL: expected 0 entries, got {data['ledger_size']}"
print("  -> Empty ledger on fresh boot  [PASS]")

# ------------------------------------------------------------------
# TEST 2: /global_retrieve commits an event to the CRDT ledger
# ------------------------------------------------------------------
section("TEST 2: /global_retrieve commits update_id to CRDT ledger")
r = httpx.get(f"{BASE1}/global_retrieve?drug_id=DB00001", timeout=60)
gr = r.json()
assert r.status_code == 200, f"FAIL: global_retrieve returned {r.status_code}"
assert "crdt_update_id" in gr, "FAIL: crdt_update_id missing from global_retrieve response"
committed_id = gr["crdt_update_id"]
print(f"  update_id returned: {committed_id}")

r2 = httpx.get(f"{BASE1}/crdt_state")
state = r2.json()
print(json.dumps(state, indent=2))
assert state["ledger_size"] == 1, f"FAIL: expected 1 entry, got {state['ledger_size']}"
entry = state["ledger"][committed_id]
assert entry["status"] == "update_committed", f"FAIL: wrong status: {entry['status']}"
assert entry["client_id"] == "Peer_1", f"FAIL: wrong client_id: {entry['client_id']}"
assert isinstance(entry["timestamp"], float), "FAIL: timestamp is not a float"
print(f"  -> CRDT entry committed: status={entry['status']}, client_id={entry['client_id']}  [PASS]")

# ------------------------------------------------------------------
# TEST 3: LWW merge — newer timestamp wins
# ------------------------------------------------------------------
section("TEST 3: merge_crdt_event LWW — newer timestamp overwrites")
old_uid = str(uuid.uuid4())
old_payload = {"status": "update_committed", "timestamp": 1000.0, "client_id": "Peer_X"}
new_payload = {"status": "update_committed", "timestamp": 2000.0, "client_id": "Peer_Y"}
stale_payload = {"status": "update_committed", "timestamp": 500.0, "client_id": "Peer_Z"}

# Insert old entry
r = httpx.post(f"{BASE1}/crdt_sync", json={old_uid: old_payload})
print("  Initial insert:", r.json())
assert r.json()["merged"] == 1, "FAIL: initial insert should merge"

# Send newer timestamp — should overwrite
r = httpx.post(f"{BASE1}/crdt_sync", json={old_uid: new_payload})
print("  Newer timestamp:", r.json())
assert r.json()["merged"] == 1, "FAIL: newer timestamp should be merged"

# Send stale timestamp — should be rejected
r = httpx.post(f"{BASE1}/crdt_sync", json={old_uid: stale_payload})
print("  Stale timestamp:", r.json())
assert r.json()["ignored"] == 1, "FAIL: stale timestamp should be ignored"

# Verify the winner
r = httpx.get(f"{BASE1}/crdt_state")
winner = r.json()["ledger"][old_uid]
assert winner["client_id"] == "Peer_Y", f"FAIL: LWW winner should be Peer_Y, got {winner['client_id']}"
print(f"  -> LWW winner = {winner['client_id']} (highest timestamp)  [PASS]")

# ------------------------------------------------------------------
# TEST 4: /crdt_sync — batch merge from another peer's ledger
# ------------------------------------------------------------------
section("TEST 4: /crdt_sync batch merge — multiple events at once")
batch = {
    str(uuid.uuid4()): {"status": "update_committed", "timestamp": time.time(), "client_id": "Peer_2"},
    str(uuid.uuid4()): {"status": "update_committed", "timestamp": time.time(), "client_id": "Peer_2"},
    str(uuid.uuid4()): {"status": "update_committed", "timestamp": time.time(), "client_id": "Peer_2"},
}
r = httpx.post(f"{BASE1}/crdt_sync", json=batch)
sync_result = r.json()
print(json.dumps(sync_result, indent=2))
assert sync_result["merged"] == 3, f"FAIL: expected 3 merged, got {sync_result['merged']}"
assert sync_result["ignored"] == 0, f"FAIL: expected 0 ignored, got {sync_result['ignored']}"
print("  -> 3 new events merged in one batch  [PASS]")

# ------------------------------------------------------------------
# TEST 5: /crdt_sync re-sending same batch — all ignored
# ------------------------------------------------------------------
section("TEST 5: /crdt_sync re-send same batch — all should be ignored")
r = httpx.post(f"{BASE1}/crdt_sync", json=batch)
result = r.json()
print(json.dumps(result, indent=2))
assert result["ignored"] == 3, f"FAIL: expected 3 ignored, got {result['ignored']}"
assert result["merged"] == 0, f"FAIL: expected 0 merged, got {result['merged']}"
print("  -> All 3 re-sent events ignored (LWW idempotent)  [PASS]")

# ------------------------------------------------------------------
# TEST 6: Cross-peer sync — Peer_1 ledger synced to Peer_2
# ------------------------------------------------------------------
section("TEST 6: Cross-peer CRDT sync — Peer_1 ledger pushed to Peer_2")
r = httpx.get(f"{BASE1}/crdt_state")
peer1_ledger = r.json()["ledger"]

r = httpx.get(f"{BASE2}/crdt_state")
peer2_before = r.json()["ledger_size"]
print(f"  Peer_2 ledger size before sync: {peer2_before}")

r = httpx.post(f"{BASE2}/crdt_sync", json=peer1_ledger)
sync_result = r.json()
print(json.dumps(sync_result, indent=2))

r = httpx.get(f"{BASE2}/crdt_state")
peer2_after = r.json()["ledger_size"]
print(f"  Peer_2 ledger size after sync: {peer2_after}")
assert peer2_after >= peer2_before, "FAIL: ledger should grow after sync"
assert sync_result["merged"] > 0, "FAIL: at least some events should be merged into Peer_2"
print(f"  -> {sync_result['merged']} events from Peer_1 merged into Peer_2  [PASS]")

# ------------------------------------------------------------------
# TEST 7: /crdt_state shows all committed events
# ------------------------------------------------------------------
section("TEST 7: /crdt_state final state inspection")
r = httpx.get(f"{BASE1}/crdt_state")
final = r.json()
print(json.dumps({**final, "ledger": f"<{final['ledger_size']} entries>"}, indent=2))
assert final["peer_id"] == "Peer_1", "FAIL: peer_id missing"
assert final["ledger_size"] > 1, "FAIL: should have multiple committed events"
print(f"  -> Ledger has {final['ledger_size']} entries  [PASS]")

print("\n" + "=" * 65)
print("  ALL 7 TESTS PASSED - Commit 4 verified successfully")
print("=" * 65)

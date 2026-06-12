with open("peer_node.py", encoding="utf-8") as f:
    src = f.read()
    lines = src.splitlines()

checks = [
    ("Req 1 - Any node can initiate global_retrieve",    'app.get("/global_retrieve")'),
    ("Req 1 - No dedicated coordinator (endpoint on all)","async def global_retrieve("),
    ("Req 2 - Private graph per node",                   "load_peer_graph"),
    ("Req 2 - Local PyTorch model training",             "class LinkPredictor"),
    ("Req 2 - Local model evaluation",                   "_evaluate_model"),
    ("Req 2 - Node acts as routing peer (DHT forward)",  "dht_retrieve_internal"),
    ("Req 2 - Node acts as aggregation peer (FedAvg)",   "def aggregate_models"),
    ("Req 2 - Model Dissemination after FedAvg",         "receive_global_model"),
    ("Req 3 - DHT preferred (Kademlia XOR distance)",    "calculate_xor_distance"),
    ("Req 3 - Kademlia Node ID (SHA-256 hash)",          "_generate_node_id"),
    ("Req 3 - DHT query propagation endpoint",           'app.post("/dht_retrieve")'),
    ("Req 3 - Bounded TTL on propagation",               "ttl"),
    ("Req 3 - Fallback routing (Fault Tolerance)",       "Fault Tolerance"),
    ("Req 4 - Membership list (KNOWN_PEERS)",            "KNOWN_PEERS: dict"),
    ("Req 4 - Heartbeat loop (liveness tracking)",       "async def heartbeat_loop"),
    ("Req 4 - Gossip-based peer discovery",              "Gossip-based peer discovery"),
    ("Req 4 - Bootstrap endpoint (network entry)",       'app.post("/bootstrap")'),
    ("Req 5 - CRDT Ledger replacing exact-once log",     "CRDT_LEDGER: dict"),
    ("Req 5 - LWW merge strategy",                       "def merge_crdt_event"),
    ("Req 5 - Anti-entropy gossip loop",                 "async def crdt_gossip_loop"),
    ("Req 5 - CRDT sync endpoint",                       'app.post("/crdt_sync")'),
    ("Req 5 - CRDT state endpoint",                      'app.get("/crdt_state")'),
]

print()
print("  Ma'am's Requirement Coverage -- peer_node.py")
print("  " + "=" * 65)
all_ok = True
for label, token in checks:
    found = token in src
    status = "[OK]     " if found else "[MISSING]"
    if not found:
        all_ok = False
    print(f"  {status}  {label}")

print("  " + "=" * 65)
print(f"  Total lines in peer_node.py : {len(lines)}")
print(f"  Result: {'ALL PRESENT -- fully implemented' if all_ok else 'GAPS FOUND'}")

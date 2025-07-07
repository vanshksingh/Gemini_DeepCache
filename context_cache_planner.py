import json
import time
import math
from collections import defaultdict, Counter
from pathlib import Path

# === CONFIG ===
CHUNKS_PATH         = Path("chunks.json")
QUERIES_PATH        = Path("queries.json")
MAX_BATCH_SIZE      = 5
CACHE_DISCOUNT      = 0.25    # pay 25% for explicit‐cached tokens
IMPLICIT_THRESHOLD  = 1024    # min tokens for implicit dynamic caching
CACHE_TTL           = 3600    # seconds (simulated TTL)

# === HELPERS ===
def estimate_tokens(text: str) -> int:
    return round(len(text.strip().split()) / 0.75)

def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 0

# === LOAD INPUT ===
chunk_data      = json.load(open(CHUNKS_PATH))
query_to_chunks = json.load(open(QUERIES_PATH))

# === DETERMINE CORE CHUNKS ===
# any chunk used in >1 query becomes “core”
freq = Counter()
for chunks in query_to_chunks.values():
    freq.update(chunks)
core_chunks = {cid for cid, f in freq.items() if f > 1}
core_tokens = sum(estimate_tokens(chunk_data[c]) for c in core_chunks)

# === BUILD BATCHES OF DYNAMIC PREFIXES ===
dynamic_map = defaultdict(list)
for q, chunks in query_to_chunks.items():
    dyn = tuple(sorted(set(chunks) - core_chunks))
    dynamic_map[dyn].append(q)

batches = []
for dyn, qs in dynamic_map.items():
    for i in range(0, len(qs), MAX_BATCH_SIZE):
        batches.append({
            "dynamic_chunks": dyn,
            "queries":        qs[i:i+MAX_BATCH_SIZE]
        })

# === ORDER BATCHES FOR IMPLICIT HITS ===
remaining = batches.copy()
ordered = [remaining.pop(0)]
while remaining:
    last = ordered[-1]["dynamic_chunks"]
    nxt = max(remaining, key=lambda b: jaccard(last, b["dynamic_chunks"]))
    remaining.remove(nxt)
    ordered.append(nxt)

# === SIMULATION ===
plan       = []
total_raw  = 0
total_opt  = 0
now        = time.time()
implicit_registry = {}

def simulate_implicit(prefix: tuple) -> int:
    """0 if first use, full dynamic token count if within TTL."""
    key = prefix
    tok = sum(estimate_tokens(chunk_data[c]) for c in prefix)
    last = implicit_registry.get(key)
    if last and (now - last) < CACHE_TTL:
        implicit_registry[key] = now
        return tok
    implicit_registry[key] = now
    return 0

# 1) Create one explicit cache for core_chunks
if core_chunks:
    plan.append({
        "action":     "create_explicit_cache",
        "chunk_ids":  sorted(core_chunks),
        "ctx_tokens": core_tokens,
        "ttl":        CACHE_TTL
    })
    total_opt += core_tokens

# 2) Process each batch
for gid, batch in enumerate(ordered, start=1):
    dyn     = batch["dynamic_chunks"]
    dyn_tok = sum(estimate_tokens(chunk_data[c]) for c in dyn)
    q_tok   = sum(estimate_tokens(q) for q in batch["queries"])

    # raw = core + dynamic + queries
    raw = core_tokens + dyn_tok + q_tok
    total_raw += raw

    # implicit dynamic hits
    imp_hits = simulate_implicit(dyn) if dyn_tok >= IMPLICIT_THRESHOLD else 0

    # explicit core hits & cost
    exp_hits = core_tokens
    exp_cost = core_tokens * CACHE_DISCOUNT

    # tokens actually sent
    uncached_dyn = max(0, dyn_tok - imp_hits)
    sent = uncached_dyn + q_tok

    # charge generate: sent + explicit discount
    total_opt += sent + exp_cost

    plan.append({
        "action":                "generate_content",
        "group_id":              gid,
        "batch_queries":         batch["queries"],
        "core_explicit":         True,
        "explicit_hits":         exp_hits,
        "explicit_cost":         exp_cost,
        "implicit_dynamic_hits": imp_hits,
        "dynamic_tokens":        dyn_tok,
        "uncached_dynamic":      uncached_dyn,
        "query_tokens":          q_tok,
        "sent_tokens":           sent
    })

# 3) Delete explicit cache at end
if core_chunks:
    plan.append({
        "action":    "delete_explicit_cache",
        "chunk_ids": sorted(core_chunks)
    })

# === PRINT INTERMEDIATE STEPS & SUMMARY ===
print("✅ Simulation Plan:")
for step in plan:
    print(json.dumps(step, ensure_ascii=False))

print("\n📊 Cost Summary:")
print(f"  Raw tokens sent     : {total_raw}")
print(f"  Optimized token cost: {total_opt}")
print(f"  Savings             : {total_raw - total_opt} tokens ({(100*(total_raw-total_opt)/total_raw):.2f}%)")

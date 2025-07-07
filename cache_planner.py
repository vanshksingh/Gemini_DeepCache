import json
import time
import math
from collections import defaultdict, Counter
from pathlib import Path

# === CONFIG ===
CHUNKS_PATH        = Path("chunks.json")
QUERIES_PATH       = Path("queries.json")
MAX_BATCH_SIZE     = 5
CACHE_DISCOUNT     = 0.25    # pay 25% for explicit‐cached tokens
IMPLICIT_THRESHOLD = 100    # min tokens for implicit dynamic caching
CACHE_TTL          = 3600    # seconds (simulated TTL)

# === HELPERS ===
def estimate_tokens(text: str) -> int:
    return round(len(text.strip().split()) / 0.75)

def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 0

# === LOAD INPUT ===
chunk_data      = json.load(open(CHUNKS_PATH))
query_to_chunks = json.load(open(QUERIES_PATH))

# === DYNAMIC CORE SELECTION ===
# Count how many generate calls each chunk would appear in:
# We assume one call per query group before batching.
query_freq = Counter()
for chunks in query_to_chunks.values():
    query_freq.update(chunks)

# Estimate number of batches per query as ceil(1/MAX_BATCH_SIZE),
# but we can approximate f_c ≈ query_freq[c] / MAX_BATCH_SIZE (rounded up)
batch_freq = {c: math.ceil(f / MAX_BATCH_SIZE)
              for c, f in query_freq.items()}

# Compute per-chunk benefit of explicit caching:
# benefit = (batch_freq[c] * chunk_tokens * (1 - discount)) - chunk_tokens
chunk_tokens = {c: estimate_tokens(t) for c, t in chunk_data.items()}
core_chunks = {
    c for c, f in batch_freq.items()
    if (f * chunk_tokens[c] * (1 - CACHE_DISCOUNT) - chunk_tokens[c]) > 0
}
core_tokens = sum(chunk_tokens[c] for c in core_chunks)

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
    key = prefix
    tok = sum(chunk_tokens[c] for c in prefix)
    last = implicit_registry.get(key)
    if last and (now - last) < CACHE_TTL:
        implicit_registry[key] = now
        return tok
    implicit_registry[key] = now
    return 0

# 1) Create explicit cache for core once
if core_chunks:
    plan.append({
        "action":     "create_explicit_cache",
        "chunk_ids":  sorted(core_chunks),
        "ctx_tokens": core_tokens,
        "ttl":        CACHE_TTL
    })
    total_opt += core_tokens

# 2) Process each dynamic batch
for gid, batch in enumerate(ordered, start=1):
    dyn     = batch["dynamic_chunks"]
    dyn_tok = sum(chunk_tokens[c] for c in dyn)
    q_tok   = sum(estimate_tokens(q) for q in batch["queries"])

    # raw = core + dynamic + queries
    raw = core_tokens + dyn_tok + q_tok
    total_raw += raw

    # implicit dynamic hits
    imp_hits = simulate_implicit(dyn) if dyn_tok >= IMPLICIT_THRESHOLD else 0

    # explicit core hits & cost
    exp_hits = core_tokens
    exp_cost = core_tokens * CACHE_DISCOUNT if core_chunks else 0

    # uncached dynamic tokens
    uncached_dyn = max(0, dyn_tok - imp_hits)
    sent = uncached_dyn + q_tok

    # charge generate: sent tokens + explicit discount
    total_opt += sent + exp_cost

    plan.append({
        "action":                "generate_content",
        "group_id":              gid,
        "batch_queries":         batch["queries"],
        "core_explicit":         bool(core_chunks),
        "implicit_dynamic_hits": imp_hits,
        "explicit_cost":         exp_cost,
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

# === OUTPUT ===
print("✅ Simulation Plan:")
for step in plan:
    print(json.dumps(step, ensure_ascii=False))
print("\n📊 Cost Summary:")
print(f"  Raw tokens sent     : {total_raw}")
print(f"  Optimized token cost: {total_opt}")
print(f"  Savings             : {total_raw - total_opt} tokens "
      f"({(100*(total_raw-total_opt)/total_raw):.2f}%)")

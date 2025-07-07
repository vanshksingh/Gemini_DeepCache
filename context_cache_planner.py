import json
from collections import defaultdict, Counter
from pathlib import Path

# === CONFIG ===
CHUNKS_PATH = Path("chunks.json")
QUERIES_PATH = Path("queries.json")
QUERY_COST_PER_TOKEN = 1.0
CHUNK_COST_PER_TOKEN = 1.0
CACHE_DISCOUNT = 0.25
TOP_K_IMPORTANT_CHUNKS = 3
MAX_BATCH_SIZE = 4

# === LOAD DATA ===
with open(CHUNKS_PATH) as f:
    chunk_data = json.load(f)

with open(QUERIES_PATH) as f:
    query_to_chunks = json.load(f)

# === TOKENIZER ===
def estimate_tokens(text):
    return round(len(text.strip().split()) / 0.75)

chunk_token_counts = {cid: estimate_tokens(text) for cid, text in chunk_data.items()}
chunk_token_texts = {cid: text for cid, text in chunk_data.items()}

# === Rank important chunks by frequency and query order weight ===
chunk_usage_freq = Counter()
query_to_index = {q: i for i, q in enumerate(query_to_chunks)}

for query, chunks in query_to_chunks.items():
    weight = len(query_to_chunks) - query_to_index[query]  # earlier queries have more weight
    for chunk in chunks:
        chunk_usage_freq[chunk] += weight

IMPORTANT_CHUNKS = set([cid for cid, _ in chunk_usage_freq.most_common(TOP_K_IMPORTANT_CHUNKS)])
print(f"🚨 Top-{TOP_K_IMPORTANT_CHUNKS} Important Chunks: {IMPORTANT_CHUNKS}")

# === Group similar queries based on shared chunk sets ===
chunk_to_queries = defaultdict(list)
for query, chunks in query_to_chunks.items():
    key = tuple(sorted(chunks))
    chunk_to_queries[key].append(query)

sorted_groups = sorted(chunk_to_queries.items(), key=lambda kv: (-len(kv[1]), -len(set(kv[0]) & IMPORTANT_CHUNKS)))

execution_plan = []
raw_cost = 0.0
optimized_cost = 0.0
current_explicit = set()

for i, (chunk_key, queries) in enumerate(sorted_groups, 1):
    grouped_queries = [queries[j:j+MAX_BATCH_SIZE] for j in range(0, len(queries), MAX_BATCH_SIZE)]
    for batch in grouped_queries:
        batch_chunk_set = set(chunk_key)
        chunk_tokens = sum(chunk_token_counts[c] for c in chunk_key)
        query_tokens = sum(estimate_tokens(q) for q in batch)
        batch_size = len(batch)
        raw_batch_cost = (chunk_tokens + query_tokens + 1) * batch_size
        raw_cost += raw_batch_cost

        use_explicit = False
        reuse_explicit = False

        if not current_explicit:
            use_explicit = True
        elif len(current_explicit & batch_chunk_set) >= TOP_K_IMPORTANT_CHUNKS - 1:
            reuse_explicit = True
        else:
            use_explicit = True

        if use_explicit:
            current_explicit = batch_chunk_set
            explicit_cost = chunk_tokens * CHUNK_COST_PER_TOKEN
            reuse_cost = chunk_tokens * CACHE_DISCOUNT * (batch_size - 1)
        elif reuse_explicit:
            explicit_cost = 0
            reuse_cost = chunk_tokens * CACHE_DISCOUNT * batch_size
        else:
            explicit_cost = chunk_tokens * CHUNK_COST_PER_TOKEN
            reuse_cost = chunk_tokens * CACHE_DISCOUNT * (batch_size - 1)

        query_cost = query_tokens * QUERY_COST_PER_TOKEN
        optimized_batch_cost = explicit_cost + reuse_cost + query_cost
        optimized_cost += optimized_batch_cost

        execution_plan.append({
            "group_id": i,
            "chunks": list(chunk_key),
            "queries": batch,
            "explicit_used": use_explicit,
            "reuse_explicit": reuse_explicit,
            "explicit_cost": explicit_cost,
            "reuse_cost": reuse_cost,
            "query_cost": query_cost,
            "chunk_token_count": chunk_tokens,
            "query_token_count": query_tokens,
            "batch_size": batch_size
        })

# === OUTPUT ===
print("\n📦 Execution Plan with Grouped Queries:")
for plan in execution_plan:
    print(f"\n🔹 Group {plan['group_id']}:")
    for cid in plan['chunks']:
        if cid in current_explicit:
            print(f"     🔐 Explicit Chunk: {cid}")
        else:
            print(f"     💭 Implicit Chunk: {cid}")
    print(f"   🔄 Queries in batch: {len(plan['queries'])}")
    print(f"   Query Token Total: {plan['query_token_count']}")
    print(f"   Chunk Token Total: {plan['chunk_token_count']}")
    print(f"   🔐 Explicit Cache: {plan['explicit_used']} | ♻️ Reuse: {plan['reuse_explicit']}")
    print(f"   💰 Explicit Cost: {plan['explicit_cost']:.2f}, Reuse Cost: {plan['reuse_cost']:.2f}, Query Cost: {plan['query_cost']:.2f}")
    print(f"   🔍 Queries:")
    for q in plan['queries']:
        print(f"      - {q}")

print("\n📊 Final Cost Report:")
print(f"   Raw cost (no cache): {raw_cost:.2f} tokens")
print(f"   Optimized cost     : {optimized_cost:.2f} tokens")
print(f"   💸 Savings         : {raw_cost - optimized_cost:.2f} tokens ({(100 * (raw_cost - optimized_cost) / raw_cost):.2f}%)")

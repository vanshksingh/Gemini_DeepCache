import json
from collections import Counter
import math
import hashlib

# CONFIG
EXPLICIT_TOKEN_BUDGET = 800
WORDS_PER_TOKEN = 1 / 0.75
TOKEN_COST = 1.0
MAX_GROUP_SIZE = 5

# --- Data Loading ---
def load_json_file(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def estimate_tokens(text):
    return math.ceil(len(text.split()) / WORDS_PER_TOKEN)

def load_chunks(chunks_raw):
    return {
        cid: {
            "text": text,
            "tokens": estimate_tokens(text)
        } for cid, text in chunks_raw.items()
    }

# --- Core Logic ---
def count_chunk_usage(queries):
    usage = Counter()
    for chunk_ids in queries.values():
        usage.update(chunk_ids)
    return usage

def select_explicit_chunks(chunks, usage, token_budget):
    scored = sorted(
        chunks.items(),
        key=lambda item: usage[item[0]] * item[1]["tokens"],
        reverse=True,
    )
    explicit = {}
    total = 0
    for cid, data in scored:
        if total + data["tokens"] > token_budget:
            continue
        explicit[cid] = data
        total += data["tokens"]
    return explicit

def assign_implicit_chunks(queries, explicit_chunks):
    return {
        query: [cid for cid in chunk_list if cid not in explicit_chunks]
        for query, chunk_list in queries.items()
    }

def group_similar_queries(queries, max_group=MAX_GROUP_SIZE):
    items = list(queries.items())
    return [dict(items[i:i + max_group]) for i in range(0, len(items), max_group)]

def compute_costs(query_chunks, chunks, explicit_chunks, seen_chunks):
    total_tokens = 0
    cost = 0
    saved = 0
    implicit_hit = True
    first_miss = None

    for i, cid in enumerate(query_chunks):
        tokens = chunks[cid]["tokens"]
        total_tokens += tokens

        if cid in explicit_chunks:
            if cid in seen_chunks:
                cost += tokens * 0.25
                saved += tokens * 0.75
            else:
                cost += tokens
        else:
            if implicit_hit and cid in seen_chunks:
                cost += tokens * 0.25
                saved += tokens * 0.75
            else:
                if implicit_hit:
                    first_miss = i
                implicit_hit = False
                cost += tokens

        seen_chunks.add(cid)

    return total_tokens, cost, saved, implicit_hit, first_miss

# --- Main ---
def main():
    chunks_raw = load_json_file("chunks.json")
    queries_raw = load_json_file("queries.json")
    chunks = load_chunks(chunks_raw)

    seen_chunks = set()
    last_cache_hash = None
    uploaded_cache_versions = set()

    total_query_cost = 0
    total_upload_cost = 0
    total_saved = 0
    total_tokens = 0

    grouped_batches = group_similar_queries(queries_raw)

    print("\n=== 🚀 CACHE PLAN EXECUTION ===")
    for i, query_batch in enumerate(grouped_batches):
        print(f"\n=== 🧩 Query Group {i+1} ===")
        usage = count_chunk_usage(query_batch)
        explicit_chunks = select_explicit_chunks(chunks, usage, EXPLICIT_TOKEN_BUDGET)

        current_hash = hashlib.md5("".join(sorted(explicit_chunks)).encode()).hexdigest()
        cache_changed = current_hash != last_cache_hash
        last_cache_hash = current_hash

        print(f"🔁 Explicit Cache Changed: {'Yes' if cache_changed else 'No'}")
        print("📌 Explicit Chunks:")
        for cid in explicit_chunks:
            print(f"  - {cid}")
            if cache_changed:
                # Always charged when cache changes, even for repeated chunks
                upload_cost = chunks[cid]["tokens"] * TOKEN_COST
                total_upload_cost += upload_cost

        implicit_plan = assign_implicit_chunks(query_batch, explicit_chunks)

        for query, chunk_list in query_batch.items():
            implicit_chunks = implicit_plan[query]
            tokens, cost, saved, implicit_hit, miss_pos = compute_costs(
                chunk_list, chunks, explicit_chunks, seen_chunks
            )

            total_query_cost += cost
            total_saved += saved
            total_tokens += tokens

            print(f"\n🔍 Query: {query}")
            print(f"   Implicit Chunks: {implicit_chunks if implicit_chunks else 'None'}")
            print(f"   🔢 Tokens: {tokens}")
            print(f"   💰 Query Cost: {cost:.2f}")
            print(f"   💸 Saved: {saved:.2f}")
            print(f"   📉 Savings %: {(saved / (cost + saved)) * 100:.1f}%")
            if not implicit_chunks:
                print("   ⚡ Implicit Cache Hit: N/A (no implicit chunks)")
            else:
                print(f"   ⚡ Implicit Cache Hit: {'Yes' if implicit_hit else f'No (missed at chunk ' + str(miss_pos) + ')'}")

    net_cost = total_query_cost + total_upload_cost

    print("\n=== 📊 FINAL SUMMARY ===")
    print(f"Total Queries: {len(queries_raw)}")
    print(f"Total Tokens Processed: {total_tokens}")
    print(f"Total Upload Cost (Explicit Cache): {total_upload_cost:.2f}")
    print(f"Total Query Cost: {total_query_cost:.2f}")
    print(f"Total Saved via Cache: {total_saved:.2f}")
    print(f"Net Cost After Savings: {net_cost:.2f}")
    print(f"Overall Savings: {(total_saved / (total_saved + net_cost)) * 100:.2f}%")

if __name__ == "__main__":
    main()

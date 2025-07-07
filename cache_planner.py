import json
import math
from collections import Counter, defaultdict

# ---- CONFIG ----
EXPLICIT_TOKEN_BUDGET = 100
WORDS_PER_TOKEN = 1 / 0.75
TOKEN_COST = 1.0
MAX_GROUP_SIZE = 5
IMPLICIT_TOKEN_LIMIT = 8192

# ---- LOADERS ----
def load_json_file(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def estimate_tokens(text):
    return math.ceil(len(text.split()) / WORDS_PER_TOKEN)

def load_chunks(chunks_raw):
    return {
        cid: {"text": text, "tokens": estimate_tokens(text)}
        for cid, text in chunks_raw.items()
    }

# ---- HELPERS ----
def count_chunk_usage(queries):
    usage = Counter()
    for chunk_ids in queries.values():
        usage.update(chunk_ids)
    return usage

def select_explicit_chunks(chunks, usage, token_budget):
    sorted_chunks = sorted(
        chunks.items(),
        key=lambda item: usage[item[0]] * item[1]["tokens"],
        reverse=True,
    )
    selected = {}
    total = 0
    for cid, data in sorted_chunks:
        if total + data["tokens"] <= token_budget:
            selected[cid] = data
            total += data["tokens"]
    return selected

def group_queries(queries, max_size=MAX_GROUP_SIZE):
    items = list(queries.items())
    return [dict(items[i:i+max_size]) for i in range(0, len(items), max_size)]

def plan_global_implicit_cache(all_queries, chunks, current_explicit):
    chunk_scores = Counter()
    for q_chunks in all_queries.values():
        for i, cid in enumerate(q_chunks):
            if cid not in current_explicit:
                chunk_scores[cid] += 1 / (i + 1)

    sorted_chunks = sorted(
        chunk_scores.items(),
        key=lambda item: (item[1], -chunks[item[0]]['tokens']),
        reverse=True
    )

    implicit_cache = []
    total_tokens = 0
    for cid, _ in sorted_chunks:
        tokens = chunks[cid]['tokens']
        if total_tokens + tokens <= IMPLICIT_TOKEN_LIMIT:
            implicit_cache.append(cid)
            total_tokens += tokens
        else:
            break

    return implicit_cache

# ---- COST SIMULATION ----
def compute_query_cost(query_chunks, chunks, explicit_chunks, seen_explicit, seen_implicit_ordered):
    total_tokens = 0
    query_cost = 0
    query_saved = 0

    explicit_ids = set(explicit_chunks.keys())
    implicit_chunks = [cid for cid in query_chunks if cid not in explicit_ids]

    for cid in query_chunks:
        if cid in explicit_ids:
            tokens = chunks[cid]["tokens"]
            total_tokens += tokens
            if cid in seen_explicit:
                query_cost += tokens * 0.25
                query_saved += tokens * 0.75
            else:
                query_cost += tokens
            seen_explicit.add(cid)

    hit_stop = None
    for i, cid in enumerate(implicit_chunks):
        tokens = chunks[cid]["tokens"]
        total_tokens += tokens

        if i < len(seen_implicit_ordered) and cid == seen_implicit_ordered[i] and hit_stop is None:
            query_cost += tokens * 0.25
            query_saved += tokens * 0.75
        else:
            if hit_stop is None:
                hit_stop = i
            query_cost += tokens

    if hit_stop == 0:
        hit_stop = len(implicit_chunks)

    return total_tokens, query_cost, query_saved, implicit_chunks, hit_stop

# ---- MAIN ----
def main():
    chunks_raw = load_json_file("chunks.json")
    queries_raw = load_json_file("queries.json")
    chunks = load_chunks(chunks_raw)

    grouped = group_queries(queries_raw)
    seen_explicit = set()
    seen_implicit_ordered = []
    current_explicit = {}
    total_upload_cost = 0
    total_query_cost = 0
    total_tokens = 0
    total_saved = 0

    print("\n=== 🚀 CACHE PLAN EXECUTION ===")

    usage = count_chunk_usage(queries_raw)
    current_explicit = select_explicit_chunks(chunks, usage, EXPLICIT_TOKEN_BUDGET)
    seen_implicit_ordered = plan_global_implicit_cache(queries_raw, chunks, current_explicit)

    print(f"\n🔁 Explicit Cache Selected (Upload cost: {sum(chunks[cid]['tokens'] for cid in current_explicit):.2f})")
    for cid in current_explicit:
        print(f"  - {cid}")

    print(f"\n🧠 Global Implicit Cache Plan: {len(seen_implicit_ordered)} chunks")
    for i, cid in enumerate(seen_implicit_ordered):
        print(f"   #{i+1}: {cid}")

    for group_index, group in enumerate(grouped):
        print(f"\n=== 🧩 Group {group_index+1} ===")
        group_tokens = group_cost = group_saved = 0
        print("\n📡 Simulating API Call for Group")

        for query, chunk_list in group.items():
            tokens, cost, saved, implicit_chunks, hit_stop = compute_query_cost(
                chunk_list, chunks, current_explicit,
                seen_explicit, seen_implicit_ordered
            )

            group_tokens += tokens
            group_cost += cost
            group_saved += saved

            total_tokens += tokens
            total_query_cost += cost
            total_saved += saved

            print(f"\n🔍 Query: {query}")
            print(f"   Implicit Chunks: {implicit_chunks if implicit_chunks else 'None'}")
            print(f"   🔢 Tokens: {tokens}")
            print(f"   💰 Cost: {cost:.2f}")
            print(f"   💸 Saved: {saved:.2f}")
            savings_pct = (saved / (saved + cost)) * 100 if (saved + cost) > 0 else 0.0
            print(f"   📉 Savings %: {savings_pct:.1f}%")
            if not implicit_chunks:
                print(f"   ⚡ Implicit Cache Hit: N/A (no implicit chunks)")
            elif hit_stop == len(implicit_chunks):
                print(f"   ⚡ Implicit Cache Hit: ❌ Full Miss")
            elif hit_stop is None:
                print(f"   ⚡ Implicit Cache Hit: ✅ Full")
            else:
                print(f"   ⚡ Implicit Cache Hit: ⚠️ Partial (miss at chunk #{hit_stop + 1} → {implicit_chunks[hit_stop]})")

        print(f"\n📦 Group Summary: Tokens={group_tokens}, Cost={group_cost:.2f}, Saved={group_saved:.2f}")

    net_cost = total_upload_cost + total_query_cost
    overall_savings_pct = (total_saved / (total_saved + net_cost)) * 100 if (total_saved + net_cost) > 0 else 0.0

    print("\n=== 📊 FINAL SUMMARY ===")
    print(f"Total Queries: {len(queries_raw)}")
    print(f"Total Tokens Processed: {total_tokens}")
    print(f"Total Upload Cost (Explicit Cache): {total_upload_cost:.2f}")
    print(f"Total Query Cost: {total_query_cost:.2f}")
    print(f"Total Saved via Cache: {total_saved:.2f}")
    print(f"Net Cost After Savings: {net_cost:.2f}")
    print(f"Overall Savings: {overall_savings_pct:.2f}%")

if __name__ == "__main__":
    main()

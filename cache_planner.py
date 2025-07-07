import json
from collections import Counter
import math

EXPLICIT_TOKEN_BUDGET = 400  # 80% of 10K budget
WORDS_PER_TOKEN = 1 / 0.75
TOKEN_COST = 1.0  # Normalize unit cost per token

def estimate_tokens(text: str) -> int:
    return math.ceil(len(text.split()) / WORDS_PER_TOKEN)

def load_chunks(path="chunks.json"):
    with open(path, "r") as f:
        chunks = json.load(f)
    return {cid: {"text": text, "tokens": estimate_tokens(text)} for cid, text in chunks.items()}

def load_queries(path="queries.json"):
    with open(path, "r") as f:
        return json.load(f)

def count_chunk_usage(queries):
    usage = Counter()
    for chunk_ids in queries.values():
        usage.update(chunk_ids)
    return usage

def select_explicit_chunks(chunks, usage, token_budget):
    scored = sorted(
        chunks.items(),
        key=lambda item: (usage[item[0]] * item[1]["tokens"]),
        reverse=True,
    )
    explicit = {}
    total_tokens = 0
    for chunk_id, data in scored:
        if total_tokens + data["tokens"] > token_budget:
            continue
        explicit[chunk_id] = data
        total_tokens += data["tokens"]
    return explicit

def assign_implicit_chunks(queries, explicit_chunks):
    implicit_plan = {}
    for query, chunk_list in queries.items():
        implicit_chunks = [cid for cid in chunk_list if cid not in explicit_chunks]
        implicit_plan[query] = implicit_chunks
    return implicit_plan

def sort_queries_by_overlap(queries, explicit_chunks):
    def overlap_score(chunk_list):
        return sum(1 for cid in chunk_list if cid in explicit_chunks)
    return sorted(queries.items(), key=lambda item: -overlap_score(item[1]))

def compute_costs(query_chunks, chunks, explicit_chunks, seen_chunks):
    full_token_cost = 0
    saved_token_cost = 0
    total_tokens = 0

    for cid in query_chunks:
        tokens = chunks[cid]["tokens"]
        total_tokens += tokens

        if cid in explicit_chunks:
            if cid in seen_chunks:
                cost = tokens * TOKEN_COST * 0.25  # 75% saved
                saved = tokens * TOKEN_COST * 0.75
            else:
                cost = tokens * TOKEN_COST
                saved = 0
        else:
            cost = tokens * TOKEN_COST
            saved = 0

        full_token_cost += cost
        saved_token_cost += saved

        seen_chunks.add(cid)  # Track chunk as seen

    return total_tokens, full_token_cost, saved_token_cost

def main():
    print("[🚀] Loading data...")
    chunks = load_chunks()
    queries = load_queries()
    usage = count_chunk_usage(queries)

    print(f"[ℹ️] Loaded {len(chunks)} chunks and {len(queries)} queries")

    print("[📦] Selecting explicit chunks...")
    explicit_chunks = select_explicit_chunks(chunks, usage, EXPLICIT_TOKEN_BUDGET)
    print(f"[✅] Selected {len(explicit_chunks)} chunks for explicit cache ({sum(c['tokens'] for c in explicit_chunks.values())} tokens)")

    print("[📑] Assigning implicit chunks per query...")
    implicit_plan = assign_implicit_chunks(queries, explicit_chunks)

    print("[🧠] Optimizing query execution order to minimize implicit cache swaps...")
    sorted_queries = sort_queries_by_overlap(queries, explicit_chunks)

    print("\n=== 🚧 CACHE PLAN ===")
    print("📌 Explicit Cache:")
    for cid in explicit_chunks:
        print(f"  - {cid}")

    print("\n📋 Per-query Plan with Cost Simulation:")
    seen_chunks = set()
    total_cost = 0
    total_saved = 0

    for query, chunk_list in sorted_queries:
        implicit_chunks = implicit_plan[query]
        all_chunks = chunk_list

        tokens, cost, saved = compute_costs(all_chunks, chunks, explicit_chunks, seen_chunks)
        total_cost += cost
        total_saved += saved

        print(f"\n🔍 Query: {query}")
        print("   Implicit Chunks:")
        if implicit_chunks:
            for cid in implicit_chunks:
                print(f"     - {cid}")
        else:
            print("     - None")
        print(f"   🔢 Tokens: {tokens}")
        print(f"   💰 Cost: {cost:.2f}")
        print(f"   💸 Saved via explicit: {saved:.2f}")
        print(f"   📉 Savings %: {((saved / (cost + saved)) * 100):.1f}%")

    print("\n=== 📊 TOTAL SUMMARY ===")
    print(f"Total Cost Across Queries: {total_cost:.2f}")
    print(f"Total Saved from Explicit Cache: {total_saved:.2f}")
    print(f"Overall Savings: {(total_saved / (total_cost + total_saved)) * 100:.2f}%")

if __name__ == "__main__":
    main()

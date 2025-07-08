import os
import json
import time
import logging
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from dotenv import load_dotenv


def plan_batches(
    chunk_texts_data,
    query_map_data,
    max_batch_size=5,
    implicit_threshold=1024,
    cache_discount=0.25,
    cache_ttl=3600,
    registry_path="implicit_registry.pkl",
    min_explicit_chunks=1,
    max_explicit_chunks=50,
    min_implicit_threshold=225,
    max_implicit_threshold=4096,
    min_cache_ttl=600,
    max_cache_ttl=86400,
    token_ratio=0.75,
    log_level=logging.WARNING
):
    """
    Simulate batching and caching of context chunks for API calls with constraint enforcement.

    Parameters:
    - chunk_texts_data: dict mapping chunk_id to text
    - query_map_data: dict mapping query text to list of chunk_ids
    - max_batch_size: max number of queries per batch
    - implicit_threshold: desired threshold for implicit cache (tokens)
    - cache_discount: fraction paid for explicit-cached tokens
    - cache_ttl: desired implicit cache TTL (seconds)
    - registry_path: file path to persist implicit cache registry
    - min_explicit_chunks: min core chunks to create explicit cache
    - max_explicit_chunks: max core chunks allowed for explicit cache
    - min_implicit_threshold: lower bound for implicit_threshold
    - max_implicit_threshold: upper bound for implicit_threshold
    - min_cache_ttl: lower bound for cache_ttl
    - max_cache_ttl: upper bound for cache_ttl
    - token_ratio: ratio of words to tokens
    - log_level: logging level
    """
    # Logging
    logging.basicConfig(level=log_level)
    now = time.time()

    # Enforce parameter constraints
    implicit_threshold = max(min_implicit_threshold, min(implicit_threshold, max_implicit_threshold))
    cache_ttl = max(min_cache_ttl, min(cache_ttl, max_cache_ttl))

    # Load or initialize implicit registry
    try:
        with open(registry_path, "rb") as f:
            implicit_registry = pickle.load(f)
    except Exception:
        implicit_registry = {}

    # Token estimation using configurable ratio
    def estimate_tokens(text: str) -> int:
        return round(len(text.strip().split()) / token_ratio)

    # Jaccard similarity for ordering batches
    def jaccard(a, b):
        a, b = set(a), set(b)
        return len(a & b) / len(a | b) if a or b else 0

    # Determine core (explicit) chunks
    freq = Counter()
    for chunks in query_map_data.values():
        freq.update(chunks)
    core_chunks = {cid for cid, f in freq.items() if f > 1}
    core_tokens = sum(estimate_tokens(chunk_texts_data[c]) for c in core_chunks)

    plan = {"explicit_cache": [], "batches": [], "cleanup": []}
    total_raw = 0
    total_opt = 0

    # Create explicit cache if within bounds
    num_core = len(core_chunks)
    if min_explicit_chunks <= num_core <= max_explicit_chunks:
        plan["explicit_cache"].append({
            "action": "create_explicit_cache",
            "chunk_ids": sorted(core_chunks),
            "ctx_tokens": core_tokens,
            "ttl": cache_ttl,
        })
        total_opt += core_tokens * cache_discount

    # Group queries by their dynamic chunk sets
    dynamic_map = defaultdict(list)
    for query, chunks in query_map_data.items():
        dynamic = tuple(sorted(set(chunks) - core_chunks))
        dynamic_map[dynamic].append(query)

    # Build and order batches
    batches = []
    for dyn, queries in dynamic_map.items():
        for i in range(0, len(queries), max_batch_size):
            batches.append({"dynamic_chunks": dyn, "queries": queries[i : i + max_batch_size]})
    ordered = []
    if batches:
        ordered.append(batches.pop(0))
        while batches:
            last = ordered[-1]["dynamic_chunks"]
            nxt = max(batches, key=lambda b: jaccard(last, b["dynamic_chunks"]))
            batches.remove(nxt)
            ordered.append(nxt)

    # Helper to simulate implicit cache hits
    def simulate_implicit(prefix: tuple) -> int:
        toks = sum(estimate_tokens(chunk_texts_data[c]) for c in prefix)
        last_time = implicit_registry.get(prefix)
        if last_time and (now - last_time) < cache_ttl:
            implicit_registry[prefix] = now
            return toks
        implicit_registry[prefix] = now
        return 0

    # Process each batch
    for gid, batch in enumerate(ordered, start=1):
        dyn = batch["dynamic_chunks"]
        dyn_toks = sum(estimate_tokens(chunk_texts_data[c]) for c in dyn)
        q_toks = sum(estimate_tokens(q) for q in batch["queries"])
        raw = core_tokens + dyn_toks + q_toks
        total_raw += raw

        # Implicit hits only if threshold met
        imp_hits = simulate_implicit(dyn) if dyn_toks >= implicit_threshold else 0
        exp_cost = core_tokens * cache_discount if min_explicit_chunks <= num_core <= max_explicit_chunks else 0
        uncached = max(0, dyn_toks - imp_hits)
        sent = uncached + q_toks
        opt_cost = sent + exp_cost
        total_opt += opt_cost

        save_pct = round((raw - opt_cost) / raw * 100, 1) if raw else 0.0

        plan["batches"].append({
            "action": "generate_content",
            "group_id": gid,
            "batch_queries": batch["queries"],
            "explicit_used": exp_cost > 0,
            "explicit_cost": round(exp_cost, 2),
            "implicit_hits": imp_hits,
            "dynamic_tokens": dyn_toks,
            "uncached_dynamic": uncached,
            "query_tokens": q_toks,
            "sent_tokens": sent,
            "batch_saving_pct": save_pct,
            "implicit_order": list(dyn) if imp_hits else [],
        })

    # Cleanup explicit cache
    if min_explicit_chunks <= num_core <= max_explicit_chunks:
        plan["cleanup"].append({
            "action": "delete_explicit_cache",
            "chunk_ids": sorted(core_chunks),
        })

    # Persist implicit registry
    try:
        with open(registry_path, "wb") as f:
            pickle.dump(implicit_registry, f)
    except Exception:
        logging.warning("Could not save implicit registry")

    # Summary
    total_saved = total_raw - total_opt
    plan["summary"] = {
        "total_raw_tokens": total_raw,
        "total_optimized_tokens": total_opt,
        "total_saved_tokens": total_saved,
        "saving_percentage": round((total_saved / total_raw) * 100, 1) if total_raw else 0.0,
    }

    return plan


if __name__ == "__main__":
    load_dotenv()
    import logging

    chunks_file = Path("chunks.json")
    queries_file = Path("queries.json")

    with open(chunks_file, "r") as cf:
        chunks_data = json.load(cf)
    with open(queries_file, "r") as qf:
        queries_data = json.load(qf)

    result = plan_batches(
        chunk_texts_data=chunks_data,
        query_map_data=queries_data,
        max_batch_size=7,
        implicit_threshold=2048,
        cache_discount=0.25,
        cache_ttl=3600,
        registry_path="implicit_registry.pkl",
        min_explicit_chunks=1,
        max_explicit_chunks=50,
        min_implicit_threshold=225,
        max_implicit_threshold=4096,
        min_cache_ttl=600,
        max_cache_ttl=86400,
        token_ratio=0.75,
        log_level=logging.WARNING,
    )

    print(json.dumps(result, indent=2))

import os
import json
import time
import pickle
import argparse
import logging
import asyncio
from math import ceil
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Any

from dotenv import load_dotenv
from google import genai

# === CONFIGURATION ===
@dataclass
class Config:
    chunks_path: Path = Path("chunks.json")
    queries_path: Path = Path("queries.json")
    registry_path: Path = Path("implicit_registry.pkl")
    model: str = "gemini-2.0-flash"
    max_batch_size: int = 5
    cache_discount: float = 0.25       # pay 25% for explicit‐cached tokens
    implicit_threshold: int = 1024     # min tokens for implicit dynamic caching
    cache_ttl: int = 3600              # seconds (simulated TTL)
    log_level: int = logging.WARNING

config = Config()

# === SETUP ===
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
logging.basicConfig(level=config.log_level, format="%(asctime)s %(levelname)s:%(message)s")

# === ASYNC TOKEN COUNTER ===
async def count_tokens_async(texts: List[str]) -> Dict[str, int]:
    loop = asyncio.get_running_loop()
    async def worker(txt: str) -> Tuple[str, int]:
        resp = await loop.run_in_executor(
            None,
            lambda: client.models.count_tokens(model=config.model, contents=txt)
        )
        if hasattr(resp, "total_tokens"):
            cnt = resp.total_tokens
        elif hasattr(resp, "total_token_count"):
            cnt = resp.total_token_count
        else:
            cnt = int(resp)
        return txt, cnt

    tasks = [worker(t) for t in texts]
    results = await asyncio.gather(*tasks)
    return {txt: cnt for txt, cnt in results}

# === PLANNER HELPERS ===
def jaccard(a: Tuple[str, ...], b: Tuple[str, ...]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0

def compute_core_chunks(query_map: Dict[str, List[str]]) -> Set[str]:
    from collections import Counter
    freq = Counter(cid for chunks in query_map.values() for cid in chunks)
    return {cid for cid, cnt in freq.items() if cnt > 1}

def build_batches(query_map: Dict[str, List[str]], core: Set[str]) -> List[Dict[str, Any]]:
    from collections import defaultdict
    groups: Dict[Tuple[str,...], List[str]] = defaultdict(list)
    for q, chunks in query_map.items():
        dyn = tuple(sorted(set(chunks) - core))
        groups[dyn].append(q)
    batches = []
    for dyn, qs in groups.items():
        for i in range(0, len(qs), config.max_batch_size):
            batches.append({"dynamic_chunks": dyn, "queries": qs[i:i+config.max_batch_size]})
    return batches

def order_batches(batches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not batches:
        return []
    remaining = batches.copy()
    ordered = [remaining.pop(0)]
    while remaining:
        last = ordered[-1]["dynamic_chunks"]
        nxt = max(
            remaining,
            key=lambda b: jaccard(last, b["dynamic_chunks"]) *
                          sum(chunk_tokens[c] for c in b["dynamic_chunks"])
        )
        remaining.remove(nxt)
        ordered.append(nxt)
    return ordered

# === SIMULATOR ===
@dataclass
class Simulator:
    chunk_data: Dict[str, str]
    query_map: Dict[str, List[str]]
    core_chunks: Set[str]
    chunk_tokens: Dict[str, int]
    query_tokens: Dict[str, int]
    now: float = field(default_factory=time.time)
    implicit_registry: Dict[Tuple[str,...], float] = field(default_factory=dict)
    plan: List[Dict[str, Any]] = field(default_factory=list)
    total_raw: int = 0
    total_opt: float = 0.0

    def load_registry(self):
        if config.registry_path.exists():
            with open(config.registry_path, "rb") as f:
                self.implicit_registry = pickle.load(f)

    def save_registry(self):
        with open(config.registry_path, "wb") as f:
            pickle.dump(self.implicit_registry, f)

    def simulate_implicit(self, prefix: Tuple[str,...]) -> int:
        tok = sum(self.chunk_tokens[c] for c in prefix)
        last = self.implicit_registry.get(prefix)
        self.implicit_registry[prefix] = self.now
        return tok if last and (self.now - last) < config.cache_ttl else 0

    def should_explicit(self, batch_count: int, core_tok: int) -> bool:
        return batch_count * (1 - config.cache_discount) > 1

    def run(self):
        self.load_registry()
        core_tok = sum(self.chunk_tokens[c] for c in self.core_chunks)
        batches = build_batches(self.query_map, self.core_chunks)
        ordered = order_batches(batches)

        use_explicit = False
        if self.core_chunks and self.should_explicit(len(ordered), core_tok):
            self.plan.append({
                "action":    "create_explicit_cache",
                "chunk_ids": sorted(self.core_chunks),
                "ctx_tokens": core_tok,
                "ttl":       config.cache_ttl
            })
            self.total_opt += core_tok
            use_explicit = True

        for gid, batch in enumerate(ordered, start=1):
            dyn = batch["dynamic_chunks"]
            dyn_tok = sum(self.chunk_tokens[c] for c in dyn)
            q_tok = sum(self.query_tokens[q] for q in batch["queries"])
            raw = (core_tok if use_explicit else core_tok + dyn_tok) + q_tok
            self.total_raw += raw

            imp = self.simulate_implicit(dyn) if dyn_tok >= config.implicit_threshold else 0
            exp_cost = core_tok * config.cache_discount if use_explicit else 0
            unc = max(0, dyn_tok - imp)
            sent = unc + q_tok
            opt = sent + exp_cost
            self.total_opt += opt

            pct = 100 * (raw - opt) / raw if raw else 0
            self.plan.append({
                "action":           "generate_content",
                "group_id":         gid,
                "batch_queries":    batch["queries"],
                "explicit_used":    use_explicit,
                "explicit_cost":    exp_cost,
                "implicit_hits":    imp,
                "dynamic_tokens":   dyn_tok,
                "uncached_dynamic": unc,
                "query_tokens":     q_tok,
                "sent_tokens":      sent,
                "batch_saving_pct": round(pct, 1),
                "implicit_order":   dyn  # record the order of implicit chunks
            })

        if use_explicit:
            self.plan.append({
                "action":    "delete_explicit_cache",
                "chunk_ids": sorted(self.core_chunks)
            })
        self.save_registry()

    def report(self):
        saved = self.total_raw - self.total_opt
        pct = 100 * saved / self.total_raw if self.total_raw else 0
        logging.info(f"FINAL: raw={self.total_raw}, opt={self.total_opt:.2f}, saved={saved} tokens ({pct:.1f}%)")

# === MAIN ===
async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-batch-size", type=int)
    parser.add_argument("--implicit-threshold", type=int)
    args = parser.parse_args()

    if args.max_batch_size:
        config.max_batch_size = args.max_batch_size
    if args.implicit_threshold:
        config.implicit_threshold = args.implicit_threshold

    chunks = json.loads(config.chunks_path.read_text())
    queries = json.loads(config.queries_path.read_text())
    core = compute_core_chunks(queries)

    raw_chunks = [chunks[cid] for cid in chunks]
    chunk_map = await count_tokens_async(raw_chunks)
    global chunk_tokens
    chunk_tokens = {cid: chunk_map[chunks[cid]] for cid in chunks}

    query_keys = list(queries.keys())
    query_map_counts = await count_tokens_async(query_keys)
    global query_tokens
    query_tokens = {q: query_map_counts[q] for q in queries}

    sim = Simulator(chunks, queries, core, chunk_tokens, query_tokens)
    sim.run()
    sim.report()
    return {
        "explicit_cache": [step for step in sim.plan if step["action"] == "create_explicit_cache"],
        "batches": [step for step in sim.plan if step["action"] == "generate_content"],
        "cleanup": [step for step in sim.plan if step["action"] == "delete_explicit_cache"],
    }

if __name__ == "__main__":
    # Optionally override defaults here, or supply flags instead
    # config.max_batch_size = 7
    # config.implicit_threshold = 2048

    plan_output = asyncio.run(main())
    # Print full JSON plan, including explicit, implicit order, and queries
    print(json.dumps(plan_output, indent=2))

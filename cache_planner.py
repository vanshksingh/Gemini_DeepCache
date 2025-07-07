import os
import json
import time
import logging
import pickle
from math import ceil
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set

from dotenv import load_dotenv
from google import genai

# === CONFIG ===

@dataclass
class Config:
    chunks_path: Path = Path("chunks.json")
    queries_path: Path = Path("queries.json")
    registry_path: Path = Path("implicit_registry.pkl")
    max_batch_size: int = 5
    cache_discount: float = 0.25       # pay 25% for explicit‐cached tokens
    implicit_threshold: int = 1024     # min tokens for implicit dynamic caching
    cache_ttl: int = 3600              # seconds (simulated TTL)
    log_level: int = logging.INFO

config = Config()

# === SETUP ===

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

logging.basicConfig(level=config.log_level, format="%(asctime)s %(levelname)s:%(message)s")

# === HELPERS ===

_token_cache: Dict[str, int] = {}

def estimate_tokens(text: str) -> int:
    if text not in _token_cache:
        _token_cache[text] = ceil(len(text.strip().split()) / 0.75)
    return _token_cache[text]

def jaccard(a: Tuple[str, ...], b: Tuple[str, ...]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0

# === CORE ===

def compute_core_chunks(query_map: Dict[str, List[str]]) -> Set[str]:
    from collections import Counter
    freq = Counter(cid for chunks in query_map.values() for cid in chunks)
    return {cid for cid, cnt in freq.items() if cnt > 1}

def build_batches(query_map: Dict[str, List[str]], core: Set[str]) -> List[Dict]:
    from collections import defaultdict
    dynamic_groups: Dict[Tuple[str,...], List[str]] = defaultdict(list)
    for q, chunks in query_map.items():
        dyn = tuple(sorted(set(chunks) - core))
        dynamic_groups[dyn].append(q)
    batches = []
    for dyn, qs in dynamic_groups.items():
        for i in range(0, len(qs), config.max_batch_size):
            batches.append({"dynamic_chunks": dyn, "queries": qs[i:i+config.max_batch_size]})
    return batches

def order_batches(batches: List[Dict]) -> List[Dict]:
    if not batches:
        return []
    remaining = batches.copy()
    ordered = [remaining.pop(0)]
    while remaining:
        last = ordered[-1]["dynamic_chunks"]
        def score(b):
            overlap = jaccard(last, b["dynamic_chunks"])
            tok = sum(estimate_tokens(c) for c in b["dynamic_chunks"])
            return overlap * tok
        nxt = max(remaining, key=score)
        remaining.remove(nxt)
        ordered.append(nxt)
    return ordered

# === SIMULATOR ===

@dataclass
class Simulator:
    chunk_data: Dict[str, str]
    query_map: Dict[str, List[str]]
    core_chunks: Set[str]
    now: float = field(default_factory=time.time)
    implicit_registry: Dict[Tuple[str,...], float] = field(default_factory=dict)
    plan: List[Dict] = field(default_factory=list)
    total_raw: int = 0
    total_opt: float = 0.0

    def load_registry(self):
        if config.registry_path.exists():
            with open(config.registry_path, "rb") as f:
                self.implicit_registry = pickle.load(f)

    def save_registry(self):
        with open(config.registry_path, "wb") as f:
            pickle.dump(self.implicit_registry, f)

    def simulate_implicit(self, prefix: Tuple[str,...], tok: int) -> int:
        last = self.implicit_registry.get(prefix)
        self.implicit_registry[prefix] = self.now
        if last and (self.now - last) < config.cache_ttl:
            return tok
        return 0

    def should_explicit(self, batch_count: int, core_tok: int) -> bool:
        # break-even check: total save > cost of creating cache
        # saving per batch = core_tok*(1 - discount)
        return batch_count * (1 - config.cache_discount) > 1

    def run(self):
        # load persistent registry
        self.load_registry()

        # precompute
        batches = build_batches(self.query_map, self.core_chunks)
        ordered = order_batches(batches)
        core_tok = sum(estimate_tokens(self.chunk_data[c]) for c in self.core_chunks)

        # maybe create explicit cache
        if self.core_chunks and self.should_explicit(len(ordered), core_tok):
            self.plan.append({
                "action": "create_explicit_cache",
                "chunk_ids": sorted(self.core_chunks),
                "ctx_tokens": core_tok,
                "ttl": config.cache_ttl
            })
            self.total_opt += core_tok
            use_explicit = True
        else:
            logging.info("Skipping explicit cache (not cost-effective).")
            use_explicit = False

        # per-batch sim
        for gid, batch in enumerate(ordered, start=1):
            dyn = batch["dynamic_chunks"]
            dyn_tok = sum(estimate_tokens(self.chunk_data[c]) for c in dyn)
            qry_tok = sum(estimate_tokens(q) for q in batch["queries"])
            raw = (core_tok if use_explicit else core_tok + dyn_tok) + qry_tok
            self.total_raw += raw

            imp_hits = self.simulate_implicit(dyn, dyn_tok) if dyn_tok >= config.implicit_threshold else 0
            exp_cost = core_tok * config.cache_discount if use_explicit else 0
            uncached_dyn = max(0, dyn_tok - imp_hits)
            sent = uncached_dyn + qry_tok
            opt = sent + exp_cost
            self.total_opt += opt

            pct = 100*(raw - opt)/raw if raw else 0
            logging.info(
                "Batch %d: raw=%d, opt=%.2f (%.1f%% saving)",
                gid, raw, opt, pct
            )

            self.plan.append({
                "action": "generate_content",
                "group_id": gid,
                "batch_queries": batch["queries"],
                "explicit_used": use_explicit,
                "explicit_cost": exp_cost,
                "implicit_hits": imp_hits,
                "dynamic_tokens": dyn_tok,
                "uncached_dynamic": uncached_dyn,
                "query_tokens": qry_tok,
                "sent_tokens": sent,
                "batch_saving_pct": round(pct,1)
            })

        # delete explicit if created
        if use_explicit:
            self.plan.append({
                "action": "delete_explicit_cache",
                "chunk_ids": sorted(self.core_chunks)
            })

        # persist registry
        self.save_registry()

    def report(self):
        savings = self.total_raw - self.total_opt
        pct = 100 * savings / self.total_raw if self.total_raw else 0
        logging.info("📊 FINAL SUMMARY: raw=%d, opt=%.2f, saved=%d tokens (%.2f%%)",
                     self.total_raw, self.total_opt, savings, pct)

def main():
    chunks = json.loads(config.chunks_path.read_text())
    queries = json.loads(config.queries_path.read_text())
    core = compute_core_chunks(queries)

    sim = Simulator(chunk_data=chunks, query_map=queries, core_chunks=core)
    sim.run()
    sim.report()

if __name__ == "__main__":
    main()

import os
import json
import time
import pickle
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any

from dotenv import load_dotenv
from google import genai

# === CONFIGURATION ===
class Config:
    def __init__(self):
        self.model: str = "gemini-2.0-flash"
        self.max_batch_size: int = 5
        self.cache_discount: float = 0.25
        self.implicit_threshold: int = 1024
        self.cache_ttl: int = 3600
        self.registry_path: Path = Path("implicit_registry.pkl")
        self.log_level = logging.WARNING

config = Config()
client: genai.Client  # set in plan_batches()

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

# === PLANNING HELPERS ===
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
            batches.append({
                "dynamic_chunks": dyn,
                "queries": qs[i:i+config.max_batch_size]
            })
    return batches

def order_batches(
    batches: List[Dict[str, Any]],
    chunk_tokens: Dict[str, int]
) -> List[Dict[str, Any]]:
    if not batches:
        return []
    remaining = batches.copy()
    ordered = [remaining.pop(0)]
    while remaining:
        last = ordered[-1]["dynamic_chunks"]
        # pick the next batch that maximizes overlap * weight
        nxt = max(
            remaining,
            key=lambda b: jaccard(last, b["dynamic_chunks"]) *
                          sum(chunk_tokens[c] for c in b["dynamic_chunks"])
        )
        remaining.remove(nxt)
        ordered.append(nxt)
    return ordered

# === SIMULATOR ===
class Simulator:
    def __init__(
        self,
        chunk_data: Dict[str, str],
        query_map: Dict[str, List[str]],
        core_chunks: Set[str],
        chunk_tokens: Dict[str, int],
        query_tokens: Dict[str, int],
        min_explicit: int,
        max_explicit: int
    ):
        self.chunk_data = chunk_data
        self.query_map = query_map
        self.core_chunks = core_chunks
        self.chunk_tokens = chunk_tokens
        self.query_tokens = query_tokens
        self.implicit_registry: Dict[Tuple[str,...], float] = {}
        self.plan: List[Dict[str, Any]] = []
        self.total_raw = 0
        self.total_opt = 0
        self.min_explicit = min_explicit
        self.max_explicit = max_explicit

    def load_registry(self):
        if config.registry_path.exists():
            with open(config.registry_path, "rb") as f:
                self.implicit_registry = pickle.load(f)

    def save_registry(self):
        with open(config.registry_path, "wb") as f:
            pickle.dump(self.implicit_registry, f)

    def simulate_implicit(self, prefix: Tuple[str,...]) -> int:
        # ensure prefix is canonical
        prefix = tuple(sorted(prefix))
        tok = sum(self.chunk_tokens[c] for c in prefix)
        now = time.time()
        last = self.implicit_registry.get(prefix)
        # update registry to now
        self.implicit_registry[prefix] = now
        # if we saw it recently, we get a free hit
        return tok if last and (now - last) < config.cache_ttl else 0

    def should_explicit(self, batch_count: int) -> bool:
        core_count = len(self.core_chunks)
        # only if core size in allowed range
        if not (self.min_explicit <= core_count <= self.max_explicit):
            return False
        # simple heuristic: if there are enough batches to amortize
        return batch_count * (1 - config.cache_discount) > 1

    def run(self):
        self.load_registry()
        core_tok = sum(self.chunk_tokens[c] for c in self.core_chunks)
        batches = build_batches(self.query_map, self.core_chunks)
        ordered = order_batches(batches, self.chunk_tokens)

        use_explicit = False
        if self.core_chunks and self.should_explicit(len(ordered)):
            # create explicit cache once
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
            uncached = max(0, dyn_tok - imp)
            sent = uncached + q_tok
            opt = sent + exp_cost
            self.total_opt += opt

            pct = 100 * (raw - opt) / raw if raw else 0.0
            self.plan.append({
                "action":           "generate_content",
                "group_id":         gid,
                "batch_queries":    batch["queries"],
                "explicit_used":    use_explicit,
                "explicit_cost":    exp_cost,
                "implicit_hits":    imp,
                "dynamic_tokens":   dyn_tok,
                "uncached_dynamic": uncached,
                "query_tokens":     q_tok,
                "sent_tokens":      sent,
                "batch_saving_pct": round(pct, 1),
                "implicit_order":   dyn
            })

        if use_explicit:
            # clean up explicit cache at end
            self.plan.append({
                "action":    "delete_explicit_cache",
                "chunk_ids": sorted(self.core_chunks)
            })
        self.save_registry()

    def report(self):
        saved = self.total_raw - self.total_opt
        pct = 100 * saved / self.total_raw if self.total_raw else 0.0
        logging.info(f"FINAL: raw={self.total_raw}, opt={self.total_opt:.2f}, saved={saved} tokens ({pct:.1f}%)")

# === CALLABLE FUNCTION ===
def plan_batches(
    chunks: Dict[str, str],
    queries: Dict[str, List[str]],
    api_key: str,
    *,
    model: str = "gemini-2.0-flash",
    max_batch_size: int = 5,
    implicit_threshold: int = 1024,
    cache_discount: float = 0.25,
    cache_ttl: int = 3600,
    registry_path: str = "implicit_registry.pkl",
    min_explicit_chunks: int = 1,
    max_explicit_chunks: int = 1000,
    min_implicit_threshold: int = 1,
    max_implicit_threshold: int = 100000,
    min_cache_ttl: int = 1,
    max_cache_ttl: int = 86400,
    log_level: int = logging.WARNING
) -> Dict[str, Any]:
    # Validate constraints
    if not (min_implicit_threshold <= implicit_threshold <= max_implicit_threshold):
        raise ValueError(f"implicit_threshold must be between {min_implicit_threshold} and {max_implicit_threshold}")
    if not (min_cache_ttl <= cache_ttl <= max_cache_ttl):
        raise ValueError(f"cache_ttl must be between {min_cache_ttl} and {max_cache_ttl}")

    # Override globals
    config.model = model
    config.max_batch_size = max_batch_size
    config.cache_discount = cache_discount
    config.implicit_threshold = implicit_threshold
    config.cache_ttl = cache_ttl
    config.registry_path = Path(registry_path)
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s:%(message)s")

    global client
    # you can still do load_dotenv() + os.getenv("GEMINI_API_KEY") here
    client = genai.Client(api_key=api_key)

    async def _inner() -> Dict[str, Any]:
        # Compute core chunks
        core = compute_core_chunks(queries)

        # Count chunk tokens
        raw_chunks = [chunks[cid] for cid in chunks]
        chunk_map = await count_tokens_async(raw_chunks)
        chunk_tokens = {cid: chunk_map[chunks[cid]] for cid in chunks}

        # Count query tokens
        qkeys = list(queries.keys())
        qmap = await count_tokens_async(qkeys)
        query_tokens = {q: qmap[q] for q in queries}

        # Run simulator
        sim = Simulator(
            chunks, queries, core,
            chunk_tokens, query_tokens,
            min_explicit_chunks, max_explicit_chunks
        )
        sim.run()
        sim.report()

        saved = sim.total_raw - sim.total_opt
        pct_saved = 100 * saved / sim.total_raw if sim.total_raw else 0.0

        return {
            "explicit_cache": [s for s in sim.plan if s["action"] == "create_explicit_cache"],
            "batches":        [s for s in sim.plan if s["action"] == "generate_content"],
            "cleanup":        [s for s in sim.plan if s["action"] == "delete_explicit_cache"],
            "summary": {
                "total_raw_tokens":       sim.total_raw,
                "total_optimized_tokens": sim.total_opt,
                "total_saved_tokens":     saved,
                "saving_percentage":      round(pct_saved, 1)
            }
        }

    return asyncio.run(_inner())

# === USAGE EXAMPLE ===
if __name__ == "__main__":
    load_dotenv()
    with open("chunks.json") as f:
        chunks = json.load(f)
    with open("queries.json") as f:
        queries = json.load(f)

    plan = plan_batches(
        chunks,
        queries,
        api_key=os.getenv("GEMINI_API_KEY") or "<YOUR_API_KEY>",
        model="gemini-2.0-flash",
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
        log_level=logging.WARNING
    )

    print(json.dumps(plan, indent=2))

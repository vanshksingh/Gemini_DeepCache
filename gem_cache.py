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
    async def worker(text: str) -> Tuple[str, int]:
        response = await loop.run_in_executor(
            None,
            lambda: client.models.count_tokens(model=config.model, contents=text)
        )
        if hasattr(response, "total_tokens"):
            token_count = response.total_tokens
        elif hasattr(response, "total_token_count"):
            token_count = response.total_token_count
        else:
            token_count = int(response)
        return text, token_count

    tasks = [worker(txt) for txt in texts]
    results = await asyncio.gather(*tasks)
    return {txt: cnt for txt, cnt in results}

# === PLANNING HELPERS ===
def jaccard_similarity(a: Tuple[str, ...], b: Tuple[str, ...]) -> float:
    set_a, set_b = set(a), set(b)
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 0.0


def compute_core_chunks(query_map: Dict[str, List[str]]) -> Set[str]:
    from collections import Counter
    freq = Counter(chunk_id for chunk_list in query_map.values() for chunk_id in chunk_list)
    return {cid for cid, count in freq.items() if count > 1}


def build_batches(query_map: Dict[str, List[str]], core_set: Set[str]) -> List[Dict[str, Any]]:
    from collections import defaultdict
    grouping: Dict[Tuple[str, ...], List[str]] = defaultdict(list)
    for query_text, chunk_list in query_map.items():
        dynamic = tuple(sorted(set(chunk_list) - core_set))
        grouping[dynamic].append(query_text)

    batch_list: List[Dict[str, Any]] = []
    for dynamic_chunks, queries_group in grouping.items():
        for start_index in range(0, len(queries_group), config.max_batch_size):
            batch_list.append({
                "dynamic_chunks": dynamic_chunks,
                "queries": queries_group[start_index:start_index + config.max_batch_size]
            })
    return batch_list


def order_batches(
    batches: List[Dict[str, Any]],
    token_map: Dict[str, int]
) -> List[Dict[str, Any]]:
    if not batches:
        return []
    remaining = batches.copy()
    ordered = [remaining.pop(0)]
    while remaining:
        last_chunks = ordered[-1]["dynamic_chunks"]
        next_batch = max(
            remaining,
            key=lambda b: jaccard_similarity(last_chunks, b["dynamic_chunks"]) *
                          sum(token_map[c] for c in b["dynamic_chunks"])
        )
        remaining.remove(next_batch)
        ordered.append(next_batch)
    return ordered

# === SIMULATOR ===
class Simulator:
    def __init__(
        self,
        chunk_texts: Dict[str, str],
        query_map: Dict[str, List[str]],
        core_chunks: Set[str],
        chunk_token_map: Dict[str, int],
        query_token_map: Dict[str, int],
        min_explicit: int,
        max_explicit: int
    ):
        self.chunk_texts = chunk_texts
        self.query_map = query_map
        self.core_chunks = core_chunks
        self.chunk_token_map = chunk_token_map
        self.query_token_map = query_token_map
        self.implicit_registry: Dict[Tuple[str, ...], float] = {}
        self.plan: List[Dict[str, Any]] = []
        self.total_raw = 0
        self.total_opt = 0
        self.min_explicit = min_explicit
        self.max_explicit = max_explicit

    def load_registry(self) -> None:
        if config.registry_path.exists():
            with open(config.registry_path, "rb") as registry_file:
                self.implicit_registry = pickle.load(registry_file)

    def save_registry(self) -> None:
        with open(config.registry_path, "wb") as registry_file:
            pickle.dump(self.implicit_registry, registry_file)

    def simulate_implicit(self, prefix: Tuple[str, ...]) -> int:
        prefix = tuple(sorted(prefix))
        token_sum = sum(self.chunk_token_map[c] for c in prefix)
        now = time.time()
        last_seen = self.implicit_registry.get(prefix)
        self.implicit_registry[prefix] = now
        return token_sum if last_seen and (now - last_seen) < config.cache_ttl else 0

    def should_use_explicit(self, batch_count: int) -> bool:
        core_count = len(self.core_chunks)
        if not (self.min_explicit <= core_count <= self.max_explicit):
            return False
        return batch_count * (1 - config.cache_discount) > 1

    def run(self) -> None:
        self.load_registry()
        core_tokens = sum(self.chunk_token_map[c] for c in self.core_chunks)
        batches = build_batches(self.query_map, self.core_chunks)
        ordered_batches = order_batches(batches, self.chunk_token_map)

        explicit_enabled = False
        if self.core_chunks and self.should_use_explicit(len(ordered_batches)):
            self.plan.append({
                "action": "create_explicit_cache",
                "chunk_ids": sorted(self.core_chunks),
                "ctx_tokens": core_tokens,
                "ttl": config.cache_ttl
            })
            self.total_opt += core_tokens
            explicit_enabled = True

        for group_id, batch in enumerate(ordered_batches, start=1):
            dynamic = batch["dynamic_chunks"]
            dynamic_tokens = sum(self.chunk_token_map[c] for c in dynamic)
            query_tokens = sum(self.query_token_map[q] for q in batch["queries"])

            raw = (core_tokens if explicit_enabled else core_tokens + dynamic_tokens) + query_tokens
            self.total_raw += raw

            implicit_hit = self.simulate_implicit(dynamic) if dynamic_tokens >= config.implicit_threshold else 0
            explicit_cost = core_tokens * config.cache_discount if explicit_enabled else 0
            uncached_dyn = max(0, dynamic_tokens - implicit_hit)
            sent = uncached_dyn + query_tokens
            optimized = sent + explicit_cost
            self.total_opt += optimized

            saving_pct = round(100 * (raw - optimized) / raw, 1) if raw else 0.0
            self.plan.append({
                "action": "generate_content",
                "group_id": group_id,
                "batch_queries": batch["queries"],
                "explicit_used": explicit_enabled,
                "explicit_cost": explicit_cost,
                "implicit_hits": implicit_hit,
                "dynamic_tokens": dynamic_tokens,
                "uncached_dynamic": uncached_dyn,
                "query_tokens": query_tokens,
                "sent_tokens": sent,
                "batch_saving_pct": saving_pct,
                "implicit_order": dynamic
            })

        if explicit_enabled:
            self.plan.append({
                "action": "delete_explicit_cache",
                "chunk_ids": sorted(self.core_chunks)
            })
        self.save_registry()

    def report(self) -> None:
        saved = self.total_raw - self.total_opt
        pct = round(100 * saved / self.total_raw, 1) if self.total_raw else 0.0
        logging.info(f"FINAL: raw={self.total_raw}, opt={self.total_opt:.2f}, saved={saved} tokens ({pct}%)")

# === CALLABLE FUNCTION ===
def plan_batches(
    chunk_texts: Dict[str, str],
    query_map: Dict[str, List[str]],
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
        raise ValueError(
            f"implicit_threshold must be between {min_implicit_threshold} and {max_implicit_threshold}"
        )
    if not (min_cache_ttl <= cache_ttl <= max_cache_ttl):
        raise ValueError(
            f"cache_ttl must be between {min_cache_ttl} and {max_cache_ttl}"
        )

    # Override global config
    config.model = model
    config.max_batch_size = max_batch_size
    config.cache_discount = cache_discount
    config.implicit_threshold = implicit_threshold
    config.cache_ttl = cache_ttl
    config.registry_path = Path(registry_path)
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s: %(message)s")

    global client
    client = genai.Client(api_key=api_key)

    async def _inner() -> Dict[str, Any]:
        core = compute_core_chunks(query_map)

        # count tokens for chunks
        all_texts = list(chunk_texts.values())
        token_counts = await count_tokens_async(all_texts)
        chunk_token_map = {key: token_counts[text] for key, text in chunk_texts.items()}

        # count tokens for queries
        query_keys = list(query_map.keys())
        query_counts = await count_tokens_async(query_keys)
        query_token_map = {q: query_counts[q] for q in query_map}

        sim = Simulator(
            chunk_texts,
            query_map,
            core,
            chunk_token_map,
            query_token_map,
            min_explicit_chunks,
            max_explicit_chunks
        )
        sim.run()
        sim.report()

        saved = sim.total_raw - sim.total_opt
        percent_saved = round(100 * saved / sim.total_raw, 1) if sim.total_raw else 0.0

        return {
            "explicit_cache": [step for step in sim.plan if step["action"] == "create_explicit_cache"],
            "batches":        [step for step in sim.plan if step["action"] == "generate_content"],
            "cleanup":        [step for step in sim.plan if step["action"] == "delete_explicit_cache"],
            "summary": {
                "total_raw_tokens":       sim.total_raw,
                "total_optimized_tokens": sim.total_opt,
                "total_saved_tokens":     saved,
                "saving_percentage":      percent_saved
            }
        }

    return asyncio.run(_inner())

# === USAGE EXAMPLE ===
if __name__ == "__main__":
    load_dotenv()
    chunks_file_path = Path("chunks.json")
    queries_file_path = Path("queries.json")

    with open(chunks_file_path, "r") as chunk_file:
        chunk_texts_data = json.load(chunk_file)
    with open(queries_file_path, "r") as query_file:
        query_map_data = json.load(query_file)

    result_plan = plan_batches(
        chunk_texts_data,
        query_map_data,
        api_key=os.getenv("GEMINI_API_KEY", "<YOUR_API_KEY>"),
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

    print(json.dumps(result_plan, indent=2))

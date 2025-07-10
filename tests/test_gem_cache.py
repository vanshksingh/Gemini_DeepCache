import pytest
import logging
from unittest.mock import patch, MagicMock

from gem_cache import (
    jaccard_similarity,
    compute_core_chunks,
    build_batches,
    order_batches,
    Simulator,
    plan_batches,
)


# === Jaccard Similarity ===

def test_jaccard_similarity_basic():
    a = ("a", "b", "c")
    b = ("b", "c", "d")
    assert jaccard_similarity(a, b) == 0.5

def test_jaccard_similarity_empty_union():
    assert jaccard_similarity((), ()) == 0.0


# === Batch Planning ===

def test_compute_core_chunks():
    query_map = {"q1": ["c1", "c2"], "q2": ["c2", "c3"]}
    core = compute_core_chunks(query_map)
    assert core == {"c2"}


def test_build_batches_and_order():
    query_map = {
        "q1": ["c1", "c2", "c3"],
        "q2": ["c3"],
        "q3": ["c4", "c5"],
    }
    core = {"c3"}
    batches = build_batches(query_map, core)

    # Expected: 3 distinct dynamic chunk groups: ("c1", "c2"), (), ("c4", "c5")
    assert len(batches) == 3
    assert any(batch["dynamic_chunks"] == () for batch in batches)
    assert set(batch["dynamic_chunks"] for batch in batches) == {
        ("c1", "c2"), (), ("c4", "c5")
    }

    token_map = {"c1": 1, "c2": 1, "c4": 1, "c5": 1}
    ordered = order_batches(batches, token_map)
    assert isinstance(ordered, list)
    assert len(ordered) == 3


# === Simulator Behavior ===

def test_simulator_run():
    chunk_texts = {"c1": "text1", "c2": "text2", "c3": "text3"}
    query_map = {"q1": ["c1", "c2"], "q2": ["c2", "c3"]}
    core_chunks = {"c2"}
    chunk_token_map = {"c1": 10, "c2": 20, "c3": 30}
    query_token_map = {"q1": 5, "q2": 6}

    sim = Simulator(
        chunk_texts,
        query_map,
        core_chunks,
        chunk_token_map,
        query_token_map,
        min_explicit=1,
        max_explicit=3
    )

    sim.implicit_registry = {}  # Prevent I/O
    sim.save_registry = lambda: None
    sim.load_registry = lambda: None

    sim.run()
    assert sim.total_raw > 0
    assert sim.total_opt > 0
    assert len(sim.plan) >= 1


# === Full Pipeline ===

@patch("gem_cache.genai.Client")
@patch("gem_cache.count_tokens_async")
def test_plan_batches_minimal(mock_count_tokens, mock_client):
    mock_client.return_value = MagicMock()
    mock_count_tokens.side_effect = lambda texts: {txt: 10 for txt in texts}

    chunk_texts = {"c1": "hello world", "c2": "another one"}
    query_map = {"q1": ["c1", "c2"]}

    result = plan_batches(
        chunk_texts,
        query_map,
        api_key="fake-key",
        model="gemini-2.0-flash",
        max_batch_size=2,
        implicit_threshold=5,
        cache_discount=0.25,
        cache_ttl=3600,
        registry_path="test_registry.pkl",
        min_explicit_chunks=1,
        max_explicit_chunks=50,
        min_implicit_threshold=1,
        max_implicit_threshold=10000,
        min_cache_ttl=1,
        max_cache_ttl=86400,
        log_level=logging.CRITICAL
    )

    assert "summary" in result
    assert result["summary"]["total_saved_tokens"] >= 0

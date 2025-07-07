import numpy as np
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from fuzzywuzzy import fuzz
import warnings

warnings.filterwarnings("ignore")

def fuzzy_grouping_adaptive(
    queries: List[str],
    min_group_size: int = 4,
    max_threshold: int = 90,
    min_threshold: int = 10,
    step: int = 5,
    verbose: bool = False
) -> List[List[str]]:
    if verbose:
        print("[DEBUG] 🔍 Starting fuzzy grouping with adaptive threshold...")

    for threshold in range(max_threshold, min_threshold - 1, -step):
        groups, used = [], set()
        if verbose:
            print(f"   → Trying fuzzy threshold: {threshold}")
        for i, q1 in enumerate(queries):
            if i in used:
                continue
            group = [q1]
            used.add(i)
            for j, q2 in enumerate(queries):
                if j in used:
                    continue
                score = fuzz.token_set_ratio(q1, q2)
                if verbose:
                    print(f"     - '{q1}' vs '{q2}' → {score}")
                if score >= threshold:
                    group.append(q2)
                    used.add(j)
            groups.append(group)
        if any(len(g) >= min_group_size for g in groups):
            if verbose:
                print(f"   ✅ Fuzzy threshold {threshold} produced valid groups.")
            return groups
    if verbose:
        print("   ⚠️ No fuzzy grouping met minimum group size.")
    return [[q] for q in queries]

def cosine_grouping_adaptive(
    queries: List[str],
    min_group_size: int = 2,
    thresholds: List[float] = [0.4, 0.3, 0.2, 0.1],
    verbose: bool = False
) -> List[List[str]]:
    if verbose:
        print("[DEBUG] 📐 Starting cosine clustering with adaptive threshold...")

    for threshold in thresholds:
        if verbose:
            print(f"   → Trying cosine threshold: {threshold}")
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(queries)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        distance_matrix = 1 - similarity_matrix

        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric='precomputed',
            linkage='average',
            distance_threshold=1 - threshold
        )
        labels = clustering.fit_predict(distance_matrix)

        grouped = {}
        for query, label in zip(queries, labels):
            grouped.setdefault(label, []).append(query)

        group_list = list(grouped.values())
        if any(len(g) >= min_group_size for g in group_list):
            if verbose:
                print(f"   ✅ Cosine threshold {threshold} produced valid groups.")
            return group_list
    if verbose:
        print("   ⚠️ No cosine grouping met minimum group size.")
    return [[q] for q in queries]

def merge_groups(
    fuzzy_groups: List[List[str]],
    cosine_groups: List[List[str]],
    verbose: bool = False
) -> List[List[str]]:
    if verbose:
        print("[DEBUG] 🔁 Merging fuzzy and cosine groups...")

    merged = [set(group) for group in fuzzy_groups]

    for cos_group in cosine_groups:
        cos_set = set(cos_group)
        overlapping = []

        for idx, fuzzy_set in enumerate(merged):
            if fuzzy_set & cos_set:
                overlapping.append(idx)

        if overlapping:
            if verbose:
                print(f"   🔗 Merging: {cos_group} into fuzzy groups {overlapping}")
            new_group = set()
            for idx in sorted(overlapping, reverse=True):
                new_group |= merged.pop(idx)
            new_group |= cos_set
            merged.append(new_group)
        else:
            if verbose:
                print(f"   ➕ Adding new cosine-only group: {cos_group}")
            merged.append(cos_set)

    return [list(group) for group in merged]

def group_queries(
    queries: List[str],
    config: Dict[str, Any] = {}
) -> List[List[str]]:
    """
    Group similar queries using adaptive fuzzy and cosine similarity.

    Parameters:
        queries (List[str]): List of queries.
        config (Dict[str, Any]): Configurable parameters:
            - min_group_size
            - max_fuzzy_threshold
            - min_fuzzy_threshold
            - fuzzy_step
            - cosine_thresholds
            - verbose

    Returns:
        List[List[str]]: Grouped queries.
    """
    verbose = config.get("verbose", False)
    min_group_size = config.get("min_group_size", 2)
    max_fuzzy = config.get("max_fuzzy_threshold", 90)
    min_fuzzy = config.get("min_fuzzy_threshold", 60)
    fuzzy_step = config.get("fuzzy_step", 5)
    cosine_thresholds = config.get("cosine_thresholds", [0.4, 0.3, 0.2, 0.1])

    if verbose:
        print("[INFO] Step 1: Fuzzy grouping...")
    fuzzy_groups = fuzzy_grouping_adaptive(
        queries, min_group_size, max_fuzzy, min_fuzzy, fuzzy_step, verbose
    )

    flat_queries = [q for group in fuzzy_groups if len(group) >= min_group_size for q in group]
    remaining_queries = [q for q in queries if q not in flat_queries]

    cosine_groups = []
    if remaining_queries:
        if verbose:
            print("[INFO] Step 2: Cosine clustering...")
        cosine_groups = cosine_grouping_adaptive(
            remaining_queries, min_group_size, cosine_thresholds, verbose
        )

    if verbose:
        print("[INFO] Step 3: Merging groups...")
    final_groups = merge_groups(fuzzy_groups, cosine_groups, verbose)

    return final_groups

def print_grouped_queries(groups: List[List[str]]):
    print("\n📦 Final Grouped Queries:")
    for i, group in enumerate(groups, 1):
        print(f"\n📌 Group {i}:")
        for q in group:
            print("   -", q)

if __name__ == "__main__":
    queries = [
        # Ambient Lighting / Interior
        "How do I reset the car's ambient lights?",
        "What's the process for resetting ambient lighting?",
        "I want to change the color of interior lights.",
        "Can I adjust ambient lighting with my voice?",
        "Is there a way to dim the cabin lights automatically?",
        "How do I switch off the ambient lights at night?",

        # Music / Audio / Bluetooth
        "How can I play music using voice commands?",
        "Start playing a song using the assistant.",
        "Turn on Bluetooth audio",
        "Play my favorite playlist through voice control.",
        "Can I start music without touching the screen?",
        "How to switch music tracks hands-free?",
        "Is there a way to adjust the volume using voice?",
        "How do I pair a new Bluetooth device?",
        "The music stopped, how do I resume playback?",

        # Cruise Control / Driving Assist
        "How do I activate cruise control?",
        "Enable adaptive cruise feature in the car",
        "What is cruise assist in my vehicle?",
        "How can I disable adaptive cruise temporarily?",
        "Increase cruise control speed using voice",
        "What’s the button for cruise mode?",
        "How do I know if cruise control is on?",

        # Navigation / Maps
        "Open Google Maps on the dashboard.",
        "How do I set a destination by speaking?",
        "Find the nearest charging station.",
        "Navigate home using voice assistant.",
        "Change current route to avoid traffic.",

        # Climate / AC
        "How do I turn off the AC?",
        "Set temperature to 22 degrees using voice.",
        "Can I change fan speed hands-free?",
        "What’s the command to defog the windshield?"
    ]

    config = {
        "verbose": False,
        "min_group_size": 4,
        "max_fuzzy_threshold": 90,
        "min_fuzzy_threshold": 20,
        "fuzzy_step": 1,
        "cosine_thresholds": [0.4, 0.3, 0.2, 0.1]
    }

    groups = group_queries(queries, config=config)
    print_grouped_queries(groups)
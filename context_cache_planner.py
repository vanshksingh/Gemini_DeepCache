from collections import defaultdict
from itertools import combinations

# === STEP 1: INPUTS ===

# Expanded example with more diversity
query_to_chunks = {
    "How do I play music?": ["chunk1", "chunk2"],
    "Connect Bluetooth to play songs": ["chunk1", "chunk2"],
    "Use voice to change music": ["chunk1", "chunk3"],
    "Where is the map icon?": ["chunk4", "chunk5"],
    "What does the navigation button do?": ["chunk4", "chunk5"],
    "Set destination using touchscreen": ["chunk5", "chunk6"],
    "Adjust the fan speed": ["chunk7", "chunk8"],
    "Change the temperature in the car": ["chunk7", "chunk9"],
    "Access climate control from the screen": ["chunk8", "chunk9"],
    "Activate defrost mode": ["chunk8", "chunk10"],
    "Open sunroof": ["chunk11", "chunk12"],
    "Check tire pressure": ["chunk13", "chunk14"]
}

# Semantic similarity pairs (grouped by meaning even if chunk overlap is low)
semantic_pairs = [
    {"How do I play music?", "Connect Bluetooth to play songs"},
    {"Where is the map icon?", "What does the navigation button do?"},
    {"Change the temperature in the car", "Access climate control from the screen"},
    {"Adjust the fan speed", "Activate defrost mode"},
    {"Use voice to change music", "Set destination using touchscreen"}
]

# Thresholds
JACCARD_SIM_THRESHOLD = 0.5
MIN_GROUP_SIZE = 2

# === STEP 2: Build Graph from Chunk Overlap ===

def jaccard_similarity(set1, set2):
    return len(set1 & set2) / len(set1 | set2)

query_graph = defaultdict(set)
query_list = list(query_to_chunks.keys())

# Add edges based on chunk overlap
for q1, q2 in combinations(query_list, 2):
    chunks1 = set(query_to_chunks[q1])
    chunks2 = set(query_to_chunks[q2])
    if jaccard_similarity(chunks1, chunks2) >= JACCARD_SIM_THRESHOLD:
        query_graph[q1].add(q2)
        query_graph[q2].add(q1)

# === STEP 3: Add Semantic Similarity Links ===

for pair in semantic_pairs:
    q1, q2 = tuple(pair)
    query_graph[q1].add(q2)
    query_graph[q2].add(q1)

# === STEP 4: Group Queries into Clusters ===

def build_clusters(graph):
    visited = set()
    clusters = []
    for node in graph:
        if node not in visited:
            cluster = set()
            stack = [node]
            while stack:
                current = stack.pop()
                if current not in visited:
                    visited.add(current)
                    cluster.add(current)
                    stack.extend(graph[current] - visited)
            if len(cluster) >= MIN_GROUP_SIZE:
                clusters.append(cluster)
    return clusters

explicit_cache_clusters = build_clusters(query_graph)

# === STEP 5: Collect Shared Chunks per Cluster ===

explicit_cache_plan = []
explicit_cache_chunks = set()

for cluster in explicit_cache_clusters:
    cluster_chunks = set()
    for q in cluster:
        cluster_chunks.update(query_to_chunks[q])
    explicit_cache_plan.append({
        "queries": list(cluster),
        "shared_chunks": list(cluster_chunks)
    })
    explicit_cache_chunks.update(cluster_chunks)

# === STEP 6: Identify Leftover Queries ===

all_queries = set(query_to_chunks.keys())
used_queries = set(q for group in explicit_cache_plan for q in group["queries"])
implicit_cache_queries = list(all_queries - used_queries)

implicit_cache_chunks = set()
for q in implicit_cache_queries:
    implicit_cache_chunks.update(query_to_chunks[q])

# === STEP 7: Verbose Output ===

print("\n📦 Explicit Cache Plan (Shared Cache Groups):")
for idx, plan in enumerate(explicit_cache_plan, 1):
    print(f"\n🔹 Group {idx} (Total Queries: {len(plan['queries'])}, Chunks: {len(plan['shared_chunks'])})")
    print("  Queries:")
    for q in plan['queries']:
        print(f"    - {q}")
    print("  Shared Chunks:")
    for ch in plan['shared_chunks']:
        print(f"    - {ch}")
    chunk_counts = {ch: sum(ch in query_to_chunks[q] for q in plan['queries']) for ch in plan['shared_chunks']}
    print("  Chunk Usage Frequency:")
    for ch, count in chunk_counts.items():
        print(f"    - {ch}: used in {count} queries")

print("\n💨 Implicit Cache (One-off or Diverse Queries):")
if not implicit_cache_queries:
    print("  ✅ No queries need implicit caching — all queries are covered by shared chunk groups.")
else:
    print(f"  Total: {len(implicit_cache_queries)}")
    print("  Queries:")
    for q in implicit_cache_queries:
        print(f"    - {q}")
    print("  Chunks:")
    for ch in implicit_cache_chunks:
        print(f"    - {ch}")

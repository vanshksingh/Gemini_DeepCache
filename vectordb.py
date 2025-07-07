import os
import textwrap
import hashlib
from typing import List, Dict, Tuple

import numpy as np
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from google import genai
from google.genai import types
from itertools import combinations
from sklearn.cluster import KMeans
from dotenv import load_dotenv

# Load environment and initialize Gemini client
def load_gemini_client() -> genai.Client:
    load_dotenv()
    return genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

client = load_gemini_client()


class GeminiEmbeddingFunction(EmbeddingFunction[Documents]):
    """
    EmbeddingFunction wrapper for Gemini embed_content API.
    """
    def __init__(
        self,
        model_id: str = "models/embedding-001",
        task_type: str = "RETRIEVAL_DOCUMENT",
        title: str = None,
        verbose: bool = False
    ):
        self.model_id = model_id
        self.task_type = task_type.upper()
        self.title = title
        self.verbose = verbose

    def __call__(self, documents: Documents) -> Embeddings:
        embeddings: List[List[float]] = []
        for doc in documents:
            if self.verbose:
                print(f"[Embed] Calling Gemini embed API for: {doc[:60]}...")
            config = types.EmbedContentConfig(task_type=self.task_type)
            if self.title and self.task_type != "SEMANTIC_SIMILARITY":
                config.title = self.title
            response = client.models.embed_content(
                model=self.model_id,
                contents=doc,
                config=config
            )
            embeddings.append(response.embeddings[0].values)
        return embeddings


# -------- Text Chunking -------- #
def chunk_text(text: str, chunk_size: int = 200, overlap: int = 40) -> List[str]:
    """
    Split text into overlapping chunks of words.
    """
    words = text.split()
    chunks: List[str] = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunks.append(" ".join(words[i : i + chunk_size]))
    return chunks


# -------- ChromaDB Helpers -------- #
def clear_chroma_collection(db_name: str) -> None:
    """Delete an existing ChromaDB collection by name."""
    store = chromadb.PersistentClient(path="./chroma_store")
    try:
        store.delete_collection(name=db_name)
        print(f"[INFO] Cleared ChromaDB collection: {db_name}")
    except ValueError:
        print(f"[INFO] Collection {db_name} not found; nothing to delete.")


def get_or_create_chroma_db(
    db_name: str,
    embedding_function: EmbeddingFunction,
    clear: bool = False
):
    """Get or create a persistent ChromaDB collection."""
    store = chromadb.PersistentClient(path="./chroma_store")
    if clear:
        clear_chroma_collection(db_name)
    return store.get_or_create_collection(
        name=db_name,
        embedding_function=embedding_function
    )


def add_documents_to_db(
    db,
    raw_documents: List[str],
    chunk_size: int = 200,
    overlap: int = 40
) -> None:
    """Chunk and add new documents to the ChromaDB collection."""
    all_chunks, all_ids = [], []
    for doc_idx, doc in enumerate(raw_documents):
        for chunk_idx, chunk in enumerate(chunk_text(doc, chunk_size, overlap)):
            all_chunks.append(chunk)
            all_ids.append(f"{doc_idx}-{chunk_idx}")

    existing = db.get(ids=all_ids) or {"ids": []}
    existing_ids = set(existing.get("ids") or [])

    new_chunks, new_ids = [], []
    for cid, chunk in zip(all_ids, all_chunks):
        if cid not in existing_ids:
            new_ids.append(cid)
            new_chunks.append(chunk)

    if new_chunks:
        db.add(ids=new_ids, documents=new_chunks)


def get_top_k_chunks_for_queries(
    queries: List[str],
    db,
    k: int = 2
) -> Dict[str, List[Tuple[str, float]]]:
    """Return top-k retrieved chunks and distances for each query."""
    results: Dict[str, List[Tuple[str, float]]] = {}
    for q in queries:
        out = db.query(query_texts=[q], n_results=k)
        docs   = out.get("documents")[0]
        scores = out.get("distances")[0]
        results[q] = list(zip(docs, scores))
    return results


# -------- Query Embedding Cache -------- #
def get_or_create_query_cache(
    model_id: str,
    task_type: str
):
    """Get or create a persistent cache collection for query embeddings."""
    key    = f"{model_id}_{task_type}".encode("utf-8")
    suffix = hashlib.md5(key).hexdigest()[:10]
    name   = f"query_cache_{suffix}"
    store  = chromadb.PersistentClient(path="./chroma_store")
    ef     = GeminiEmbeddingFunction(model_id=model_id, task_type=task_type)
    return store.get_or_create_collection(name=name, embedding_function=ef)


def embed_queries_cached(
    queries: List[str],
    cache_db,
    model_id: str = "models/embedding-001",
    task_type: str = "SEMANTIC_SIMILARITY"
) -> Dict[str, np.ndarray]:
    """
    Persistently embed queries. Only calls Gemini API for uncached queries.
    """
    task_type = task_type.upper()
    # Stable IDs
    qid_map = {
        q: hashlib.md5(f"{task_type}::{q}".encode("utf-8")).hexdigest()
        for q in queries
    }

    # Retrieve existing cache entries
    retrieved = cache_db.get(ids=list(qid_map.values()))
    safe      = retrieved or {}
    ids_list  = safe.get("ids") or []
    embs_list = safe.get("embeddings") or []
    cache_map = dict(zip(ids_list, embs_list))

    results, to_embed = {}, []
    for q, qid in qid_map.items():
        if qid in cache_map:
            results[q] = np.array(cache_map[qid])
        else:
            to_embed.append(q)

    if to_embed:
        new_embs = []
        for q in to_embed:
            print(f"[Embed] Calling Gemini API for query: {q}")
            resp = client.models.embed_content(
                model=model_id,
                contents=q,
                config=types.EmbedContentConfig(task_type=task_type)
            )
            new_embs.append(resp.embeddings[0].values)

        new_ids = [qid_map[q] for q in to_embed]
        cache_db.add(ids=new_ids, embeddings=new_embs, documents=to_embed)
        for q, emb in zip(to_embed, new_embs):
            results[q] = np.array(emb)

    return results


# -------- Embedding Analysis -------- #
def get_disjoint_pairs(
    embeddings: Dict[str, np.ndarray]
) -> List[Tuple[Tuple[str, str], float]]:
    """Return highest-similarity disjoint pairs of queries."""
    sim = lambda a, b: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    qs, vecs = list(embeddings.keys()), [embeddings[q] for q in embeddings]
    used, pairs = set(), []
    for i, j in sorted(
        combinations(range(len(qs)), 2),
        key=lambda ij: -sim(vecs[ij[0]], vecs[ij[1]])
    ):
        if i in used or j in used:
            continue
        pairs.append(((qs[i], qs[j]), sim(vecs[i], vecs[j])))
        used.update([i, j])
        if len(used) >= len(qs):
            break
    return pairs


def group_queries(
    embeddings: Dict[str, np.ndarray],
    num_groups: int = 2
) -> Dict[int, List[str]]:
    """Cluster queries into groups via k-means."""
    qs   = list(embeddings.keys())
    vecs = [embeddings[q] for q in qs]
    kmeans = KMeans(n_clusters=num_groups, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(vecs)
    clusters = {i: [] for i in range(num_groups)}
    for idx, lbl in enumerate(labels):
        clusters[lbl].append(qs[idx])
    return clusters


# -------- Main Script -------- #
if __name__ == "__main__":
    # Sample document
    DOCUMENT = """
    Operating the Climate Control System  Your Googlecar has a climate control
    system that allows you to adjust the temperature and airflow in the car.
    ... (rest of your text)
    """

    # Build document DB
    doc_ef = GeminiEmbeddingFunction(
        model_id="models/embedding-001",
        task_type="RETRIEVAL_DOCUMENT",
        verbose=True
    )
    doc_db = get_or_create_chroma_db(
        db_name="googlecar_testdb",
        embedding_function=doc_ef,
        clear=False
    )
    add_documents_to_db(doc_db, [DOCUMENT], chunk_size=150, overlap=30)

    # Define queries
    test_queries = [
        "How do I play music?",
        "Where is the map icon?",
        "Change the temperature in the car",
        "Adjust the fan speed",
        "Activate defrost mode",
        "Connect Bluetooth to play songs",
        "Use voice to change music",
        "Access climate control from the screen",
        "Set destination using touchscreen",
        "What does the navigation button do?"
    ]

    # Retrieve and print top-k chunks
    chunk_results = get_top_k_chunks_for_queries(test_queries, doc_db, k=2)
    for q, hits in chunk_results.items():
        print(f"\nQuery: {q}")
        for doc, score in hits:
            print(f"  - {textwrap.shorten(doc, width=80)} (score: {score:.4f})")

    # Embed queries with cache
    cache_db = get_or_create_query_cache(
        model_id="models/embedding-001",
        task_type="SEMANTIC_SIMILARITY"
    )
    q_embs = embed_queries_cached(test_queries, cache_db)

    # Similarity and clustering
    print("\nTop non-overlapping pairs:")
    for (q1, q2), score in get_disjoint_pairs(q_embs):
        print(f"  {q1} <> {q2}: {score:.4f}")

    print("\nGrouped queries:")
    clusters = group_queries(q_embs, num_groups=2)
    for grp, members in clusters.items():
        print(f" Group {grp+1}:")
        for m in members:
            print(f"  - {m}")

import os
import textwrap
import chromadb
import numpy as np
from dotenv import load_dotenv
from chromadb import Documents, EmbeddingFunction, Embeddings
from google import genai
from google.genai import types
from itertools import combinations
from sklearn.cluster import KMeans
import hashlib

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_id="models/embedding-001", task_type="retrieval_document", title=None, verbose=False):
        self.model_id = model_id
        self.task_type = task_type
        self.title = title
        self.verbose = verbose

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for doc in input:
            if self.verbose:
                print(f"[Embed] Calling Gemini embed API for: {doc[:60]}...")
            config = types.EmbedContentConfig(task_type=self.task_type)
            if self.title and self.task_type.lower() != "semantic_similarity":
                config.title = self.title
            response = client.models.embed_content(
                model=self.model_id,
                contents=doc,
                config=config
            )
            embeddings.append(response.embeddings[0].values)
        return embeddings

def chunk_text(text, chunk_size=200, overlap=40):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def clear_chroma_collection(db_name):
    chroma_client = chromadb.PersistentClient(path="./chroma_store")
    try:
        chroma_client.delete_collection(name=db_name)
        print(f"[INFO] Cleared ChromaDB collection: {db_name}")
    except ValueError:
        print(f"[INFO] Collection {db_name} not found. Nothing to delete.")

def get_or_create_chroma_db(db_name, embedding_function, clear=False):
    chroma_client = chromadb.PersistentClient(path="./chroma_store")
    if clear:
        clear_chroma_collection(db_name)
    return chroma_client.get_or_create_collection(name=db_name, embedding_function=embedding_function)

def add_documents_to_db(db, raw_documents, chunk_size=200, overlap=40):
    all_chunks = []
    all_ids = []

    for doc_index, doc in enumerate(raw_documents):
        chunks = chunk_text(doc, chunk_size=chunk_size, overlap=overlap)
        for chunk_index, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{doc_index}-{chunk_index}")

    existing = db.get(ids=all_ids)
    existing_ids = set(existing["ids"]) if existing and "ids" in existing else set()

    new_chunks = []
    new_ids = []

    for i, chunk_id in enumerate(all_ids):
        if chunk_id not in existing_ids:
            new_chunks.append(all_chunks[i])
            new_ids.append(chunk_id)

    if new_chunks:
        db.add(documents=new_chunks, ids=new_ids)

def get_top_k_chunks_for_queries(queries, db, k=2):
    results = {}
    for query in queries:
        out = db.query(query_texts=[query], n_results=k)
        passages = out['documents'][0]
        scores = out['distances'][0]
        results[query] = list(zip(passages, scores))
    return results

def get_or_create_query_cache(model_id: str, task_type: str):
    key = f"{model_id}_{task_type}".lower().encode("utf-8")
    collection_hash = hashlib.md5(key).hexdigest()[:10]
    collection_name = f"query_cache_{collection_hash}"
    chroma_client = chromadb.PersistentClient(path="./chroma_store")
    embedding_function = GeminiEmbeddingFunction(model_id=model_id, task_type=task_type)
    return chroma_client.get_or_create_collection(name=collection_name, embedding_function=embedding_function)

def embed_queries_cached(queries, cache_db, model_id="models/embedding-001", task_type="SEMANTIC_SIMILARITY"):
    embeddings_dict = {}
    to_embed = []

    def make_query_id(query: str, task_type: str) -> str:
        key = f"{task_type.lower()}::{query}".encode("utf-8")
        return hashlib.md5(key).hexdigest()

    query_to_id = {q: make_query_id(q, task_type) for q in queries}

    existing = cache_db.get(ids=list(query_to_id.values()))
    existing_map = {}

    if existing and existing.get("ids") and existing.get("embeddings"):
        existing_map = {id_: vec for id_, vec in zip(existing["ids"], existing["embeddings"])}

    for query, qid in query_to_id.items():
        if qid in existing_map:
            embeddings_dict[query] = np.array(existing_map[qid])
        else:
            to_embed.append(query)

    if to_embed:
        new_embeddings = []
        for query in to_embed:
            print(f"[Embed] Gemini API called for query: {query}")
            response = client.models.embed_content(
                model=model_id,
                contents=query,
                config=types.EmbedContentConfig(task_type=task_type)
            )
            new_embeddings.append(response.embeddings[0].values)

        new_ids = [query_to_id[q] for q in to_embed]
        existing = cache_db.get(ids=new_ids)
        existing_ids = set(existing["ids"]) if existing and "ids" in existing else set()

        final_docs, final_embs, final_ids = [], [], []
        for q, emb, qid in zip(to_embed, new_embeddings, new_ids):
            if qid not in existing_ids:
                final_docs.append(q)
                final_embs.append(emb)
                final_ids.append(qid)

        if final_docs:
            cache_db.add(documents=final_docs, embeddings=final_embs, ids=final_ids)

        for q, emb in zip(to_embed, new_embeddings):
            embeddings_dict[q] = np.array(emb)

    return embeddings_dict

def get_disjoint_pairs(embeddings_dict):
    used = set()
    pairs = []
    queries = list(embeddings_dict.keys())
    vectors = [embeddings_dict[q] for q in queries]
    sim = lambda a, b: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    for (i, j) in sorted(combinations(range(len(queries)), 2),
                         key=lambda pair: -sim(vectors[pair[0]], vectors[pair[1]])):
        if i in used or j in used:
            continue
        pairs.append(((queries[i], queries[j]), sim(vectors[i], vectors[j])))
        used.add(i)
        used.add(j)
        if len(used) >= len(queries):
            break
    return pairs

def group_queries(embeddings_dict, num_groups=2):
    queries = list(embeddings_dict.keys())
    vectors = [embeddings_dict[q] for q in queries]
    kmeans = KMeans(n_clusters=num_groups, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(vectors)
    clusters = {i: [] for i in range(num_groups)}
    for i, label in enumerate(labels):
        clusters[label].append(queries[i])
    return clusters

if __name__ == "__main__":
    DOCUMENT = """     Operating the Climate Control System  Your Googlecar has a climate control     system that allows you to adjust the temperature and airflow in the car.     To operate the climate control system, use the buttons and knobs located on     the center console.  Temperature: The temperature knob controls the     temperature inside the car. Turn the knob clockwise to increase the     temperature or counterclockwise to decrease the temperature.     Airflow: The airflow knob controls the amount of airflow inside the car.     Turn the knob clockwise to increase the airflow or counterclockwise to     decrease the airflow. Fan speed: The fan speed knob controls the speed     of the fan. Turn the knob clockwise to increase the fan speed or     counterclockwise to decrease the fan speed.     Mode: The mode button allows you to select the desired mode. The available     modes are: Auto: The car will automatically adjust the temperature and     airflow to maintain a comfortable level.     Cool: The car will blow cool air into the car.     Heat: The car will blow warm air into the car.     Defrost: The car will blow warm air onto the windshield to defrost it.     Your Googlecar has a large touchscreen display that provides access to a     variety of features, including navigation, entertainment, and climate     control. To use the touchscreen display, simply touch the desired icon.     For example, you can touch the "Navigation" icon to get directions to     your destination or touch the "Music" icon to play your favorite songs.     Shifting Gears Your Googlecar has an automatic transmission. To     shift gears, simply move the shift lever to the desired position.     Park: This position is used when you are parked. The wheels are locked     and the car cannot move.     Reverse: This position is used to back up.     Neutral: This position is used when you are stopped at a light or in traffic.     The car is not in gear and will not move until and unless you press the gas pedal.     Drive: This position is used to drive forward.     Low: This position is used for driving in snow or other slippery conditions.     """


    documents = [DOCUMENT]
    chunk_size = 150
    overlap = 30
    db_name = "googlecar_testdb"
    model_id = "models/embedding-001"
    task_type = "SEMANTIC_SIMILARITY"

    embedding_func = GeminiEmbeddingFunction(model_id=model_id, task_type="RETRIEVAL_DOCUMENT", verbose=True)

    # Optional: clear document DB if needed
    # clear_chroma_collection(db_name)

    db = get_or_create_chroma_db(db_name=db_name, embedding_function=embedding_func, clear=False)
    add_documents_to_db(db, documents, chunk_size=chunk_size, overlap=overlap)

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

    print("\n🔍 Top-k chunks for each query:")
    chunk_results = get_top_k_chunks_for_queries(test_queries, db, k=2)
    for query, chunks in chunk_results.items():
        print(f"\nQuery: {query}")
        for i, (chunk, score) in enumerate(chunks, start=1):
            print(f"  Result {i}: {textwrap.shorten(chunk, width=80)}  (score: {score:.4f})")

    print("\n🧠 Top Non-Overlapping Pairs:")
    query_cache = get_or_create_query_cache(model_id=model_id, task_type=task_type)
    query_embeddings_dict = embed_queries_cached(
        test_queries, cache_db=query_cache, model_id=model_id, task_type=task_type
    )

    closest_pairs = get_disjoint_pairs(query_embeddings_dict)
    for i, ((q1, q2), score) in enumerate(closest_pairs, start=1):
        print(f"\nPair {i}:\n🔹 {q1}\n🔹 {q2}\n→ Similarity: {score:.4f}")

    print("\n🔗 Grouped Queries (2 sets of 5):")
    clustered = group_queries(query_embeddings_dict, num_groups=2)
    for i, group in clustered.items():
        print(f"\nGroup {i + 1}:")
        for q in group:
            print(f" - {q}")
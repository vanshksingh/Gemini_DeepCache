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

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_id="models/embedding-001", task_type="retrieval_document", title=None):
        self.model_id = model_id
        self.task_type = task_type
        self.title = title

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for doc in input:
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

def get_or_create_chroma_db(db_name, embedding_function, clear=False):
    chroma_client = chromadb.PersistentClient(path="./chroma_store")
    if clear:
        try:
            chroma_client.delete_collection(name=db_name)
        except ValueError:
            pass
    return chroma_client.get_or_create_collection(name=db_name, embedding_function=embedding_function)

def add_documents_to_db(db, raw_documents, chunk_size=200, overlap=40):
    all_chunks = []
    all_ids = []
    for doc_index, doc in enumerate(raw_documents):
        chunks = chunk_text(doc, chunk_size=chunk_size, overlap=overlap)
        for chunk_index, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{doc_index}-{chunk_index}")
    db.add(documents=all_chunks, ids=all_ids)

def get_top_k_chunks_for_queries(queries, db, k=2):
    results = {}
    for query in queries:
        out = db.query(query_texts=[query], n_results=k)
        passages = out['documents'][0]
        scores = out['distances'][0]
        results[query] = list(zip(passages, scores))
    return results

def get_embeddings(queries, model_id="models/embedding-001", task_type="SEMANTIC_SIMILARITY"):
    embeddings = []
    for query in queries:
        response = client.models.embed_content(
            model=model_id,
            contents=query,
            config=types.EmbedContentConfig(task_type=task_type)
        )
        embeddings.append(np.array(response.embeddings[0].values))
    return embeddings

def get_disjoint_pairs(queries, embeddings):
    used = set()
    pairs = []
    sim = lambda a, b: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    for (i, j) in sorted(combinations(range(len(queries)), 2),
                         key=lambda pair: -sim(embeddings[pair[0]], embeddings[pair[1]])):
        if i in used or j in used:
            continue
        pairs.append(((queries[i], queries[j]), sim(embeddings[i], embeddings[j])))
        used.add(i)
        used.add(j)
        if len(used) >= len(queries):
            break
    return pairs

def group_queries(queries, embeddings, num_groups=2):
    kmeans = KMeans(n_clusters=num_groups, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(embeddings)
    clusters = {i: [] for i in range(num_groups)}
    for i, label in enumerate(labels):
        clusters[label].append(queries[i])
    return clusters

if __name__ == "__main__":
    DOCUMENT = """
  Operating the Climate Control System  Your Googlecar has a climate control
  system that allows you to adjust the temperature and airflow in the car.
  To operate the climate control system, use the buttons and knobs located on
  the center console.  Temperature: The temperature knob controls the
  temperature inside the car. Turn the knob clockwise to increase the
  temperature or counterclockwise to decrease the temperature.
  Airflow: The airflow knob controls the amount of airflow inside the car.
  Turn the knob clockwise to increase the airflow or counterclockwise to
  decrease the airflow. Fan speed: The fan speed knob controls the speed
  of the fan. Turn the knob clockwise to increase the fan speed or
  counterclockwise to decrease the fan speed.
  Mode: The mode button allows you to select the desired mode. The available
  modes are: Auto: The car will automatically adjust the temperature and
  airflow to maintain a comfortable level.
  Cool: The car will blow cool air into the car.
  Heat: The car will blow warm air into the car.
  Defrost: The car will blow warm air onto the windshield to defrost it.
  Your Googlecar has a large touchscreen display that provides access to a
  variety of features, including navigation, entertainment, and climate
  control. To use the touchscreen display, simply touch the desired icon.
  For example, you can touch the "Navigation" icon to get directions to
  your destination or touch the "Music" icon to play your favorite songs.
  Shifting Gears Your Googlecar has an automatic transmission. To
  shift gears, simply move the shift lever to the desired position.
  Park: This position is used when you are parked. The wheels are locked
  and the car cannot move.
  Reverse: This position is used to back up.
  Neutral: This position is used when you are stopped at a light or in traffic.
  The car is not in gear and will not move unless you press the gas pedal.
  Drive: This position is used to drive forward.
  Low: This position is used for driving in snow or other slippery conditions.
"""

    documents = [DOCUMENT]

    chunk_size = 150
    overlap = 30
    db_name = "googlecar_testdb"
    model_id = "models/embedding-001"
    task_type = "SEMANTIC_SIMILARITY"

    embedding_func = GeminiEmbeddingFunction(model_id=model_id, task_type="RETRIEVAL_DOCUMENT")
    db = get_or_create_chroma_db(db_name=db_name, embedding_function=embedding_func, clear=True)
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
    query_embeddings = get_embeddings(test_queries, model_id=model_id, task_type=task_type)
    closest_pairs = get_disjoint_pairs(test_queries, query_embeddings)

    for i, ((q1, q2), score) in enumerate(closest_pairs, start=1):
        print(f"\nPair {i}:\n🔹 {q1}\n🔹 {q2}\n→ Similarity: {score:.4f}")

    print("\n🔗 Grouped Queries (2 sets of 5):")
    clustered = group_queries(test_queries, query_embeddings, num_groups=2)
    for i, group in clustered.items():
        print(f"\nGroup {i + 1}:")
        for q in group:
            print(f" - {q}")

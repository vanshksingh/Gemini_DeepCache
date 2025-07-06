import textwrap
import chromadb
import numpy as np
import pandas as pd
from IPython.display import Markdown
from chromadb import Documents, EmbeddingFunction, Embeddings
from google import genai
from dotenv import load_dotenv
import os
from google.genai import types

# Load environment variables
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ----------------------------
# 📌 CHUNKING FUNCTION
# ----------------------------
def chunk_text(text, chunk_size=200, overlap=40):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# ----------------------------
# 📌 Custom Gemini Embedding Function
# ----------------------------
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        EMBEDDING_MODEL_ID = "models/embedding-001"

        # Loop over all input documents and embed each one individually
        all_embeddings = []
        for doc in input:
            response = client.models.embed_content(
                model=EMBEDDING_MODEL_ID,
                contents=doc,
                config=types.EmbedContentConfig(
                    task_type="retrieval_document",
                    title="Document Chunk"
                )
            )
            all_embeddings.append(response.embeddings[0].values)

        return all_embeddings  # List of embeddings (shape: [n_chunks][dim])


# ----------------------------
# 📌 Create Vector DB from Raw Documents
# ----------------------------
def create_chroma_db_from_raw_documents(raw_documents, db_name):
    chroma_client = chromadb.Client()
    db = chroma_client.create_collection(
        name=db_name,
        embedding_function=GeminiEmbeddingFunction()
    )

    all_chunks = []
    all_ids = []

    for doc_index, doc in enumerate(raw_documents):
        chunks = chunk_text(doc)
        for chunk_index, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{doc_index}-{chunk_index}")  # ID format: doc-chunk

    db.add(documents=all_chunks, ids=all_ids)
    return db

# ----------------------------
# 📌 Search Functions
# ----------------------------
def get_top_k_passages(query, db, k=3):
    results = db.query(query_texts=[query], n_results=k)
    return results['documents'][0]

def get_top_k_passages_with_scores(query, db, k=3):
    results = db.query(query_texts=[query], n_results=k)
    passages = results['documents'][0]
    scores = results['distances'][0]
    return list(zip(passages, scores))

# ----------------------------
# 📌 Example Input Documents (Any Length)
# ----------------------------

# Sample documents
DOCUMENT1 = """
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
"""

DOCUMENT2 = """
  Your Googlecar has a large touchscreen display that provides access to a
  variety of features, including navigation, entertainment, and climate
  control. To use the touchscreen display, simply touch the desired icon.
  For example, you can touch the "Navigation" icon to get directions to
  your destination or touch the "Music" icon to play your favorite songs.
"""

DOCUMENT3 = """
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

documents = [DOCUMENT1, DOCUMENT2, DOCUMENT3]

# ----------------------------
# ✅ MAIN
# ----------------------------
db = create_chroma_db_from_raw_documents(documents, "googlecars_db")

query = "touch screen features"
top_k = 4
top_passages = get_top_k_passages_with_scores(query, db, k=top_k)

for i, (passage, score) in enumerate(top_passages, start=1):
    print(f"\n🔹 Top {i} (Score: {score:.4f}):\n{textwrap.fill(passage, width=100)}")

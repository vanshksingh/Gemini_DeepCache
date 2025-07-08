import os
import json
from pathlib import Path
from dotenv import load_dotenv

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction
from google import genai
from google.genai import types

# === CONFIGURATION ===
CHUNK_SIZE_DEFAULT = 200
OVERLAP_DEFAULT = 40
TOP_K_DEFAULT = 10
DB_PATH = "./chroma_store"


def load_gemini_client() -> genai.Client:
    """Load environment and initialize Gemini client."""
    load_dotenv()
    return genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


class GeminiEmbeddingFunction(EmbeddingFunction[Documents]):
    """
    EmbeddingFunction wrapper for Gemini embed_content API.
    """
    def __init__(
        self,
        client: genai.Client,
        model_id: str = "models/embedding-001",
        task_type: str = "RETRIEVAL_DOCUMENT",
    ):
        self.client = client
        self.model_id = model_id
        self.task_type = task_type.upper()

    def __call__(self, documents: Documents):
        embeddings = []
        for doc in documents:
            resp = self.client.models.embed_content(
                model=self.model_id,
                contents=doc,
                config=types.EmbedContentConfig(task_type=self.task_type)
            )
            embeddings.append(resp.embeddings[0].values)
        return embeddings


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split text into overlapping chunks of words.
    """
    words = text.split()
    step = chunk_size - overlap
    return [
        " ".join(words[i : i + chunk_size])
        for i in range(0, len(words), step)
    ]


def get_or_create_chroma_db(
    client: genai.Client,
    db_name: str,
    embedding_fn: EmbeddingFunction,
    clear: bool = False
):
    """Get or create a persistent ChromaDB collection."""
    store = chromadb.PersistentClient(path=DB_PATH)
    if clear:
        try:
            store.delete_collection(name=db_name)
        except ValueError:
            pass
    return store.get_or_create_collection(
        name=db_name,
        embedding_function=embedding_fn
    )


def generate_chunk_and_query_mappings(
    *,
    text_file: str,
    queries_file: str,
    output_chunk_json: str = "chunks.json",
    output_query_json: str = "queries.json",
    chunk_size: int = CHUNK_SIZE_DEFAULT,
    overlap: int = OVERLAP_DEFAULT,
    top_k: int = TOP_K_DEFAULT,
    db_name: str = "document_db"
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """
    Read a text file and a JSON file with queries, then:
    1) Chunk the text and write a JSON mapping chunk IDs to chunk text.
    2) Retrieve top-k chunk IDs for each query and write a JSON mapping queries to lists of chunk IDs.
    Returns the two mappings.
    """
    # initialize client and DB
    client = load_gemini_client()
    ef = GeminiEmbeddingFunction(client=client, task_type="RETRIEVAL_DOCUMENT")
    db = get_or_create_chroma_db(client, db_name, ef, clear=True)

    # load and chunk document
    text = Path(text_file).read_text(encoding="utf-8")
    chunks = chunk_text(text, chunk_size, overlap)
    chunk_map = {
        f"chunk{str(i+1).zfill(3)}": chunk
        for i, chunk in enumerate(chunks)
    }

    # add to ChromaDB
    db.add(
        ids=list(chunk_map.keys()),
        documents=list(chunk_map.values())
    )

    # load queries
    raw = json.loads(Path(queries_file).read_text(encoding="utf-8"))
    # support either list or dict of queries
    queries = list(raw) if isinstance(raw, dict) else raw

    # retrieve top-k chunk IDs per query
    query_map: dict[str, list[str]] = {}
    for q in queries:
        result = db.query(query_texts=[q], n_results=top_k)
        ids = result.get("ids", [[]])[0]
        query_map[q] = ids

    # write outputs
    Path(output_chunk_json).write_text(json.dumps(chunk_map, indent=2), encoding="utf-8")
    Path(output_query_json).write_text(json.dumps(query_map, indent=2), encoding="utf-8")

    return chunk_map, query_map


if __name__ == "__main__":
    # Example usage:
    #   put your long document in "document.txt"
    #   put your queries list or dict in "queries.json"
    chunk_mapping, query_mapping = generate_chunk_and_query_mappings(
        text_file="example_document.txt",
        queries_file="example_queries.json",
        output_chunk_json="chunks.json",
        output_query_json="queries.json",
        chunk_size=200,
        overlap=40,
        top_k=10,
        db_name="my_doc_db"
    )
    print("Generated chunks.json and queries.json")

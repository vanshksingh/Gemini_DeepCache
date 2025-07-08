import os
import json
from pathlib import Path
from dotenv import load_dotenv
import asyncio

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction
from google import genai
from google.genai import types

# === CONFIGURATION ===
CHUNK_SIZE_DEFAULT = 200
OVERLAP_DEFAULT = 40
TOP_K_DEFAULT = 10
DB_PATH = "./chroma_store"


def _load_gemini_client() -> genai.Client:
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


async def _async_generate_chunk_and_query_mappings(
    text_file: str,
    queries_file: str,
    output_chunk_json: str,
    output_query_json: str,
    chunk_size: int,
    overlap: int,
    top_k: int,
    db_name: str
) -> tuple[dict[str, str], dict[str, list[str]]]:
    # 1. Init client & DB
    client = await asyncio.get_running_loop().run_in_executor(None, _load_gemini_client)
    ef = GeminiEmbeddingFunction(client=client, task_type="RETRIEVAL_DOCUMENT")
    def _get_db():
        store = chromadb.PersistentClient(path=DB_PATH)
        try:
            store.delete_collection(name=db_name)
        except ValueError:
            pass
        return store.get_or_create_collection(name=db_name, embedding_function=ef)
    db = await asyncio.get_running_loop().run_in_executor(None, _get_db)

    # 2. Read & chunk document
    text = Path(text_file).read_text(encoding="utf-8")
    chunks = chunk_text(text, chunk_size, overlap)
    chunk_map = {f"chunk{str(i+1).zfill(3)}": c for i, c in enumerate(chunks)}

    # 3. Add chunks to DB
    await asyncio.get_running_loop().run_in_executor(
        None,
        lambda: db.add(ids=list(chunk_map.keys()), documents=list(chunk_map.values()))
    )

    # 4. Load queries
    raw = json.loads(Path(queries_file).read_text(encoding="utf-8"))
    queries = list(raw) if isinstance(raw, dict) else raw

    # 5. Query top-k concurrently
    async def fetch_ids(q: str):
        res = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: db.query(query_texts=[q], n_results=top_k)
        )
        return q, res.get("ids", [[]])[0]

    pairs = await asyncio.gather(*(fetch_ids(q) for q in queries))
    query_map = dict(pairs)

    # 6. Write outputs
    Path(output_chunk_json).write_text(json.dumps(chunk_map, indent=2), encoding="utf-8")
    Path(output_query_json).write_text(json.dumps(query_map, indent=2), encoding="utf-8")

    return chunk_map, query_map


def generate_chunk_and_query_mappings(
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
    Synchronous facade for async chunk/query mapper.
    """
    return asyncio.run(
        _async_generate_chunk_and_query_mappings(
            text_file,
            queries_file,
            output_chunk_json,
            output_query_json,
            chunk_size,
            overlap,
            top_k,
            db_name
        )
    )


if __name__ == "__main__":
    load_dotenv()
    chunk_map, query_map = generate_chunk_and_query_mappings(
        text_file="example_document.txt",
        queries_file="example_queries.json",
        output_chunk_json="chunks.json",
        output_query_json="queries.json",
        chunk_size=200,
        overlap=40,
        top_k=10,
        db_name="my_sync_db"
    )
    print(f"Generated {len(chunk_map)} chunks and {len(query_map)} query mappings.")

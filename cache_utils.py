import os
import time
import pathlib
import datetime
from typing import Union, List, Any, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types


# === INIT ===
def load_gemini_client() -> genai.Client:
    """
    Load environment variables and initialize a Gemini client.
    Requires GEMINI_API_KEY in .env
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY in .env")
    return genai.Client(api_key=api_key)


# === FILE UTILITIES ===
def upload_file(client: genai.Client, path: Union[str, pathlib.Path]) -> Any:
    """
    Upload a file to the Gemini Files API and wait until processing is complete.
    Returns the uploaded file resource.
    """
    path = pathlib.Path(path)
    file_obj = client.files.upload(file=path)
    while file_obj.state.name == 'PROCESSING':
        # Avoid tight spin loops
        time.sleep(2)
        file_obj = client.files.get(name=file_obj.name)
    return file_obj


# === CACHE UTILITIES ===
def create_explicit_cache(
    client: genai.Client,
    model: str,
    contents: List[types.Content],
    system_instruction: str,
    ttl_seconds: int = 360,
    display_name: str = "cache"
) -> Any:
    """
    Create an explicit cache for `model`.

    Parameters
    ----------
    client : genai.Client
    model : str
        Fully-qualified model id, e.g. "models/gemini-2.0-flash-001"
    contents : List[types.Content]
        Pre-constructed messages with roles. Typically role="user".
    system_instruction : str
    ttl_seconds : int
    display_name : str

    Returns
    -------
    Cache resource (cachedContents/...)
    """
    cache = client.caches.create(
        model=model,
        config=types.CreateCachedContentConfig(
            display_name=display_name,
            system_instruction=system_instruction,
            contents=contents,
            ttl=f"{ttl_seconds}s"
        )
    )
    return cache


def generate_from_cache(
    client: genai.Client,
    model: str,
    cache_name: str,
    prompt: str
) -> Any:
    """
    Use an existing explicit cache to generate content with `prompt`.
    Ensures valid roles and passes cached_content via GenerateContentConfig.
    """
    return client.models.generate_content(
        model=model,
        contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
        config=types.GenerateContentConfig(cached_content=cache_name)
    )


def list_caches(client: genai.Client) -> List[Any]:
    """List metadata for all caches."""
    return list(client.caches.list())


def get_cache_metadata(client: genai.Client, name: str) -> Any:
    """Retrieve metadata for a single cache by `name`."""
    return client.caches.get(name=name)


def update_cache_ttl(client: genai.Client, name: str, ttl_seconds: int) -> Any:
    """Update the TTL for cache `name` to `ttl_seconds`."""
    return client.caches.update(
        name=name,
        config=types.UpdateCachedContentConfig(ttl=f"{ttl_seconds}s")
    )


def update_cache_expiry_time(
    client: genai.Client,
    name: str,
    expire_time: datetime.datetime
) -> Any:
    """
    Update the expire_time for cache `name` to a timezone-aware datetime.
    """
    if expire_time.tzinfo is None or expire_time.tzinfo.utcoffset(expire_time) is None:
        raise ValueError("expire_time must be timezone-aware (e.g., UTC).")
    return client.caches.update(
        name=name,
        config=types.UpdateCachedContentConfig(expire_time=expire_time)
    )


def delete_cache(client: genai.Client, name: str) -> None:
    """Delete cache with `name`."""
    client.caches.delete(name=name)


# === FILE METADATA UTILITIES ===
def list_files(client: genai.Client) -> List[Any]:
    """List all uploaded files."""
    return list(client.files.list())


def get_file_metadata(client: genai.Client, file_name: str) -> Any:
    """Get metadata for a single uploaded file by `file_name`."""
    return client.files.get(name=file_name)


def delete_file(client: genai.Client, file_name: str) -> None:
    """Delete an uploaded file by `file_name`."""
    client.files.delete(name=file_name)


# === Optional local demo (kept minimal and non-network-heavy by default) ===
if __name__ == "__main__":
    # Example scaffold (commented to avoid accidental network calls)
    # client = load_gemini_client()
    # model_id = "models/gemini-2.0-flash-001"
    #
    # # Example: create a tiny text cache from a couple of user messages
    # contents = [
    #     types.Content(role="user", parts=[types.Part(text="Project context: ACME RAG system design.")]),
    #     types.Content(role="user", parts=[types.Part(text="Key terms: chunking, hybrid search, cache TTL.")]),
    # ]
    # cache = create_explicit_cache(
    #     client=client,
    #     model=model_id,
    #     contents=contents,
    #     system_instruction="Answer concisely using the cached context.",
    #     ttl_seconds=300,
    #     display_name="tiny_demo_cache"
    # )
    # print("Cache created:", cache.name)
    #
    # resp = generate_from_cache(
    #     client=client,
    #     model=model_id,
    #     cache_name=cache.name,
    #     prompt="What are the key terms and why do they matter?"
    # )
    # print(resp.text)
    #
    # delete_cache(client, cache.name)
    # print("Deleted cache.")
    pass

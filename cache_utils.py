import os
import time
import pathlib
import datetime
import requests
from typing import Union, List, Any

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
    return genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# === FILE UTILITIES ===
def download_file(url: str, dest_path: Union[str, pathlib.Path]) -> pathlib.Path:
    """
    Download a file from `url` to `dest_path` if it does not already exist.
    Returns the Path to the downloaded file.
    """
    path = pathlib.Path(dest_path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('wb') as wf:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=32768):
                wf.write(chunk)
    return path


def upload_file(client: genai.Client, path: Union[str, pathlib.Path]) -> Any:
    """
    Upload a file to Gemini Files API and wait until processing is complete.
    Returns the uploaded file resource.
    """
    file_obj = client.files.upload(file=pathlib.Path(path))
    while file_obj.state.name == 'PROCESSING':
        print('Waiting for file to be processed...')
        time.sleep(2)
        file_obj = client.files.get(name=file_obj.name)
    return file_obj


# === CACHE UTILITIES ===
def create_explicit_cache(
    client: genai.Client,
    model: str,
    contents: List[Any],
    system_instruction: str,
    ttl_seconds: int = 360,
    display_name: str = "cache"
) -> Any:
    """
    Create an explicit cache for `model` with given contents and system instruction.
    TTL is in seconds. Returns the cache resource.
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
    Use an existing cache to generate content with `prompt`.
    Returns the generation response.
    """
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(cached_content=cache_name)
    )
    return response


def list_caches(client: genai.Client) -> List[Any]:
    """
    List metadata for all caches.
    """
    return list(client.caches.list())


def get_cache_metadata(client: genai.Client, name: str) -> Any:
    """
    Retrieve metadata for a single cache by `name`.
    """
    return client.caches.get(name=name)


def update_cache_ttl(client: genai.Client, name: str, ttl_seconds: int) -> Any:
    """
    Update the TTL for cache `name` to `ttl_seconds`.
    """
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
    return client.caches.update(
        name=name,
        config=types.UpdateCachedContentConfig(expire_time=expire_time)
    )


def delete_cache(client: genai.Client, name: str) -> None:
    """
    Delete cache with `name`.
    """
    client.caches.delete(name=name)


# === FILE METADATA UTILITIES ===
def list_files(client: genai.Client) -> List[Any]:
    """
    List all uploaded files.
    """
    return list(client.files.list())


def get_file_metadata(client: genai.Client, file_name: str) -> Any:
    """
    Get metadata for a single uploaded file by `file_name`.
    """
    return client.files.get(name=file_name)


def delete_file(client: genai.Client, file_name: str) -> None:
    """
    Delete an uploaded file by `file_name`.
    """
    client.files.delete(name=file_name)


# === USAGE EXAMPLE ===
if __name__ == "__main__":
    # Initialize client
    client = load_gemini_client()
    model_id = "models/gemini-2.0-flash-001"

    # --- FILE UPLOAD EXAMPLE ---
    video_url = (
        "https://storage.googleapis.com/generativeai-downloads/data/SherlockJr._10min.mp4"
    )
    video_path = download_file(video_url, "./SherlockJr._10min.mp4")
    uploaded_video = upload_file(client, video_path)
    print("Uploaded file URI:", uploaded_video.uri)

    # --- EXPLICIT CACHE EXAMPLE ---
    cache = create_explicit_cache(
        client=client,
        model=model_id,
        contents=[uploaded_video],
        system_instruction=(
            "You are an expert video analyzer. Answer queries based on the video file."
        ),
        ttl_seconds=300,
        display_name="sherlock_jr_cache"
    )
    print("Cache created:", cache.name)

    # Generate with cache
    prompt_text = (
        "List characters introduced in the movie with descriptions and timestamps."
    )
    response = generate_from_cache(
        client=client,
        model=model_id,
        cache_name=cache.name,
        prompt=prompt_text
    )
    print("Usage metadata:", response.usage_metadata)
    print("Response text:\n", response.text)

    # --- CACHE LISTING EXAMPLE ---
    all_caches = list_caches(client)
    for c in all_caches:
        print(c.name, c.create_time, c.expire_time)

    # --- CACHE METADATA EXAMPLE ---
    meta = get_cache_metadata(client, cache.name)
    print("Cache metadata:", meta)

    # --- CACHE UPDATE EXAMPLES ---
    update_cache_ttl(client, cache.name, ttl_seconds=600)
    print("Updated TTL to 600s")

    new_expiry = datetime.datetime.now(
        datetime.timezone.utc
    ) + datetime.timedelta(minutes=10)
    update_cache_expiry_time(client, cache.name, new_expiry)
    print("Updated expiry time to", new_expiry.isoformat())

    # --- CACHE DELETE EXAMPLE ---
    delete_cache(client, cache.name)
    print("Deleted cache", cache.name)

    # --- FILE METADATA EXAMPLES ---
    files = list_files(client)
    for f in files:
        print(f.name, f.create_time)

    file_meta = get_file_metadata(client, uploaded_video.name)
    print("File metadata:", file_meta)

    delete_file(client, uploaded_video.name)
    print("Deleted file", uploaded_video.name)

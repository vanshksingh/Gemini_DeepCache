import os
import time
import json
import pathlib
import datetime
from typing import List, Dict, Any, Optional, Union

from dotenv import load_dotenv
from google import genai
from google.genai import types


# === INIT ===
def load_gemini_client() -> genai.Client:
    """
    Load environment variables and return a configured Gemini client.
    Requires GEMINI_API_KEY in .env file.
    """
    load_dotenv()
    return genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# === INLINE BATCH JOBS ===
def create_inline_batch(
    client: genai.Client,
    model: str,
    inline_requests: List[Dict[str, Any]],
    display_name: str
) -> Any:
    """
    Create a batch job with inline GenerateContentRequest objects.
    Returns the created batch job resource.
    """
    batch_job = client.batches.create(
        model=model,
        src=inline_requests,
        config={"display_name": display_name}
    )
    return batch_job


# === FILE-BASED BATCH JOBS ===
def prepare_jsonl_file(
    requests: List[Dict[str, Any]],
    file_path: Union[str, pathlib.Path]
) -> pathlib.Path:
    """
    Write a list of request dicts to a JSONL file at file_path.
    Returns the pathlib.Path to the file.
    """
    path = pathlib.Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")
    return path


def upload_batch_file(
    client: genai.Client,
    file_path: Union[str, pathlib.Path],
    display_name: str = "batch_requests",
    mime_type: str = "application/jsonl"
) -> Any:
    """
    Upload a JSONL input file for batch processing via Files API.
    Returns the uploaded file resource.
    """
    uploaded = client.files.upload(
        file=str(file_path),
        config=types.UploadFileConfig(
            display_name=display_name,
            mime_type=mime_type
        )
    )
    return uploaded


def create_file_batch(
    client: genai.Client,
    model: str,
    src_file_name: str,
    display_name: str
) -> Any:
    """
    Create a batch job using an uploaded JSONL file name.
    Returns the created batch job resource.
    """
    batch_job = client.batches.create(
        model=model,
        src=src_file_name,
        config={"display_name": display_name}
    )
    return batch_job


# === MONITORING & RETRIEVAL ===
def wait_for_batch_completion(
    client: genai.Client,
    job_name: str,
    poll_interval: int = 30,
    timeout: Optional[int] = None
) -> Any:
    """
    Poll the batch job until it's done or timeout (in seconds) is reached.
    Returns the final batch job resource.
    """
    start = time.time()
    completed_states = {
        'JOB_STATE_SUCCEEDED',
        'JOB_STATE_FAILED',
        'JOB_STATE_CANCELLED'
    }
    job = client.batches.get(name=job_name)
    while job.state.name not in completed_states:
        print(f"Current state: {job.state.name}")
        time.sleep(poll_interval)
        if timeout and (time.time() - start) > timeout:
            raise TimeoutError(f"Batch job {job_name} did not complete within {timeout} seconds.")
        job = client.batches.get(name=job_name)
    print(f"Batch job completed with state: {job.state.name}")
    return job


def retrieve_batch_results(
    client: genai.Client,
    job: Any
) -> None:
    """
    Extract and print results from a completed batch job.
    Handles both inline and file-based outputs.
    """
    if job.state.name != 'JOB_STATE_SUCCEEDED':
        print(f"Job did not succeed. State: {job.state.name}")
        if job.error:
            print(f"Error: {job.error}")
        return

    dest = job.dest
    # File-based results
    if getattr(dest, 'file_name', None):
        print(f"Results in file: {dest.file_name}")
        content = client.files.download(file=dest.file_name)
        text = content.decode('utf-8')
        print(text)
        return

    # Inline results
    if getattr(dest, 'inlined_responses', None):
        for idx, ir in enumerate(dest.inlined_responses, start=1):
            print(f"Response {idx}:")
            if getattr(ir, 'response', None):
                try:
                    print(ir.response.text)
                except Exception:
                    print(ir.response)
            elif getattr(ir, 'error', None):
                print(f"Error: {ir.error}")
        return

    print("No results found.")


# === CANCEL & DELETE ===
def cancel_batch_job(client: genai.Client, job_name: str) -> Any:
    """
    Cancel a running batch job by name.
    Returns the updated job resource.
    """
    return client.batches.cancel(name=job_name)


def delete_batch_job(client: genai.Client, job_name: str) -> None:
    """
    Delete a batch job by name.
    """
    client.batches.delete(name=job_name)


# === USAGE EXAMPLE ===
if __name__ == "__main__":
    client = load_gemini_client()
    model = "models/gemini-2.5-flash"

    # --- INLINE BATCH EXAMPLE ---
    inline_requests = [
        { 'contents': [{ 'parts': [{ 'text': 'Tell me a one-sentence joke.' }], 'role': 'user' }] },
        { 'contents': [{ 'parts': [{ 'text': 'Why is the sky blue?' }], 'role': 'user' }] }
    ]
    inline_job = create_inline_batch(
        client, model, inline_requests, display_name="inline-jokes"
    )
    print(f"Created inline batch job: {inline_job.name}")

    # Poll and retrieve inline results
    job_done = wait_for_batch_completion(client, inline_job.name)
    retrieve_batch_results(client, job_done)

    # --- FILE BATCH EXAMPLE ---
    requests = [
        { "key": "r1", "request": { "contents": [{ "parts": [{ "text": "Describe photosynthesis." }] }] } },
        { "key": "r2", "request": { "contents": [{ "parts": [{ "text": "Main ingredients of Margherita pizza?" }] }] } }
    ]
    jsonl_path = prepare_jsonl_file(requests, "./batch_requests.jsonl")
    uploaded = upload_batch_file(
        client, jsonl_path, display_name="photosyn-pizza-requests"
    )
    print(f"Uploaded batch requests file: {uploaded.name}")

    file_job = create_file_batch(
        client, model, src_file_name=uploaded.name, display_name="file-batch-job"
    )
    print(f"Created file batch job: {file_job.name}")

    # Poll and retrieve file-based results
    job_file_done = wait_for_batch_completion(client, file_job.name)
    retrieve_batch_results(client, job_file_done)

    # --- CANCEL & DELETE EXAMPLE ---
    # cancel_batch_job(client, inline_job.name)
    # delete_batch_job(client, inline_job.name)
    # delete_batch_job(client, file_job.name)

import os
import json
import logging
import time
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from cache_utils import (
    create_explicit_cache,
    generate_from_cache,
    delete_cache
)
from vectordb import generate_chunk_and_query_mappings
from gem_cache import plan_batches

# === Configuration ===
TEXT_FILE = "example_document.txt"
QUERIES_FILE = "example_queries.json"
CHUNK_SIZE = 200
OVERLAP = 40
TOP_K = 10
DB_NAME = "my_sync_db"
MODEL = "gemini-2.0-flash"  # Use this model for both generate_content and cache
MAX_BATCH_SIZE = 7
IMPLICIT_THRESHOLD = 2048
CACHE_DISCOUNT = 0.25
CACHE_TTL = 60  # seconds for now
REGISTRY_PATH = "implicit_registry.pkl"
MIN_EXPLICIT_CHUNKS = 15
MAX_EXPLICIT_CHUNKS = 50
MIN_IMPLICIT_THRESHOLD = 225
MAX_IMPLICIT_THRESHOLD = 4096
MIN_CACHE_TTL = 60
MAX_CACHE_TTL = 3600
LOG_LEVEL = logging.WARNING
DISPLAY_CACHE_NAME = "gsoc_gemini_deepcache"
STATE_FILE = "pipeline_state.json"
ANSWERS_FILE = "answers.json"
ANSWERS_MAP_FILE = "answers_map.json"
# When True, delete state and answers files after completion
delete_state_on_exit = False
# If True, map each query individually to its answer
JSON_MODE = True
RETRY_LIMIT = 5
RETRY_DELAY = 2  # seconds between retries


def with_retries(fn, *args, **kwargs):
    last_exc = None
    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_exc = e
            print(f"Attempt {attempt}/{RETRY_LIMIT} failed for {fn.__name__}: {e}")
            if attempt < RETRY_LIMIT:
                time.sleep(RETRY_DELAY)
    print(f"All {RETRY_LIMIT} retries failed for {fn.__name__}.")
    raise last_exc


def init_client():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY in .env")
    return genai.Client(api_key=api_key)


def load_state():
    if Path(STATE_FILE).exists():
        content = Path(STATE_FILE).read_text()
        if not content.strip():
            print("[!] Empty state file; resetting.")
            return {}
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            print("[!] Corrupted state file; resetting.")
            return {}
    return {}


def save_state(state: dict):
    Path(STATE_FILE).write_text(json.dumps(state, indent=2))


def cleanup_files():
    for fn in (STATE_FILE, ANSWERS_FILE, ANSWERS_MAP_FILE):
        if Path(fn).exists():
            os.remove(fn)
            print(f"Deleted {fn}")


def step_generate_chunks(state: dict):
    print('[Step] Generating chunks & queries...')
    chunk_map, query_map = with_retries(
        generate_chunk_and_query_mappings,
        text_file=TEXT_FILE,
        queries_file=QUERIES_FILE,
        output_chunk_json='chunks.json',
        output_query_json='queries.json',
        chunk_size=CHUNK_SIZE,
        overlap=OVERLAP,
        top_k=TOP_K,
        db_name=DB_NAME
    )
    state['chunk_map'] = chunk_map
    state['query_map'] = query_map
    state['step'] = 'chunks_generated'
    save_state(state)


def step_create_plan(state: dict):
    print('[Step] Creating plan...')
    # plan_batches takes chunk_map and query_map and keyword args
    state['plan'] = with_retries(
        lambda: plan_batches(
            state['chunk_map'],
            state['query_map'],
            api_key=os.getenv('GEMINI_API_KEY'),
            model=MODEL,
            max_batch_size=MAX_BATCH_SIZE,
            implicit_threshold=IMPLICIT_THRESHOLD,
            cache_discount=CACHE_DISCOUNT,
            cache_ttl=CACHE_TTL,
            registry_path=REGISTRY_PATH,
            min_explicit_chunks=MIN_EXPLICIT_CHUNKS,
            max_explicit_chunks=MAX_EXPLICIT_CHUNKS,
            min_implicit_threshold=MIN_IMPLICIT_THRESHOLD,
            max_implicit_threshold=MAX_IMPLICIT_THRESHOLD,
            min_cache_ttl=MIN_CACHE_TTL,
            max_cache_ttl=MAX_CACHE_TTL,
            log_level=LOG_LEVEL
        )
    )
    state['step'] = 'plan_created'
    save_state(state)


def step_create_cache(state: dict, client):
    print('[Step] Creating explicit cache...')
    cache_plan = state['plan'].get('explicit_cache', [{}])[0]
    ctx = cache_plan.get('ctx_tokens', 0)
    state['cache_name'] = ''
    if ctx >= 4096:
        contents = [state['chunk_map'][cid] for cid in cache_plan.get('chunk_ids', [])]
        try:
            cache = with_retries(
                lambda: create_explicit_cache(
                    client=client,
                    model=MODEL,
                    contents=contents,
                    system_instruction="You are an expert assistant using cache.",
                    ttl_seconds=cache_plan.get('ttl', CACHE_TTL),
                    display_name=DISPLAY_CACHE_NAME
                )
            )
            state['cache_name'] = cache.name
            print(f"Cache created: {cache.name}")
        except Exception as e:
            print(f"[!] Cache creation failed after retries: {e}")
    else:
        print(f"Skipping cache: ctx={{ctx}} <4096 tokens")
    state['step'] = 'cache_created'
    save_state(state)


def step_execute_batches(state: dict, client):
    print('[Step] Executing queries...')
    mapping = {}
    tin, tout = 0, 0
    if JSON_MODE:
        for batch in state['plan'].get('batches', []):
            for q in batch.get('batch_queries', []):
                def call_query():
                    contents = []
                    if state['cache_name']:
                        contents.append(types.Content(parts=[types.Part(text=f"@use_cache {state['cache_name']}")]))
                    return client.models.generate_content(model=MODEL, contents=contents + [q])
                try:
                    resp = with_retries(call_query)
                    tin += resp.usage_metadata.prompt_token_count
                    tout += resp.usage_metadata.candidates_token_count
                    mapping[q] = resp.text
                    print(f"Answered '{q[:30]}...': {resp.usage_metadata.total_token_count} tokens")
                except Exception as e:
                    print(f"[!] Query failed after retries: {e}")
                    break
        Path(ANSWERS_MAP_FILE).write_text(json.dumps(mapping, indent=2))
    else:
        results = []
        for batch in state['plan'].get('batches', []):
            def call_batch():
                contents = []
                if state['cache_name']:
                    contents.append(types.Content(parts=[types.Part(text=f"@use_cache {state['cache_name']}")]))
                return client.models.generate_content(model=MODEL, contents=contents + batch.get('batch_queries', []))
            try:
                resp = with_retries(call_batch)
                tin += resp.usage_metadata.prompt_token_count
                tout += resp.usage_metadata.candidates_token_count
                results.append({'group_id': batch['group_id'], 'text': resp.text})
                print(f"Batch {batch['group_id']}: {resp.usage_metadata.total_token_count} tokens")
            except Exception as e:
                print(f"[!] Batch failed after retries: {e}")
                break
        Path(ANSWERS_FILE).write_text(json.dumps(results, indent=2))
    state.update({'input_tokens': tin, 'output_tokens': tout})
    state['step'] = 'batches_executed'
    save_state(state)


def step_cleanup(state: dict, client):
    print('[Step] Cleaning up...')
    if state['cache_name']:
        try:
            with_retries(lambda: delete_cache(client, state['cache_name']))
            print(f"Deleted cache {state['cache_name']}")
        except Exception as e:
            print(f"[!] Cleanup failed after retries: {e}")
    state['step'] = 'cleanup_done'
    save_state(state)


def step_report(state: dict):
    print('[Step] Report:')
    s = state['plan']['summary']
    raw, opt = s['total_raw_tokens'], s['total_optimized_tokens']
    print(f"Planned raw={raw}, opt={opt}, saved={raw-opt} ({s['saving_percentage']}%)")
    print(f"Actual in={state['input_tokens']}, out={state['output_tokens']}, total={state['input_tokens']+state['output_tokens']}")
    state['step'] = 'report_done'
    save_state(state)

if __name__ == '__main__':
    logging.basicConfig(level=LOG_LEVEL)
    client = init_client()
    state = load_state()
    try:
        if state.get('step') != 'chunks_generated': step_generate_chunks(state)
        if state['step']=='chunks_generated': step_create_plan(state)
        if state['step']=='plan_created': step_create_cache(state, client)
        if state['step']=='cache_created': step_execute_batches(state, client)
        if state['step']=='batches_executed': step_cleanup(state, client)
        if state['step']=='cleanup_done': step_report(state)
    except Exception as e:
        print(f"[!] Pipeline error: {e}")
        save_state(state)
    else:
        print(f"Done: {state['step']}")
        if delete_state_on_exit: cleanup_files()


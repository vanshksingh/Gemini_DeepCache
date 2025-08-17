import os
import json
import logging
import time
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv
from google import genai
from google.genai import types

from cache_utils import (
    create_explicit_cache,
    generate_from_cache,   # kept for completeness; not used directly in this file
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
# Prefer fully qualified model id
MODEL = "models/gemini-2.0-flash-001"  # works for both generate_content and caches
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
FINAL_TEXT_FILE = "final_output.txt"   # <— NEW

# When True, delete state and answers files after completion
delete_state_on_exit = False

# If True, map each query individually to its answer
JSON_MODE = True

RETRY_LIMIT = 5
RETRY_DELAY = 2  # seconds between retries


# === Helpers ===
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


def to_user_content(text: str) -> types.Content:
    """Wrap plain text into a valid user message for the GenAI SDK."""
    return types.Content(role="user", parts=[types.Part(text=text)])


def call_model(
    client: genai.Client,
    model: str,
    prompt: str,
    cached_content_name: str | None = None
):
    """
    Generate with optional explicit cache handle.
    Ensures valid roles and correct explicit-cache wiring.
    """
    kwargs = {
        "model": model,
        "contents": [to_user_content(prompt)]
    }
    if cached_content_name:
        kwargs["config"] = types.GenerateContentConfig(cached_content=cached_content_name)
    return client.models.generate_content(**kwargs)


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
    for fn in (STATE_FILE, ANSWERS_FILE, ANSWERS_MAP_FILE, FINAL_TEXT_FILE):
        if Path(fn).exists():
            os.remove(fn)
            print(f"Deleted {fn}")


# === Steps ===
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
        # Wrap each chunk as a proper user message
        chunk_ids = cache_plan.get('chunk_ids', [])
        contents = [to_user_content(state['chunk_map'][cid]) for cid in chunk_ids]
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
        print(f"Skipping cache: ctx={ctx} < 4096 tokens")
    state['step'] = 'cache_created'
    save_state(state)


def step_execute_batches(state: dict, client):
    print('[Step] Executing queries...')
    tin, tout = 0, 0
    cache_name = state.get('cache_name') or None

    if JSON_MODE:
        mapping = {}
        for batch in state['plan'].get('batches', []):
            for q in batch.get('batch_queries', []):
                def _call():
                    return call_model(
                        client=client,
                        model=MODEL,
                        prompt=q,
                        cached_content_name=cache_name
                    )
                try:
                    resp = with_retries(_call)
                    if getattr(resp, "usage_metadata", None):
                        tin  += getattr(resp.usage_metadata, "prompt_token_count", 0) or 0
                        tout += getattr(resp.usage_metadata, "candidates_token_count", 0) or 0
                        total = getattr(resp.usage_metadata, "total_token_count", None)
                    else:
                        total = None
                    mapping[q] = resp.text
                    if total is not None:
                        print(f"Answered '{q[:30]}...': {total} tokens")
                    else:
                        print(f"Answered '{q[:30]}...'")
                except Exception as e:
                    print(f"[!] Query failed after retries: {e}")
                    break

        # Persist answers and also keep in state for final text generation
        Path(ANSWERS_MAP_FILE).write_text(json.dumps(mapping, indent=2))
        state['answers_mode'] = 'map'
        state['answers_map'] = mapping

    else:
        # Grouped mode
        results = []
        for batch in state['plan'].get('batches', []):
            texts = batch.get('batch_queries', [])
            combined_prompt = "Answer each question separately:\n\n" + "\n\n".join(
                f"Q{i+1}. {t}" for i, t in enumerate(texts)
            )

            def _call():
                return call_model(
                    client=client,
                    model=MODEL,
                    prompt=combined_prompt,
                    cached_content_name=cache_name
                )

            try:
                resp = with_retries(_call)
                if getattr(resp, "usage_metadata", None):
                    tin  += getattr(resp.usage_metadata, "prompt_token_count", 0) or 0
                    tout += getattr(resp.usage_metadata, "candidates_token_count", 0) or 0
                    total = getattr(resp.usage_metadata, "total_token_count", None)
                else:
                    total = None

                results.append({'group_id': batch['group_id'], 'text': resp.text})
                if total is not None:
                    print(f"Batch {batch['group_id']}: {total} tokens")
                else:
                    print(f"Batch {batch['group_id']}: done")
            except Exception as e:
                print(f"[!] Batch failed after retries: {e}")
                break

        # Persist answers and also keep in state for final text generation
        Path(ANSWERS_FILE).write_text(json.dumps(results, indent=2))
        state['answers_mode'] = 'batches'
        state['answers_batches'] = results

    state.update({'input_tokens': tin, 'output_tokens': tout})
    state['step'] = 'batches_executed'
    save_state(state)


def step_cleanup(state: dict, client):
    print('[Step] Cleaning up...')
    if state.get('cache_name'):
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
    saved = raw - opt
    print(f"Planned raw={raw}, opt={opt}, saved={saved} ({s['saving_percentage']}%)")

    in_tok = state.get('input_tokens', 0)
    out_tok = state.get('output_tokens', 0)
    print(f"Actual in={in_tok}, out={out_tok}, total={in_tok + out_tok}")

    # Persist a JSON report alongside the text one (optional, handy for tooling)
    report_json = {
        "planned_raw": raw,
        "planned_opt": opt,
        "planned_saved": saved,
        "saving_percentage": s['saving_percentage'],
        "actual_input_tokens": in_tok,
        "actual_output_tokens": out_tok,
        "actual_total_tokens": in_tok + out_tok
    }
    Path("report.json").write_text(json.dumps(report_json, indent=2))

    # Store for final text file generation
    state['report'] = report_json
    state['step'] = 'report_done'
    save_state(state)


# === NEW: Final text writer ===
def _format_header(title: str, ch: str = "=") -> str:
    return f"{title}\n{ch * len(title)}"


def _load_answers_from_disk_if_missing(state: dict):
    """
    If answers are not in memory (state), try to read from disk to make the
    final text robust even across runs.
    """
    if 'answers_mode' in state:
        return

    if Path(ANSWERS_MAP_FILE).exists():
        try:
            mapping = json.loads(Path(ANSWERS_MAP_FILE).read_text())
            state['answers_mode'] = 'map'
            state['answers_map'] = mapping
            return
        except Exception:
            pass

    if Path(ANSWERS_FILE).exists():
        try:
            batches = json.loads(Path(ANSWERS_FILE).read_text())
            state['answers_mode'] = 'batches'
            state['answers_batches'] = batches
            return
        except Exception:
            pass

    # Fallback if nothing is found
    state['answers_mode'] = 'none'


def _build_answers_text(state: dict) -> str:
    mode = state.get('answers_mode', 'none')
    lines = []
    if mode == 'map':
        mapping = state.get('answers_map', {})
        if not mapping:
            return "No answers captured.\n"
        lines.append(_format_header("Answers"))
        for i, (q, a) in enumerate(mapping.items(), start=1):
            lines.append(f"\nQ{i}. {q}\nA{i}. {a}\n")
        return "\n".join(lines) + "\n"
    elif mode == 'batches':
        batches = state.get('answers_batches', [])
        if not batches:
            return "No answers captured.\n"
        lines.append(_format_header("Answers by Batch"))
        for b in batches:
            gid = b.get('group_id', 'unknown')
            text = b.get('text', '')
            lines.append(f"\n[Group: {gid}]\n{text}\n")
        return "\n".join(lines) + "\n"
    else:
        return "No answers captured.\n"


def _build_report_text(state: dict) -> str:
    rpt = state.get('report', {})
    if not rpt:
        return "Report data not available.\n"

    lines = [
        _format_header("Report Summary"),
        f"Planned raw tokens     : {rpt.get('planned_raw', 'N/A')}",
        f"Planned optimized tokens: {rpt.get('planned_opt', 'N/A')}",
        f"Planned saved tokens    : {rpt.get('planned_saved', 'N/A')}",
        f"Savings percentage      : {rpt.get('saving_percentage', 'N/A')}",
        "",
        f"Actual input tokens     : {rpt.get('actual_input_tokens', 'N/A')}",
        f"Actual output tokens    : {rpt.get('actual_output_tokens', 'N/A')}",
        f"Actual total tokens     : {rpt.get('actual_total_tokens', 'N/A')}",
        ""
    ]
    return "\n".join(lines)


def step_write_final_text(state: dict):
    """
    Build a single human-friendly text file that includes:
      - Run metadata (timestamp, model, cache info)
      - Report summary
      - Answers (per-query or per-batch)
    """
    print('[Step] Writing final text output...')
    _load_answers_from_disk_if_missing(state)

    timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    cache_name = state.get('cache_name', '')
    plan = state.get('plan', {})
    plan_summary = plan.get('summary', {}) if isinstance(plan, dict) else {}
    planned_groups = plan_summary.get('groups', 'N/A')

    header = [
        _format_header("Gemini Pipeline Run"),
        f"Timestamp  : {timestamp}",
        f"Model      : {MODEL}",
        f"Cache Used : {bool(cache_name)}",
        f"Cache Name : {cache_name or 'N/A'}",
        f"JSON_MODE  : {JSON_MODE}",
        f"Planned Groups: {planned_groups}",
        ""
    ]

    report_text = _build_report_text(state)
    answers_text = _build_answers_text(state)

    body = "\n".join(header) + report_text + "\n" + answers_text

    Path(FINAL_TEXT_FILE).write_text(body, encoding="utf-8")
    print(f"Final text written to: {FINAL_TEXT_FILE}")

    state['step'] = 'final_written'
    save_state(state)


if __name__ == '__main__':
    logging.basicConfig(level=LOG_LEVEL)
    client = init_client()
    state = load_state()
    try:
        if state.get('step') != 'chunks_generated':
            step_generate_chunks(state)

        if state.get('step') == 'chunks_generated':
            step_create_plan(state)

        if state.get('step') == 'plan_created':
            step_create_cache(state, client)

        if state.get('step') == 'cache_created':
            step_execute_batches(state, client)

        if state.get('step') == 'batches_executed':
            step_cleanup(state, client)

        if state.get('step') == 'cleanup_done':
            step_report(state)

        # NEW: write a single consolidated text artifact
        if state.get('step') == 'report_done':
            step_write_final_text(state)

    except Exception as e:
        print(f"[!] Pipeline error: {e}")
        save_state(state)
    else:
        print(f"Done: {state.get('step')}")
        if delete_state_on_exit:
            cleanup_files()

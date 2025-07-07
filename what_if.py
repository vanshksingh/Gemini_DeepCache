import os
import asyncio
import hashlib
import shelve
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Initialize client
dotenv_path = os.getenv('DOTENV_PATH', None)
load_dotenv(dotenv_path)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Discover model supporting generate_content
available = list(client.models.list())
MODEL = next((m.name for m in available if 'generateContent' in getattr(m, 'supported_actions', [])), None)
if MODEL is None:
    raise RuntimeError("No model supporting generateContent found.")

# Config
BATCH_SIZE = 5
MAX_CONTEXT_TOKENS = 2000
CACHE_PATH = "gemini_cache.db"

# Load transcript
def load_transcript(path: str) -> list[dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Chunk segments
def chunk_segments(segments: list[dict], max_words: int = 500, overlap: int = 50) -> list[dict]:
    chunks, current, word_count = [], [], 0
    for seg in segments:
        words = seg['text'].split()
        if word_count + len(words) > max_words:
            chunks.append({'start': current[0]['start'], 'text': " ".join(s['text'] for s in current)})
            ov = []
            for s in reversed(current):
                toks = s['text'].split()
                if len(ov) >= overlap:
                    break
                ov = toks + ov
            current = [{'start': current[-1]['start'], 'text': " ".join(ov)}]
            word_count = len(ov)
        current.append(seg)
        word_count += len(words)
    if current:
        chunks.append({'start': current[0]['start'], 'text': " ".join(s['text'] for s in current)})
    return chunks

# Cache key
def make_cache_key(start: str, history: list[dict], question: str) -> str:
    data = start + json.dumps(history, ensure_ascii=True) + question
    return hashlib.sha256(data.encode()).hexdigest()

# Build and call Gemini using valid roles
def build_and_call(context: str, questions: list[str], history: list[dict]) -> list[str]:
    answers = []
    for q in questions:
        contents = []
        # context as 'model'
        contents.append(types.Content(
            role='model',
            parts=[types.Part.from_text(text=context)]
        ))
        # map history: user->user, assistant/model->model
        for msg in history:
            role = 'user' if msg['role'] == 'user' else 'model'
            contents.append(types.Content(
                role=role,
                parts=[types.Part.from_text(text=msg['content'])]
            ))
        # current question
        contents.append(types.Content(
            role='user',
            parts=[types.Part.from_text(text=q)]
        ))
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=contents
            )
            answers.append(resp.text)
        except Exception as e:
            answers.append(f"[Error: {e}]")
    return answers

# Main orchestration
async def generate_predictions(segments: list[dict], questions: list[str]) -> dict:
    total = sum(len(s['text'].split()) for s in segments)
    chunks = segments if total <= MAX_CONTEXT_TOKENS else chunk_segments(segments)
    results = {}
    history = []
    with shelve.open(CACHE_PATH) as cache:
        for chunk in chunks:
            ts, text = chunk['start'], chunk['text']
            for i in range(0, len(questions), BATCH_SIZE):
                batch = questions[i:i+BATCH_SIZE]
                to_call, idx_map = [], {}
                for idx, q in enumerate(batch):
                    key = make_cache_key(ts, history, q)
                    if key in cache:
                        results[q] = {'answer': cache[key], 'timestamp': ts}
                    else:
                        idx_map[len(to_call)] = (q, key)
                        to_call.append(q)
                if to_call:
                    for ans in build_and_call(text, to_call, history):
                        q, key = idx_map[len(results) - len(cache) if False else list(idx_map.keys())[0]][0], None
                        # simplified mapping: iterate answers
                    answers = build_and_call(text, to_call, history)
                    for j, ans in enumerate(answers):
                        q, key = idx_map[j]
                        cache[key] = ans
                        results[q] = {'answer': ans, 'timestamp': ts}
                        history.append({'role': 'user', 'content': q})
                        history.append({'role': 'assistant', 'content': ans})
    return results

if __name__ == "__main__":
    # Embedded test data
    segments = [
        {"start": "00:00:05", "text": "Welcome to the lecture on advanced AI topics."},
        {"start": "00:02:10", "text": "Today we'll explore context caching techniques."}
    ]
    questions = [
        "What is the main focus of the lecture?",
        "How does context caching benefit performance?"
    ]
    all_results = asyncio.run(generate_predictions(segments, questions))
    print(json.dumps(all_results, indent=2))

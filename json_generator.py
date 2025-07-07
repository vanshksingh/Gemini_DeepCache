import json
import random
from pathlib import Path

# === CONFIG ===
NUM_CHUNKS = 100
NUM_QUERIES = 300
CHUNKS_PER_QUERY = (10, 10)

CHUNKS_PATH = Path("chunks.json")
QUERIES_PATH = Path("queries.json")

# === CHUNK GENERATION ===
def generate_random_sentence():
    subjects = ["The system", "This feature", "You", "The user", "The car"]
    verbs = ["can activate", "adjusts", "displays", "controls", "monitors"]
    objects = [
        "the ambient lighting", "lane keep assist", "media settings",
        "tire pressure", "wireless charging", "navigation display",
        "cruise control", "voice assistant", "map routing",
        "mirror fold settings", "seat memory"
    ]
    endings = ["automatically.", "manually.", "as needed.", "in real time.", "on startup."]
    return f"{random.choice(subjects)} {random.choice(verbs)} {random.choice(objects)} {random.choice(endings)}"

def generate_chunks(num_chunks):
    return {f"chunk{str(i+1).zfill(3)}": generate_random_sentence() for i in range(num_chunks)}

# === QUERY GENERATION ===
def generate_random_query_text():
    templates = [
        "How do I {verb} {thing}?",
        "Can the system {verb} {thing}?",
        "Enable {thing} now",
        "Where are the settings for {thing}?",
        "I want to {verb} {thing}."
    ]
    verbs = ["enable", "disable", "change", "access", "reset"]
    things = ["cruise control", "ambient lights", "seat memory", "charging", "Bluetooth", "navigation", "lane assist", "mirror fold", "auto hold", "climate control"]
    return random.choice(templates).format(verb=random.choice(verbs), thing=random.choice(things))

def generate_queries(num_queries, chunk_keys):
    queries = {}
    for _ in range(num_queries):
        query = generate_random_query_text()
        chunk_count = random.randint(*CHUNKS_PER_QUERY)
        chunks = random.sample(chunk_keys, chunk_count)
        queries[query] = chunks
    return queries

# === SAVE TO FILE ===
def save_json(data, filename):
    Path(filename).write_text(json.dumps(data, indent=2))

# === MAIN ===
if __name__ == "__main__":
    if not CHUNKS_PATH.exists():
        chunks = generate_chunks(NUM_CHUNKS)
        save_json(chunks, CHUNKS_PATH)
        print(f"✅ Created {CHUNKS_PATH.name} with {NUM_CHUNKS} chunks.")
    else:
        print(f"📦 {CHUNKS_PATH.name} already exists. Skipping generation.")

    if not QUERIES_PATH.exists():
        with open(CHUNKS_PATH) as f:
            chunks = json.load(f)
        queries = generate_queries(NUM_QUERIES, list(chunks.keys()))
        save_json(queries, QUERIES_PATH)
        print(f"✅ Created {QUERIES_PATH.name} with {NUM_QUERIES} queries.")
    else:
        print(f"📦 {QUERIES_PATH.name} already exists. Skipping generation.")
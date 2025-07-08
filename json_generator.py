import json
import random
from pathlib import Path

# === DEFAULT CONFIG ===
DEFAULT_NUM_CHUNKS = 100
DEFAULT_NUM_QUERIES = 30
DEFAULT_CHUNKS_PER_QUERY = (10, 10)
DEFAULT_CHUNKS_PATH = Path("chunks.json")
DEFAULT_QUERIES_PATH = Path("queries.json")


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


def generate_chunks(num_chunks=DEFAULT_NUM_CHUNKS):
    """
    Returns a dict mapping chunk IDs to random sentences.
    """
    return {
        f"chunk{str(i+1).zfill(3)}": generate_random_sentence()
        for i in range(num_chunks)
    }


def generate_random_query_text():
    templates = [
        "How do I {verb} {thing}?",
        "Can the system {verb} {thing}?",
        "Enable {thing} now",
        "Where are the settings for {thing}?",
        "I want to {verb} {thing}."
    ]
    verbs = ["enable", "disable", "change", "access", "reset"]
    things = [
        "cruise control", "ambient lights", "seat memory", "charging",
        "Bluetooth", "navigation", "lane assist", "mirror fold",
        "auto hold", "climate control"
    ]
    return random.choice(templates).format(
        verb=random.choice(verbs),
        thing=random.choice(things)
    )


def generate_queries(
    num_queries=DEFAULT_NUM_QUERIES,
    chunks_per_query=DEFAULT_CHUNKS_PER_QUERY,
    chunk_keys=None
):
    """
    Returns a dict mapping random query texts to a list of chunk IDs.
    """
    if chunk_keys is None:
        raise ValueError("chunk_keys must be provided")

    min_c, max_c = chunks_per_query
    queries = {}
    for _ in range(num_queries):
        q = generate_random_query_text()
        count = random.randint(min_c, max_c)
        queries[q] = random.sample(chunk_keys, count)
    return queries


def save_json(data, path: Path):
    """
    Serializes 'data' to JSON at 'path'.
    """
    path.write_text(json.dumps(data, indent=2))


def delete_json_files(*paths: Path):
    """
    Deletes each file if it exists.
    """
    for p in paths:
        try:
            p.unlink()
            print(f"🗑️ Deleted {p}")
        except FileNotFoundError:
            print(f"⚠️ {p} not found, skipping.")


def main(
    num_chunks=DEFAULT_NUM_CHUNKS,
    num_queries=DEFAULT_NUM_QUERIES,
    chunks_per_query=DEFAULT_CHUNKS_PER_QUERY,
    chunks_path=DEFAULT_CHUNKS_PATH,
    queries_path=DEFAULT_QUERIES_PATH
):
    """
    Generates and saves chunks and queries JSON files.
    """
    # 1) Generate chunks if missing
    if not chunks_path.exists():
        chunks = generate_chunks(num_chunks)
        save_json(chunks, chunks_path)
        print(f"✅ Created {chunks_path.name} with {num_chunks} chunks.")
    else:
        print(f"📦 {chunks_path.name} already exists. Skipping chunk generation.")

    # 2) Generate queries if missing
    if not queries_path.exists():
        chunks = json.loads(chunks_path.read_text())
        queries = generate_queries(
            num_queries=num_queries,
            chunks_per_query=chunks_per_query,
            chunk_keys=list(chunks.keys())
        )
        save_json(queries, queries_path)
        print(f"✅ Created {queries_path.name} with {num_queries} queries.")
    else:
        print(f"📦 {queries_path.name} already exists. Skipping query generation.")


if __name__ == "__main__":
    # Default run
    main()

    # --- Examples of custom runs ---
    # main(num_chunks=50, num_queries=150, chunks_per_query=(5, 15))
    # main(chunks_path=Path("my_chunks.json"), queries_path=Path("my_queries.json"))

    # To delete both JSON files before regenerating:
    # delete_json_files(Path("chunks.json"), Path("queries.json"))

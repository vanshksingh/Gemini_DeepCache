import os
import time
import pathlib
import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types

# -------------------------------------
# 🔐 API Configuration
# -------------------------------------
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=API_KEY)

# -------------------------------------
# 📦 File Handling Utilities
# -------------------------------------
def download_if_missing(url: str, local_path: str):
    path = pathlib.Path(local_path)
    if not path.exists():
        print(f"Downloading from: {url}")
        with path.open('wb') as f:
            response = requests.get(url, stream=True)
            for chunk in response.iter_content(chunk_size=32768):
                f.write(chunk)
    return path


def wait_until_processed(file):
    while file.state.name == 'PROCESSING':
        print("Waiting for file to be processed...")
        time.sleep(2)
        file = client.files.get(name=file.name)
    print(f"✅ File ready: {file.uri}")
    return file

# -------------------------------------
# 📤 Upload + Cache
# -------------------------------------
def upload_video_and_cache(file_path, cache_name):
    uploaded = client.files.upload(file=file_path)
    processed_file = wait_until_processed(uploaded)

    model_id = "models/gemini-2.0-flash-001"

    cache = client.caches.create(
        model=model_id,
        config=types.CreateCachedContentConfig(
            display_name=cache_name,
            system_instruction="Analyze this short film and answer any relevant questions based on its content.",
            contents=[processed_file],
            ttl="300s"
        )
    )
    return cache, model_id

# -------------------------------------
# 🧠 Explicit Query via Cache
# -------------------------------------
def analyze_video(cache, model_id, query):
    response = client.models.generate_content(
        model=model_id,
        contents=query,
        config=types.GenerateContentConfig(cached_content=cache.name)
    )
    print(response.text)
    return response

# -------------------------------------
# 🧠 Implicit Caching: Character Simulation
# -------------------------------------
def run_character_simulation():
    model_id = "gemini-2.5-flash-preview-04-17"
    persona_prompt = (
        "Assume the identity of a seasoned general who has led numerous campaigns. "
        "Offer strategic, rational advice on maintaining troop morale and structure "
        "during prolonged conflict. Your tone should be composed, wise, and decisive."
    )

    response = client.models.generate_content(
        model=model_id,
        contents=persona_prompt
    )
    print("🧠 General's Counsel:\n", response.text)

    followup = "What should I do if desertion rates start increasing?"
    extended = persona_prompt + "\n\n" + followup

    response = client.models.generate_content(
        model=model_id,
        contents=extended
    )
    print("\n🎯 Follow-up Advice:\n", response.text)

# -------------------------------------
# 📼 Video Understanding with Multimodal Input
# -------------------------------------
def summarize_video_content(file):
    model_id = "gemini-2.5-flash-preview-04-17"
    parts = [
        types.Part.from_uri(file_uri=file.uri, mime_type=file.mime_type),
        types.Part(text="Summarize the key events and tone of this video.")
    ]

    response = client.models.generate_content(
        model=model_id,
        contents=[types.Content(role="user", parts=parts)],
        config=types.GenerateContentConfig(response_mime_type="text/plain")
    )

    print("📽️ Summary:\n", response.text)
    return response

# -------------------------------------
# 📺 YouTube Video Analysis (if supported)
# -------------------------------------
def analyze_youtube_video(youtube_url):
    model_id = "gemini-2.5-flash-preview-04-17"

    prompt = (
        "Please analyze this YouTube video by summarizing the following:\n"
        "1. Main argument or narrative\n"
        "2. Key topics discussed\n"
        "3. Any audience engagement or call-to-action\n"
        "4. Overall summary"
    )

    response = client.models.generate_content(
        model=model_id,
        contents=types.Content(parts=[
            types.Part(text=prompt),
            types.Part(file_data=types.FileData(file_uri=youtube_url))
        ])
    )
    print("📺 YouTube Summary:\n", response.text)

# -------------------------------------
# 🔁 Main Execution
# -------------------------------------
if __name__ == "__main__":
    video_file_path = download_if_missing(
        url="https://storage.googleapis.com/generativeai-downloads/data/SherlockJr._10min.mp4",
        local_path="SherlockJr._10min.mp4"
    )

    cache,model = upload_video_and_cache(video_file_path, cache_name="silent-film-cache")

    analyze_video(
        cache=cache,
        model_id=model,
        query="Who are the main characters in this short film and how are they portrayed?"
    )

    run_character_simulation()

    print("\n🧩 Performing Video Multimodal Summary")
    summarize_video_content(file=client.files.get(name=cache.contents[0].name))

    print("\n🧠 Performing YouTube Video Summary")
    analyze_youtube_video("https://www.youtube.com/watch?v=RDOMKIw1aF4")

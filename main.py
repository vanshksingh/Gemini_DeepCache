import os
import time
import datetime
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai import caching

# Load .env and configure API
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Upload file
file_name = "weekly-ai-papers.txt"
file = genai.upload_file(path=file_name)

# Wait for file to be processed
while file.state.name == "PROCESSING":
    print("Waiting for file to be processed...")
    time.sleep(2)
    file = genai.get_file(file.name)

print(f"File processing complete: {file.uri}")

# Create cache with 15-minute TTL
cache = caching.CachedContent.create(
    model="models/gemini-1.5-flash-001",
    display_name="ml papers of the week",
    system_instruction="You are an expert AI researcher, and your job is to answer user's query based on the file you have access to.",
    contents=[file],
    ttl=datetime.timedelta(minutes=15),
)

# Load model from cache
model = genai.GenerativeModel.from_cached_content(cached_content=cache)

# First Query
response = model.generate_content(["Can you please tell me the latest AI papers of the week?"])
print(response.text)

# Mamba-specific Query
response = model.generate_content(["Can you list the papers that mention Mamba? List the title of the paper and summary."])
print(response.text)

# Long-context innovations Query
response = model.generate_content(["What are some of the innovations around long context LLMs? List the title of the paper and summary."])
print(response.text)

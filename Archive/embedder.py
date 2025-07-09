from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

# Supported task types for embeddings:
# SEMANTIC_SIMILARITY         -> For assessing text similarity.
# CLASSIFICATION              -> For classifying texts with preset labels.
# CLUSTERING                  -> For grouping similar texts.
# RETRIEVAL_DOCUMENT          -> For retrieving documents based on query embeddings.
# RETRIEVAL_QUERY             -> For querying against a set of documents.
# QUESTION_ANSWERING          -> For optimizing embeddings in QA tasks.
# FACT_VERIFICATION           -> For checking factual consistency between claim and evidence.
# CODE_RETRIEVAL_QUERY        -> For querying code blocks using natural language.

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

result = client.models.embed_content(
    model="gemini-embedding-exp-03-07",
    contents="What is the meaning of life?",
    config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
)

print(result.embeddings)

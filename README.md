# Gemini DeepCache

<img width="969" alt="Screenshot 2025-07-10 at 4 22 27 AM" src="https://github.com/user-attachments/assets/c05f567e-27eb-44c3-b9f8-c645081ed5a9" />



🚀 GSoC 2025 Project with **Google DeepMind**  
**Efficient Batch Prediction with Long Context and Smart Caching using Gemini API**

---

## 📖 Overview

**Gemini DeepCache** is a modular, production-grade Python pipeline designed to efficiently answer large batches of queries over long documents (e.g. video transcripts, research papers) by:

- Minimizing token usage via **context caching**
- Supporting **long-context windows** (up to 32K tokens)
- Performing **semantic batching** to reduce API calls
- Utilizing **explicit and implicit cache mechanisms**
- Persisting state and handling retries automatically

This project demonstrates robust use of **Gemini 2.5 API** for scalable inference while saving cost and improving performance.

---

## 🔧 Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/vanshksingh/Gemini_DeepCache.git
cd Gemini_DeepCache
```

### 2. Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Get Your Gemini API Key

Visit: 👉 https://aistudio.google.com/app/apikey

- Click **"Create API Key"**
- Copy the key

### 5. Configure `.env`

Create a `.env` file in the root directory:

```env
GEMINI_API_KEY=your_api_key_here
```

---

## 📦 Repository Structure

```bash
Gemini_DeepCache/
├── main.py               # Orchestrates the full pipeline
├── gem_cache.py          # Cache-aware batch planning
├── cache_utils.py        # Explicit cache creation, usage, deletion
├── vectordb.py           # Chunk embedding and semantic mapping
├── example_document.txt  # Sample document
├── example_queries.json  # Sample questions
├── requirements.txt      # Python dependencies
├── README.md             # Documentation
```

---

## 🧠 Core Components Explained

### `main.py`

The pipeline controller with the following **steps**:
- `step_generate_chunks`: Chunk and embed input document
- `step_create_plan`: Optimize batch planning based on token cost
- `step_create_cache`: Upload cache blocks if explicit caching is possible
- `step_execute_batches`: Generate answers with Gemini model, using cache where possible
- `step_cleanup`: Delete explicit caches
- `step_report`: Output token savings and performance stats

Includes:
- State saving and resuming (`pipeline_state.json`)
- Retry logic on API/network failure (`with_retries`)

---

### `vectordb.py` – Semantic Chunking 🧩

- Splits documents into overlapping chunks
- Uses Gemini Embedding API to store and search with **ChromaDB**
- Supports semantic query-to-chunk mapping

---

### `gem_cache.py` – Batch Planning Engine 🧠

- Groups related queries together
- Determines what can be cached explicitly vs implicitly
- Plans token-efficient execution order
- Outputs a **plan** with TTLs, cache thresholds, and reuse strategy

---

### `cache_utils.py` – Context Caching 💾

- Create, manage, and delete **explicit cache blocks**
- Uses `@use_cache {name}` instruction for guaranteed reuse
- Helps achieve up to **75% per-token cost savings**

---

## 🔍 Example Usage

### Input Files

- `example_document.txt`: The long document (e.g. transcript)
- `example_queries.json`: Array of queries:
```json
[
  "What is context caching?",
  "How does semantic clustering help efficiency?",
  "Explain the role of Gemini embeddings."
]
```

### Run the Pipeline

```bash
python main.py
```

---
## 📊 Token Savings Report

After running the pipeline, several artifacts are generated to help you audit and visualize cost efficiency and Gemini API usage.

### ✅ `pipeline_state.json`

Stores internal state and token metrics. Example:

```json
{
  "plan": {
    "summary": {
      "total_raw_tokens": 70543,
      "total_optimized_tokens": 27069.0,
      "total_saved_tokens": 43474.0,
      "saving_percentage": 61.6
    }
  },
  "cache_name": "",
  "input_tokens": 323,
  "output_tokens": 10787
}
```
The total savings in input cost come to about **61.6%** in this example.

**Fields:**
- `total_raw_tokens`: Estimated token usage without optimization  
- `total_optimized_tokens`: Actual usage after batch planning and caching  
- `saving_percentage`: Overall savings (%)  
- `input_tokens`, `output_tokens`: Actual API usage stats for billing

---

### 📈 `Raw vs Optimized chart`

A visual trend of **raw vs optimized** token usage over multiple runs. It compares:
![final_token_savings_chart](https://github.com/user-attachments/assets/032df410-b319-458a-bb5c-3bfe069eeeb1)

- Total raw vs optimized tokens
- Input-side raw vs optimized tokens

---

### 📄 `token_savings_data.csv`

CSV file used to generate the chart. Includes for each run:
- `Run`
- `Raw Total Tokens`
- `Optimized Total Tokens`
- `Input Raw Tokens`
- `Optimized Input Tokens`


---

## 📏 Long Context Handling

- Supports large contexts (e.g., 10K–32K tokens)
- Chunked with overlap (`CHUNK_SIZE` and `OVERLAP` configurable)
- Automatically skips explicit caching for chunks <4096 tokens (customizable)
- Handles out-of-bound contexts by splitting into smaller pieces

---

## 📦 Batch Prediction Optimization

- Batches are grouped to **maximize token reuse**
- Each batch has its own `group_id` and reuses shared context
- Automatically selects between **explicit cache**, **implicit cache**, or **raw prompt**
- Configurable: `MAX_BATCH_SIZE`, `IMPLICIT_THRESHOLD`, `CACHE_TTL`, etc.

---

## 💾 Context Caching Logic

| Type        | Benefit                          | Cost   |
|-------------|----------------------------------|--------|
| **Explicit**| Guaranteed reuse, up to 75% cost saved | Requires one-time upload |
| **Implicit**| Reuse based on token overlap     | Less reliable, no guarantees |
| **No Cache**| Always sends full context        | Full token cost |

---

## 🛡️ Error Handling

All Gemini API calls are wrapped using:

```python
with_retries(fn, *args, **kwargs)
```

- Retries up to `RETRY_LIMIT` times
- Waits `RETRY_DELAY` seconds between attempts
- Logs and reports errors with context
- Resumes from last successful pipeline step using saved state

---

## ✅ Expected Output

- `answers_map.json`: Per-query results in JSON mode
```json
{
  "What is context caching?": "Context caching is a technique where...",
  ...
}
```

- `answers.json`: Grouped answers in batch mode
```json
[
  {"group_id": 1, "text": "..."},
  {"group_id": 2, "text": "..."}
]
```

- Console Report:
```
Planned raw=18300, opt=9400, saved=8900 (48.6%)
Actual in=800, out=1600, total=2400
```

---

## 🌐 GSoC Deliverable Targets

| Feature                | ✅ Completed |
|------------------------|-------------|
| Detailed Code Comments | ✅ Yes |
| Setup Instructions     | ✅ Yes |
| Batch Optimization     | ✅ Yes |
| Long Context Handling  | ✅ Yes |
| Context Caching        | ✅ Yes |
| Error Handling         | ✅ Yes |
| Modular Functions      | ✅ Yes |


---

## 🧑‍💻 Contributing

Want to build on top of DeepCache?

1. Fork this repo
2. Create a branch: `git checkout -b feature/xyz`
3. Commit your changes with tests and docs
4. Submit a Pull Request 🚀

---

## 📄 License

MIT License © 2025 [Vansh Kumar Singh](https://github.com/vanshksingh)

---

## 🔗 Useful Links

- [Gemini API Key](https://aistudio.google.com/app/apikey)
- [Gemini API Docs](https://ai.google.dev/docs)
- [Google DeepMind](https://deepmind.google/)

# 🚀 Gemini DeepCache  
**Efficient Batch Prediction with Long Context and Smart Caching**  
*A Google Summer of Code 2025 project with Google DeepMind*

---

## 📌 Overview

**Gemini DeepCache** is an open-source code sample demonstrating advanced techniques for answering multiple queries over large content (e.g., video transcripts, documents) using **Google’s Gemini API** with long-context support.

It focuses on reducing redundant computation and token usage by leveraging both **explicit and implicit context caching**, along with intelligent techniques like **chunk deduplication**, **query clustering**, and **cache-aware batching**.

---

## 🧠 Core Techniques

### 🔁 Context De-Duplication
Before sending context to Gemini, DeepCache identifies and removes overlapping or redundant chunks across multiple queries. This ensures the API processes each unique chunk only once — dramatically reducing token consumption in batched queries.

### 🧩 Chunk Packing & Semantic Overlap
Queries are semantically grouped, and their associated content is **merged into unified context windows** using vector-based similarity. This avoids repetition and maximizes cache hit potential across similar questions.

### 📊 Cache-Aware Query Scheduling *(Planned)*
Instead of sending queries in arbitrary order, DeepCache will include a scheduler that prioritizes queries based on **chunk reuse potential**, ensuring related queries are processed back-to-back to maximize Gemini’s caching effectiveness.

### 🧭 Learned Query Routing *(Planned)*
Prototype router to predict which cached context group a new query belongs to, using embedding similarity or few-shot classification. This enables dynamic reuse of earlier context blocks without recomputing embeddings.

### ✂️ Retrieval-Masked Generation *(Exploration Phase)*
Partial outputs will be reused where only a delta is needed — by masking overlapping context, the system avoids regenerating entire responses for slightly changed queries.

---

## ⚙️ Gemini-Specific Optimizations

- ✅ **Explicit Caching API**: Utilizes `cache_id` and `cache_type` parameters for shared context blocks.
- ✅ **Implicit Caching**: Automatically reuses identical tokens within session memory.
- ✅ **Dynamic Chunk Splitting**: Long documents are split into meaning-preserving segments.
- ✅ **Rate Limit Resilience**: Async-safe request handling (Streamlit-compatible strategies explored).

---

## 📈 Use Cases

- Question-answering over full-length **video lectures**, **meetings**, or **technical documents**
- **Multi-query pipelines** where questions share overlapping background
- Scenarios where **context length or API quota is a limiting factor**

---

## 🛠 Status

| Component                      | Status       |
|-------------------------------|--------------|
| Basic Chunking & Caching      | ✅ Complete   |
| Streamlit UI (Async WIP)      | ⚙️ In Progress |
| Query Scheduling Logic        | 🧠 Designing  |
| Retrieval-Masked Gen Logic    | 🔍 Exploring  |
| Learned Query Router          | 🧪 Researching|

---

## 🔗 Resources & References

- [Gemini Caching Docs (Explicit & Implicit)](https://ai.google.dev/gemini-api/docs/caching?hl=en)
- [Gemini Embeddings API](https://ai.google.dev/gemini-api/docs/embeddings)
- [Google Dev Blog on Implicit Caching](https://developers.googleblog.com/en/gemini-2-5-models-now-support-implicit-caching/)
- [Prompt Engineering Guide – Gemini Caching Notebook](https://github.com/dair-ai/Prompt-Engineering-Guide/blob/main/notebooks/gemini-context-caching.ipynb)
- [Asyncio with Streamlit - Medium Guide](https://sehmi-conscious.medium.com/got-that-asyncio-feeling-f1a7c37cab8b)

---

## 🤝 Contributing & Contact

This is an evolving project under Google Summer of Code 2025. Contributions, ideas, and suggestions are welcome!

Author: [Vansh Kumar Singh](https://github.com/vanshksingh)  
Project Mentor: Google DeepMind Team

---

## 📜 License

MIT License — Open to all. Please cite if used in derivative work or demos.


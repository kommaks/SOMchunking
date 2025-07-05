# SOMchunking

> **Flexible framework for chunking, anomaly-based splitting, and evaluation in Retrieval-Augmented Generation (RAG) and NLP pipelines, with Self-Organizing Maps (SOM) for semantic clustering.**

---

## 🚩 Project Overview

**SOMchunking** is an experimental and extensible library for *semantic chunking* of text, integrating multiple advanced algorithms—such as cosine distance, double-pass semantic grouping, and SOM-based anomaly detection.  
It’s designed for use in RAG pipelines, semantic search, QA, and text clustering, and supports fast GPU embedding and modular evaluation.

---

## 💡 Features

| Chunking Methods    | GPU Support | RAG Evaluation | SOM Clustering | Visualization | Extensible |
|---------------------|:-----------:|:--------------:|:--------------:|:-------------:|:----------:|
| Standard            | ✅          | ✅             |                | Colored Chunks| ✅         |
| Double-pass         | ✅          | ✅             |                | Colored Chunks| ✅         |
| **SOM-based**       | ✅          | ✅             | ✅             | Cluster Stats | ✅         |

- **Standard**: Split by cosine distance between sentence embeddings.
- **Double-pass**: Grouping with threshold-based merging (semantic-aware).
- **SOM-based**: Detects “anomalies” or cluster boundaries in text via a Self-Organizing Map.
- **Ready for RAG**: Supports embedding models, chunk indexing with FAISS, and automatic LLM-based answer generation/evaluation.
- **Highly modular**: Add new chunkers or plug in different models easily.
- **Batch experiments**: YAML-configurable for multi-dataset benchmarking.
- **GPU-optimized**: Designed for PyTorch and HuggingFace transformers.

---

## 📦 Installation

Requirements: `python >=3.8`, `torch`, `sentence-transformers`, `scikit-learn`, `faiss`, `nltk`, `transformers`, `pyyaml`, `datasets`, and optionally `llama-index`.

```bash
git clone https://github.com/kommaks/SOMchunking.git
cd SOMchunking
pip install -r requirements.txt
# Or, for development:
pip install -e .

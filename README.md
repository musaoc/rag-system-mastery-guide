# Mastering RAG Systems — Complete End-to-End Implementation

A production-grade Retrieval-Augmented Generation (RAG) framework incorporating semantic chunking, dense vector embeddings, cross-encoder re-ranking, and Google Gemini LLM synthesis.

[![Kaggle Notebook](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/lazer999/rag-guide-for-beginners-walkthrough-slides)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Field](https://img.shields.io/badge/Field-LLMs%20/%20Retrieval-Augmented%20Generation-brightgreen)](#)

---

## Table of Contents
- [Project Overview](#project-overview)
- [Key Highlights & Results](#key-highlights--results)
- [System Architecture & Workflow](#system-architecture--workflow)
- [Repository Structure](#repository-structure)
- [Quickstart & Reproduction](#quickstart--reproduction)
- [Dataset Details](#dataset-details)
- [Author & Acknowledgments](#author--acknowledgments)

---

## Project Overview

This repository provides the complete, production-structured implementation of the **[Mastering RAG Systems — Complete End-to-End Implementation](https://www.kaggle.com/code/lazer999/rag-guide-for-beginners-walkthrough-slides)** project originally published on Kaggle. 

The primary focus of this work is translating complex data into actionable machine learning solutions using disciplined data engineering, rigorous validation strategies, and clean, leak-free preprocessing pipelines.

---

## Key Highlights & Results

- Complete end-to-end RAG architecture: ingestion, bi-encoder embeddings, vector search, cross-encoder re-ranking, and LLM synthesis.
- Integrated `sentence-transformers` for dense document and query representations.
- Cross-Encoder re-ranking stage improving Top-K precision before feeding context to the generative model.
- 2D vector space PCA visualization plotting query proximity to semantic document clusters.
- Google Gemini API integration with prompt injection defenses and temperature controls.

---

## System Architecture & Workflow

The pipeline follows a structured, modular execution path:

```mermaid
flowchart TD
    A[User Query] --> B[Bi-Encoder Embedding]
    C[Knowledge Corpus] --> D[Vector Store]
    B & D --> E[Top-K Dense Retrieval]
    E --> F[Cross-Encoder Re-Ranking]
    F --> G[Augmented Prompt Synthesis]
    G --> H[Google Gemini Generative AI]
    H --> I[Accurate Grounded Response]
```

---

## Repository Structure

```plaintext
rag-system-mastery-guide/
├── notebooks/
│   └── rag-system-mastery-guide.ipynb      # Original Jupyter notebook with full exploratory visuals
├── src/
│   └── main.py                # Modular, executable Python pipeline
├── .gitignore                 # Standard Python/Jupyter ignores
├── LICENSE                    # MIT License
├── README.md                  # Human-friendly documentation
└── requirements.txt           # Verified Python dependencies
```

---

## Quickstart & Reproduction

### 1. Clone the Repository
```bash
git clone https://github.com/musaoc/rag-system-mastery-guide.git
cd rag-system-mastery-guide
```

### 2. Set Up a Virtual Environment
```bash
# Linux / macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Pipeline
You can run the end-to-end script directly:
```bash
python src/main.py
```

Or open and run the interactive notebook:
```bash
jupyter lab notebooks/rag-system-mastery-guide.ipynb
```

---

## Dataset Details

- **Dataset / Competition**: [Customer Support FAQ Dataset](https://www.kaggle.com/datasets/lazer999/rag-sample-data)
- **Origin Platform**: Kaggle
- For automated dataset downloading via Kaggle CLI:
  ```bash
  kaggle datasets download -d lazer999/rag-sample-data
  ```

---

## Author & Acknowledgments

- **Author**: **Muhammad Musa Khan** (Kaggle Master)
- **Kaggle Profile**: [@lazer999](https://www.kaggle.com/lazer999)
- **GitHub**: [@musaoc](https://github.com/musaoc)
- **Original Kaggle Solution**: [Mastering RAG Systems — Complete End-to-End Implementation](https://www.kaggle.com/code/lazer999/rag-guide-for-beginners-walkthrough-slides)

If you found this project helpful or insightful, please consider starring the repository ⭐!

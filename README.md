# Job Intelligence Platform
**Built with:** Python • Scikit-learn • Pydantic • Ollama • Retrieval-Augmented Generation

An end-to-end **AI system for intelligent job discovery** that combines **classical information retrieval, hybrid ranking, and retrieval-augmented generation (RAG)**.

This project demonstrates how modern job recommendation systems can be built by combining **traditional ML techniques with LLM-based reasoning**.


## Key Features

- **Candidate generation using TF-IDF retrieval** over job titles and descriptions
- **Hybrid ranking model** combining:
  - textual relevance
  - recency signals
  - popularity signals (views)
  - optional salary boost
- **Retrieval-Augmented Generation (RAG)** explanations using a **locally hosted LLM (Ollama)**
- **Offline evaluation framework** using **Recall@K and MRR@K** to compare ranking strategies
- **FastAPI microservice** exposing the search + RAG pipeline via a /search endpoint
The system is designed as a **modular ML pipeline**, allowing each stage to be independently improved or replaced.


## Architecture Overview
```
Raw Job Dataset
↓
Data Preprocessing (Pydantic validation)
↓
TF-IDF Retrieval Engine
↓
Hybrid Ranking Model
↓
Top-K Candidate Jobs
↓
RAG Explanation Layer (Local LLM)
```

This architecture reflects many **real-world recommendation and search systems**, where:

- retrieval generates candidate items
- ranking optimizes ordering
- LLMs provide interpretability


## Project Structure
```
src/
├── data/
│ ├── models.py # Pydantic schema (PreprocessedJob)
│ ├── preprocessing.py # data cleaning + validation
│ └── csv/ # raw & processed datasets (gitignored)
│
├── retrieval/
│ └── tfidf_engine.py # candidate generation
│
├── ranking/
│ └── scorer.py # hybrid ranking logic
│
├── rag/
│ ├── explainer.py # local LLM explanation
│ └── run_rag.py
│
├── evaluation/
│ └── metrics.py # Recall@K / MRR@K evaluation
│
├── api/
│ └── app.py # FastAPI service (search endpoint)
│
└── cli.py # unified CLI entrypoint
```

## Core System Components

### Retrieval Layer

Generates candidate jobs using **TF-IDF vectorization with cosine similarity**.

This stage efficiently narrows the search space by identifying jobs whose titles and descriptions are textually similar to the user query.


### Hybrid Ranking Model

Retrieved candidates are re-ranked using multiple signals.
```
Final Score =
α × textual similarity
+ β × recency score
+ γ × popularity score
+ optional salary boost
```

Signal normalization ensures stable ranking behavior across different features.


### Evaluation Framework

The platform includes an **offline evaluation pipeline** implementing:

- **Recall@K**
- **Mean Reciprocal Rank (MRR@K)**

This allows systematic comparison between:

- retrieval-only baseline
- raw hybrid ranking
- normalized ranking
- salary-boost ranking

Example output:

```
Retrieval only: Recall@10 = 0.70 MRR@10 = 0.45
Normalized ranking: Recall@10 = 0.68 MRR@10 = 0.43
Salary boost ranking: Recall@10 = 0.68 MRR@10 = 0.43
```


### RAG Explanation Layer

A local LLM (via **Ollama**) generates explanations for recommended jobs.

The model receives the **top ranked jobs as context** and produces grounded reasoning explaining why a job matches the user query.

Example:

```
Query: machine learning engineer

Top Job: Senior Machine Learning Engineer

Explanation:
This role matches the query because the job description emphasizes Python,
PyTorch, and experience developing machine learning pipelines.
```

This improves **interpretability and transparency** of recommendations.


## Setup

You can install dependencies using **uv** (recommended) or `pip`.

### Using uv

```
uv sync
```

This creates a virtual environment and installs dependencies from `pyproject.toml`.


### Using pip

```
pip install -r requirements.txt
```


## Dataset

Large CSV files are intentionally **excluded from the repository** due to GitHub file size limits.

Place datasets in:

```
src/data/csv/postings.csv
```

After preprocessing, the cleaned dataset will be saved as:

```
src/data/csv/jobs_clean.csv
```


## Configuration

Configuration is managed using **Pydantic Settings** in `src/config.py`.

Optional `.env` file:

```
LOG_LEVEL=INFO
OLLAMA_URL=http://host.docker.internal:11434/api/generate
OLLAMA_MODEL=tinyllama
```


## Usage

All commands should be run from the project root.


### 1. Preprocess dataset

```
python main.py preprocess --input src/data/csv/postings.csv --output src/data/csv/jobs_clean.csv
```


### 2. Retrieve and rank jobs

```
python main.py retrieve --query "entry level machine learning engineer" --candidate-k 50 --top-k 10
```

Disable ranking experiments if needed:

```
python main.py retrieve --no-normalize
python main.py retrieve --no-salary
```


### 3. Generate RAG explanation

Ensure Ollama is running:

```
ollama run phi3
```

Then run:

```
python main.py rag --query "entry level machine learning engineer" --rag-top-n 3
```


### 4. Run evaluation

```
python main.py eval
```

### 5. Run the FastAPI service

After preprocessing your data so that `src/data/csv/jobs_clean.csv` exists, you can start the HTTP API:

```bash
uvicorn src.api.app:app --reload

## Technologies Used

- **Python**
- **Scikit-learn**
- **Pydantic**
- **NumPy**
- **Pandas**
- **Ollama (local LLM inference)**
- **FastAPI**


## Future Improvements

Possible extensions include:

- semantic retrieval using sentence-transformer embeddings
- FAISS vector search
- online learning from user interaction signals
- agent-based job recommendation workflows


## Why This Project Matters

This project illustrates how **classical ML pipelines and modern LLM reasoning can be combined** to build intelligent recommendation systems.

It demonstrates key concepts used in real-world systems:

- candidate generation
- ranking models
- offline evaluation
- retrieval-augmented generation
- modular ML system design
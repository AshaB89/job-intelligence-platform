# Job Intelligence Platform

An end-to-end **job discovery and recommendation prototype** that combines:

- **Classical retrieval (TF‑IDF)** for fast candidate generation
- **Hybrid ranking** (re-ranking by relevance + recency + popularity + optional salary boost)
- **RAG-style explanations** (local Ollama by default) to explain *why* a job matches
- **Offline evaluation** (Recall@K, MRR@K) to compare retrieval vs ranking variants

This repo is designed to be **portfolio-friendly**: clean structure, typed data models, centralized configuration, and production-grade logging.

## Project structure

- `src/data/`
  - `models.py`: Pydantic schema (`PreprocessedJob`)
  - `preprocessing.py`: CSV cleaning + validation → `jobs_clean.csv`
  - `csv/`: local datasets (ignored by git; kept via `.gitkeep`)
- `src/retrieval/`: TF‑IDF retrieval engine + runner
- `src/ranking/`: hybrid ranker (normalization + salary boost toggles)
- `src/rag/`: retrieval + ranking + LLM explanation
- `src/evaluation/`: Recall@K + MRR@K evaluation runner
- `src/cli.py`: single entrypoint CLI for end-to-end runs

## Setup

You can use either **uv** (recommended) or plain `pip`.

### Using uv

```bash
uv sync
```

This will create/refresh the virtualenv and install dependencies from `pyproject.toml`.

### Using pip

```bash
pip install -r requirements.txt
```

### Data (not committed)

Large CSVs are intentionally **ignored** (GitHub file-size limits). Place your datasets here:

- `src/data/csv/postings.csv` (raw)
- `src/data/csv/jobs_clean.csv` (generated)

## Configuration

Configuration is centralized in `src/config.py` (Pydantic `Settings`).

Optional `.env` file (ignored by git):

```env
LOG_LEVEL=INFO
OLLAMA_URL=http://localhost:11434/api/generate
OLLAMA_MODEL=phi3
```

## Usage (CLI)

All commands are run from the project root.

### 1) Preprocess raw postings → clean dataset

```bash
python main.py preprocess --input src/data/csv/postings.csv --output src/data/csv/jobs_clean.csv
```

### 2) Retrieve + rank (hybrid)

```bash
python main.py retrieve --query "entry level machine learning engineer" --candidate-k 50 --top-k 10
```

Disable ranker normalization or salary boost for experiments:

```bash
python main.py retrieve --no-normalize
python main.py retrieve --no-salary
```

### 3) RAG explanation (local Ollama)

Make sure Ollama is running and the model exists (example):

```bash
ollama run phi3
```

Then:

```bash
python main.py rag --query "entry level machine learning engineer" --rag-top-n 3
```

### 4) Offline evaluation (Recall@10 / MRR@10)

```bash
python main.py eval
```

Example output:

```text
Retrieval only: Recall@10=0.70 MRR@10=0.45
Raw ranking: Recall@10=0.26 MRR@10=0.10
Normalized ranking: Recall@10=0.68 MRR@10=0.43
Salary boost: Recall@10=0.68 MRR@10=0.43
```

## Notes

- **Why `.gitignore` ignores datasets**: large files break GitHub pushes; keep data local and version the code.
- **Ranking controls**: `use_normalization` and `use_salary_boost` are exposed to make experiments reproducible.
- **Logging**: all runners use standard Python logging; configure with `LOG_LEVEL`.

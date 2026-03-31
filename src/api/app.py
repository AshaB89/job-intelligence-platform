from fastapi import FastAPI
import pandas as pd

from src.data.models import PreprocessedJob
from src.retrieval.tfidf_engine import TFIDFRetrievalEngine
from src.ranking.scorer import JobRanker
from src.rag.explainer import JobRAGExplainer


app = FastAPI(title="Job Intelligence API")


# Load dataset once when API starts
df = pd.read_csv("src/data/csv/jobs_clean.csv", dtype={"job_id": str})

jobs = [PreprocessedJob(**row.to_dict()) for _, row in df.iterrows()]

retriever = TFIDFRetrievalEngine(jobs)
ranker = JobRanker()
explainer = JobRAGExplainer()


@app.get("/")
def root():
    return {"message": "Job Intelligence Platform API"}


@app.get("/search")
def search_jobs(query: str):

    # retrieval
    candidates = retriever.retrieve(query, top_k=50)

    # ranking
    ranked_jobs = ranker.rank(candidates)

    top_jobs = ranked_jobs[:3]

    # explanation
    explanation = explainer.explain(query, [job for job, _ in top_jobs])

    results = [
        {
            "title": job.title,
            "location": job.location,
            "score": float(score)
        }
        for job, score in top_jobs
    ]

    return {
        "query": query,
        "results": results,
        "explanation": explanation
    }
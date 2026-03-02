import pandas as pd

from src.data.models import PreprocessedJob
from src.retrieval.tfidf_engine import TFIDFRetrievalEngine
from src.ranking.scorer import JobRanker


def main():

    # Load cleaned dataset from dedicated CSV folder
    df = pd.read_csv("src/data/csv/jobs_clean.csv", dtype={"job_id": str})

    jobs = [
        PreprocessedJob(**record)
        for record in df.to_dict(orient="records")
    ]

    engine = TFIDFRetrievalEngine(jobs)

    query = "machine learning engineer"

    # Step 1: retrieve candidates (IR)
    candidate_k = 50
    retrieved = engine.retrieve(query, top_k=candidate_k)

    # Step 2: hybrid re-ranking (learned-ish heuristic ranker)
    ranker = JobRanker(alpha=0.6, beta=0.2, gamma=0.1, delta=0.1)
    ranked = ranker.rank(retrieved)

    # Step 3: print final ranked results
    sim_by_job_id = {job.job_id: float(sim) for job, sim in retrieved}
    final_k = 10

    print(f"Query: {query}")
    print(f"Retrieved {len(retrieved)} candidates, showing top {final_k} after ranking.")
    print("-" * 100)

    for i, (job, final_score) in enumerate(ranked[:final_k], start=1):
        sim = sim_by_job_id.get(job.job_id, 0.0)

        print(
            f"{i:>2}. final={final_score:.4f} | sim={sim:.4f} | views={job.views:.0f} | "
            f"salary={job.normalized_salary:.0f} | {job.title}"
        )

if __name__ == "__main__":
    main()
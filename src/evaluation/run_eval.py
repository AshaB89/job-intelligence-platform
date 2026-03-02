import pandas as pd
from src.data.models import PreprocessedJob
from src.retrieval.tfidf_engine import TFIDFRetrievalEngine
from src.ranking.scorer import JobRanker
from src.evaluation.metrics import recall_at_k, mrr_at_k


def _evaluate(engine, jobs, ranker, k_retrieve=50, k_eval=10):
    recall_scores = []
    mrr_scores = []

    for job in jobs:
        query = job.title
        retrieved = engine.retrieve(query, top_k=k_retrieve)

        if ranker is None:
            recommended_ids = [j.job_id for j, _ in retrieved]
        else:
            ranked = ranker.rank(retrieved)
            recommended_ids = [j.job_id for j, _ in ranked]

        ground_truth = [job.job_id]
        recall_scores.append(recall_at_k(recommended_ids, ground_truth, k=k_eval))
        mrr_scores.append(mrr_at_k(recommended_ids, ground_truth, k=k_eval))

    return (sum(recall_scores) / len(recall_scores)), (sum(mrr_scores) / len(mrr_scores))


def main():

    # Use a capped corpus so evaluation runs quickly on large datasets
    corpus_rows = 5000
    eval_rows = 200

    df = pd.read_csv("src/data/csv/jobs_clean.csv", dtype={"job_id": str}, nrows=corpus_rows)

    jobs = [
        PreprocessedJob(**record)
        for record in df.to_dict(orient="records")
    ]

    engine = TFIDFRetrievalEngine(jobs)
    sample_jobs = jobs[: min(eval_rows, len(jobs))]

    # Configurations
    ranker_retrieval_only = None  # skip ranking
    ranker_raw = JobRanker(use_normalization=False, use_salary_boost=False)
    ranker_normalized = JobRanker(use_normalization=True, use_salary_boost=False)
    ranker_salary = JobRanker(use_normalization=True, use_salary_boost=True)

    # Evaluate
    r_recall, r_mrr = _evaluate(engine, sample_jobs, ranker_retrieval_only)
    raw_recall, raw_mrr = _evaluate(engine, sample_jobs, ranker_raw)
    norm_recall, norm_mrr = _evaluate(engine, sample_jobs, ranker_normalized)
    sal_recall, sal_mrr = _evaluate(engine, sample_jobs, ranker_salary)

    print("Retrieval only:")
    print("Recall@10:", round(r_recall, 2))
    print("MRR@10:", round(r_mrr, 2))
    print()

    print("Raw ranking (no normalization, no salary boost):")
    print("Recall@10:", round(raw_recall, 2))
    print("MRR@10:", round(raw_mrr, 2))
    print()

    print("Normalized ranking:")
    print("Recall@10:", round(norm_recall, 2))
    print("MRR@10:", round(norm_mrr, 2))
    print()

    print("Salary boost:")
    print("Recall@10:", round(sal_recall, 2))
    print("MRR@10:", round(sal_mrr, 2))


if __name__ == "__main__":
    main()
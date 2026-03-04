import pandas as pd
import logging
from src.data.models import PreprocessedJob
from src.retrieval.tfidf_engine import TFIDFRetrievalEngine
from src.ranking.scorer import JobRanker
from src.evaluation.metrics import recall_at_k, mrr_at_k

logger = logging.getLogger(__name__)

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

    from src.config import get_settings
    settings = get_settings()

    logger.info("Loading evaluation corpus: %s", settings.jobs_clean_csv)
    df = pd.read_csv(
        str(settings.jobs_clean_csv),
        dtype={"job_id": str},
        nrows=settings.eval_corpus_rows,
    )

    jobs = [
        PreprocessedJob(**record)
        for record in df.to_dict(orient="records")
    ]

    engine = TFIDFRetrievalEngine(jobs)
    sample_jobs = jobs[: min(settings.eval_rows, len(jobs))]

    # Configurations
    ranker_retrieval_only = None  # skip ranking
    ranker_raw = JobRanker(use_normalization=False, use_salary_boost=False)
    ranker_normalized = JobRanker(use_normalization=True, use_salary_boost=False)
    ranker_salary = JobRanker(use_normalization=True, use_salary_boost=True)

    # Evaluate
    r_recall, r_mrr = _evaluate(engine, sample_jobs, ranker_retrieval_only, k_retrieve=settings.eval_k_retrieve, k_eval=settings.eval_k)
    raw_recall, raw_mrr = _evaluate(engine, sample_jobs, ranker_raw, k_retrieve=settings.eval_k_retrieve, k_eval=settings.eval_k)
    norm_recall, norm_mrr = _evaluate(engine, sample_jobs, ranker_normalized, k_retrieve=settings.eval_k_retrieve, k_eval=settings.eval_k)
    sal_recall, sal_mrr = _evaluate(engine, sample_jobs, ranker_salary, k_retrieve=settings.eval_k_retrieve, k_eval=settings.eval_k)

    logger.info("Retrieval only: Recall@%s=%.2f MRR@%s=%.2f", settings.eval_k, r_recall, settings.eval_k, r_mrr)
    logger.info("Raw ranking: Recall@%s=%.2f MRR@%s=%.2f", settings.eval_k, raw_recall, settings.eval_k, raw_mrr)
    logger.info("Normalized ranking: Recall@%s=%.2f MRR@%s=%.2f", settings.eval_k, norm_recall, settings.eval_k, norm_mrr)
    logger.info("Salary boost: Recall@%s=%.2f MRR@%s=%.2f", settings.eval_k, sal_recall, settings.eval_k, sal_mrr)


if __name__ == "__main__":
    from src.config import get_settings
    from src.logging_config import configure_logging

    settings = get_settings()
    configure_logging(settings.log_level)
    main()
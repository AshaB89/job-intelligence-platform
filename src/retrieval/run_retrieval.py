import pandas as pd
import logging

from src.data.models import PreprocessedJob
from src.retrieval.tfidf_engine import TFIDFRetrievalEngine
from src.ranking.scorer import JobRanker

logger = logging.getLogger(__name__)

def main():

    from src.config import get_settings
    settings = get_settings()

    logger.info("Loading cleaned dataset: %s", settings.jobs_clean_csv)
    df = pd.read_csv(str(settings.jobs_clean_csv), dtype={"job_id": str})

    jobs = [
        PreprocessedJob(**record)
        for record in df.to_dict(orient="records")
    ]

    engine = TFIDFRetrievalEngine(jobs)

    query = settings.default_query

    # Step 1: retrieve candidates (IR)
    candidate_k = settings.candidate_k
    retrieved = engine.retrieve(query, top_k=candidate_k)

    # Step 2: hybrid re-ranking (learned-ish heuristic ranker)
    ranker = JobRanker(alpha=0.6, beta=0.2, gamma=0.1, salary_boost=0.1)
    ranked = ranker.rank(retrieved)

    # Step 3: print final ranked results
    sim_by_job_id = {job.job_id: float(sim) for job, sim in retrieved}
    final_k = settings.final_k

    logger.info("Query: %s", query)
    logger.info("Retrieved %s candidates, showing top %s after ranking.", len(retrieved), final_k)

    for i, (job, final_score) in enumerate(ranked[:final_k], start=1):
        sim = sim_by_job_id.get(job.job_id, 0.0)

        logger.info(
            "%2d. final=%.4f | sim=%.4f | views=%.0f | salary=%.0f | %s",
            i,
            final_score,
            sim,
            job.views,
            job.normalized_salary,
            job.title,
        )

if __name__ == "__main__":
    from src.config import get_settings
    from src.logging_config import configure_logging

    settings = get_settings()
    configure_logging(settings.log_level)
    main()
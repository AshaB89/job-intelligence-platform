import pandas as pd
import logging

from src.data.models import PreprocessedJob
from src.retrieval.tfidf_engine import TFIDFRetrievalEngine
from src.ranking.scorer import JobRanker
from src.rag.explainer import JobRAGExplainer

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
    ranker = JobRanker()
    explainer = JobRAGExplainer()

    query = settings.default_query

    retrieved = engine.retrieve(query, top_k=50)
    ranked = ranker.rank(retrieved)

    top_jobs = [job for job, _ in ranked[:3]]

    logger.info("Top Jobs:")

    for job in top_jobs:
        logger.info("- %s", job.title)

    explanation = explainer.explain(query, top_jobs)

    logger.info("LLM Explanation:\n%s", explanation)


if __name__ == "__main__":
    from src.config import get_settings
    from src.logging_config import configure_logging

    settings = get_settings()
    configure_logging(settings.log_level)
    main()
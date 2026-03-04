from __future__ import annotations

import argparse
import logging
from typing import List, Optional

import pandas as pd

from src.config import get_settings
from src.data.models import PreprocessedJob
from src.data.preprocessing import run_preprocessing
from src.evaluation.run_eval import main as run_eval_main
from src.logging_config import configure_logging
from src.rag.explainer import JobRAGExplainer
from src.ranking.scorer import JobRanker
from src.retrieval.tfidf_engine import TFIDFRetrievalEngine

logger = logging.getLogger(__name__)


def _load_jobs(csv_path: str, nrows: Optional[int] = None) -> List[PreprocessedJob]:
    df = pd.read_csv(csv_path, dtype={"job_id": str}, nrows=nrows)
    return [PreprocessedJob(**record) for record in df.to_dict(orient="records")]


def cmd_preprocess(args: argparse.Namespace) -> None:
    run_preprocessing(args.input, args.output)


def cmd_retrieve(args: argparse.Namespace) -> None:
    settings = get_settings()
    jobs = _load_jobs(args.data, nrows=args.nrows)
    engine = TFIDFRetrievalEngine(jobs)

    retrieved = engine.retrieve(args.query, top_k=args.candidate_k)
    ranker = JobRanker(
        use_normalization=not args.no_normalize,
        use_salary_boost=not args.no_salary,
    )
    ranked = ranker.rank(retrieved)

    logger.info("Query: %s", args.query)
    logger.info("Retrieved %s candidates, showing top %s.", len(retrieved), args.top_k)

    sim_by_job_id = {job.job_id: float(sim) for job, sim in retrieved}
    for i, (job, final_score) in enumerate(ranked[: args.top_k], start=1):
        logger.info(
            "%2d. final=%.4f | sim=%.4f | views=%.0f | salary=%.0f | %s",
            i,
            final_score,
            sim_by_job_id.get(job.job_id, 0.0),
            job.views,
            job.normalized_salary,
            job.title,
        )


def cmd_rag(args: argparse.Namespace) -> None:
    jobs = _load_jobs(args.data, nrows=args.nrows)
    engine = TFIDFRetrievalEngine(jobs)
    retrieved = engine.retrieve(args.query, top_k=args.candidate_k)

    ranker = JobRanker(
        use_normalization=not args.no_normalize,
        use_salary_boost=not args.no_salary,
    )
    ranked = ranker.rank(retrieved)
    top_jobs = [job for job, _ in ranked[: args.rag_top_n]]

    logger.info("Top Jobs:")
    for job in top_jobs:
        logger.info("- %s", job.title)

    explainer = JobRAGExplainer()
    explanation = explainer.explain(args.query, top_jobs)
    logger.info("LLM Explanation:\n%s", explanation)


def cmd_eval(_: argparse.Namespace) -> None:
    # Uses Settings for corpus size + file paths
    run_eval_main()


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()

    parser = argparse.ArgumentParser(prog="job-intelligence-platform")
    sub = parser.add_subparsers(dest="command", required=True)

    p_pre = sub.add_parser("preprocess", help="Clean/validate raw postings into jobs_clean.csv")
    p_pre.add_argument("--input", default=str(settings.postings_csv), help="Path to postings.csv")
    p_pre.add_argument("--output", default=str(settings.jobs_clean_csv), help="Path to jobs_clean.csv")
    p_pre.set_defaults(func=cmd_preprocess)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data", default=str(settings.jobs_clean_csv), help="Path to jobs_clean.csv")
    common.add_argument("--query", default=settings.default_query, help="Search query")
    common.add_argument("--candidate-k", type=int, default=settings.candidate_k, help="Retriever top_k candidates")
    common.add_argument("--top-k", type=int, default=settings.final_k, help="Final results to print")
    common.add_argument("--nrows", type=int, default=None, help="Load only first N rows (debug/speed)")
    common.add_argument("--no-normalize", action="store_true", help="Disable min-max normalization in ranker")
    common.add_argument("--no-salary", action="store_true", help="Disable salary boost in ranker")

    p_ret = sub.add_parser("retrieve", help="Retrieve + rank jobs for a query", parents=[common])
    p_ret.set_defaults(func=cmd_retrieve)

    p_rag = sub.add_parser("rag", help="Retrieve + rank + generate explanation (RAG)", parents=[common])
    p_rag.add_argument("--rag-top-n", type=int, default=3, help="Number of jobs to include in RAG context")
    p_rag.set_defaults(func=cmd_rag)

    p_eval = sub.add_parser("eval", help="Run offline evaluation (Recall@K, MRR@K)")
    p_eval.set_defaults(func=cmd_eval)

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    settings = get_settings()
    configure_logging(settings.log_level)

    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()


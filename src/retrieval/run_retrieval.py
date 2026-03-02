import pandas as pd

from src.data.models import PreprocessedJob
from src.retrieval.tfidf_engine import TFIDFRetrievalEngine


def main():

    df = pd.read_csv("../data/jobs_clean.csv")

    jobs = [
        PreprocessedJob(**record)
        for record in df.to_dict(orient="records")
    ]

    engine = TFIDFRetrievalEngine(jobs)

    results = engine.retrieve("machine learning engineer", top_k=5)

    for job, score in results:
        print(job.title, "→", round(score, 3))


if __name__ == "__main__":
    main()
import numpy as np
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.data.models import PreprocessedJob


class TFIDFRetrievalEngine:

    def __init__(self, jobs: List[PreprocessedJob]):

        self.jobs = jobs

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=5000
        )

        self.job_corpus = [
            self._build_document(job) for job in jobs
        ]

        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.job_corpus
        )

    def _build_document(self, job: PreprocessedJob) -> str:
        return f"{job.title} {job.description}"

    def retrieve(self, query: str, top_k: int = 10):

        query_vector = self.vectorizer.transform([query])

        similarities = cosine_similarity(
            query_vector,
            self.tfidf_matrix
        )[0]

        ranked_indices = np.argsort(similarities)[::-1][:top_k]

        results = [
            (self.jobs[i], similarities[i])
            for i in ranked_indices
        ]

        return results
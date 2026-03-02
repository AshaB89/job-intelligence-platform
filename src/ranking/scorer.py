import numpy as np
from datetime import datetime


class JobRanker:

    def __init__(self, alpha=0.6, beta=0.2, gamma=0.1, delta=0.1):
        """
        alpha: weight for retrieval relevance
        beta: weight for recency
        gamma: weight for popularity (views)
        delta: weight for salary
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def _recency_score(self, listed_time):
        if listed_time is None:
            return 0.0

        days_old = (datetime.now() - listed_time).days
        return np.exp(-0.05 * days_old)

    def _popularity_score(self, views):
        return np.log1p(views)

    def _salary_score(self, salary, salary_available):
        if salary_available == 0:
            return 0.0
        return np.log1p(salary)

    def rank(self, retrieved_results):
        """
        retrieved_results:
        List of (PreprocessedJob, similarity_score)
        """

        scored = []

        for job, sim_score in retrieved_results:

            recency = self._recency_score(job.listed_time)
            popularity = self._popularity_score(job.views)
            salary = self._salary_score(
                job.normalized_salary,
                getattr(job, "salary_available", 0)
            )

            final_score = (
                self.alpha * sim_score
                + self.beta * recency
                + self.gamma * popularity
                + self.delta * salary
            )

            scored.append((job, final_score))

        scored.sort(key=lambda x: x[1], reverse=True)

        return scored
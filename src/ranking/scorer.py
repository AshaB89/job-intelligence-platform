import numpy as np
from datetime import datetime


class JobRanker:

    def __init__(
        self,
        use_normalization=True,
        use_salary_boost=True,
        alpha=0.6,
        beta=0.2,
        gamma=0.2,
        salary_boost=0.1
    ):
        self.use_normalization = use_normalization
        self.use_salary_boost = use_salary_boost

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.salary_boost = salary_boost

    def _minmax(self, values):
        if not values:
            return []

        v_min = min(values)
        v_max = max(values)
        denom = (v_max - v_min) + 1e-8
        return [(v - v_min) / denom for v in values]

    def _recency_score(self, listed_time):
        if listed_time is None:
            return 0.0

        # Handle tz-aware datetimes safely
        now = datetime.now(listed_time.tzinfo) if getattr(listed_time, "tzinfo", None) else datetime.now()
        days_old = (now - listed_time).days
        return np.exp(-0.05 * days_old)

    def _popularity_score(self, views):
        return np.log1p(views)

    def _salary_score(self, salary):
        return np.log1p(salary)

    def rank(self, retrieved_results):

        if not retrieved_results:
            return []

        jobs = []
        sim_scores = []
        recency_scores = []
        popularity_scores = []
        salary_available_flags = []

        for job, sim_score in retrieved_results:
            jobs.append(job)
            sim_scores.append(float(sim_score))
            recency_scores.append(self._recency_score(job.listed_time))
            popularity_scores.append(self._popularity_score(job.views))
            salary_available_flags.append(
                getattr(job, "salary_available", 0)
            )

        # Salary signal is optional (experimental toggle)
        salary_scores = None
        if self.use_salary_boost:
            salary_scores = [
                self._salary_score(job.normalized_salary) for job in jobs
            ]

        # Optional normalization (experimental toggle)
        if self.use_normalization:
            sim_used = self._minmax(sim_scores)
            recency_used = self._minmax(recency_scores)
            popularity_used = self._minmax(popularity_scores)
            salary_used = self._minmax(salary_scores) if salary_scores is not None else None
        else:
            sim_used = sim_scores
            recency_used = recency_scores
            popularity_used = popularity_scores
            salary_used = salary_scores

        ranked = []

        for i, job in enumerate(jobs):

            base_score = (
                self.alpha * sim_used[i]
                + self.beta * recency_used[i]
                + self.gamma * popularity_used[i]
            )

            # Salary logic can be fully disabled for experiments
            final_score = base_score
            if self.use_salary_boost and salary_used is not None and salary_available_flags[i] == 1:
                final_score = base_score * (1 + self.salary_boost * salary_used[i])

            ranked.append((job, final_score))

        ranked.sort(key=lambda x: x[1], reverse=True)

        return ranked
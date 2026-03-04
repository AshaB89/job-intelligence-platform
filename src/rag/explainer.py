import requests
import logging

from src.config import get_settings


class JobRAGExplainer:

    def __init__(self):
        self.settings = get_settings()
        self.url = self.settings.ollama_url
        self.model = self.settings.ollama_model
        self.timeout_s = self.settings.ollama_timeout_s
        self.logger = logging.getLogger(__name__)

    def build_context(self, jobs):
        context = ""

        for i, job in enumerate(jobs, start=1):
            context += f"""
Job {i}
Title: {job.title}
Description: {job.description}
"""

        return context

    def explain(self, query, jobs):

        context = self.build_context(jobs)

        prompt = f"""
User query: {query}

Retrieved jobs:
{context}

Instructions:
1. Identify which job best matches the query.
2. Explain why using skills or requirements mentioned.
3. Be concise.
"""

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }

        try:
            response = requests.post(self.url, json=payload, timeout=self.timeout_s)
        except requests.RequestException as e:
            self.logger.exception("RAG request failed: %s", e)
            return "RAG explanation unavailable (request failed)."

        if response.status_code >= 400:
            self.logger.error(
                "RAG server returned error status=%s body=%s",
                response.status_code,
                (response.text or "")[:500],
            )
            return f"RAG explanation unavailable (server returned {response.status_code})."

        try:
            data = response.json()
        except ValueError:
            self.logger.error(
                "RAG response was not JSON. status=%s body=%s",
                response.status_code,
                (response.text or "")[:500],
            )
            return "RAG explanation unavailable (invalid response format)."

        text = data.get("response")
        if not text:
            self.logger.error("RAG JSON missing 'response' field: keys=%s", list(data.keys()))
            return "RAG explanation unavailable (missing response text)."

        return text
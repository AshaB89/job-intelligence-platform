from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class PreprocessedJob(BaseModel):
    job_id: str
    title: str
    description: str

    location: Optional[str] = None

    views: float = 0
    salary_available: int = 0
    normalized_salary: float = 0

    listed_time: Optional[datetime] = None
import pandas as pd
import logging
from typing import List

from src.data.models import PreprocessedJob

logger = logging.getLogger(__name__)

SELECTED_COLUMNS = [
    "job_id",
    "title",
    "description",
    "location",
    "views",
    "listed_time",
    "normalized_salary"
]


# DATA CLEANING LOGIC
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    # Column Selection
    df = df[SELECTED_COLUMNS]

    # Date Parsing 

    df["listed_time"] = pd.to_datetime(
        df["listed_time"],
        errors="coerce"
    )

    # Drop rows missing core retrieval signals
    df = df.dropna(subset=["title", "description"])


    # Text Normalization
    df["job_id"] = df['job_id'].astype(str)
    df["title"] = df["title"].astype(str).str.strip()
    df["description"] = df["description"].astype(str).str.strip()

    # Missing Value Handling
    df["location"] = df["location"].where(df["location"].notna(), None)
    df["views"] = df["views"].fillna(0)

    df["salary_available"] = df["normalized_salary"].notna().astype(int)
    df["normalized_salary"] = df["normalized_salary"].fillna(0)
    return df


# Pydantic Validation Layer
def validate_jobs(df: pd.DataFrame) -> List[PreprocessedJob]:

    jobs = []

    for _, row in df.iterrows():
        try:
            job = PreprocessedJob(**row.to_dict())
            jobs.append(job)

        except Exception as e:
            logger.debug("Skipping invalid row: %s", e)

    return jobs


def run_preprocessing(input_csv: str, output_csv: str) -> None:
    logger.info("Starting preprocessing pipeline.")
    logger.info("Reading raw CSV: %s", input_csv)

    df_raw = pd.read_csv(input_csv)
    logger.info("Raw dataset shape: %s", df_raw.shape)

    df_clean = clean_dataframe(df_raw)

    logger.info("After cleaning shape: %s", df_clean.shape)

    jobs = validate_jobs(df_clean)

    logger.info("Valid jobs: %s", len(jobs))

    df_valid = pd.DataFrame([job.model_dump() for job in jobs])

    df_valid.to_csv(output_csv, index=False)
    logger.info("Cleaned dataset saved: %s", output_csv)
    logger.info("Final dataset shape: %s", df_valid.shape)


if __name__ == "__main__":
    from src.config import get_settings
    from src.logging_config import configure_logging

    settings = get_settings()
    configure_logging(settings.log_level)
    run_preprocessing(str(settings.postings_csv), str(settings.jobs_clean_csv))
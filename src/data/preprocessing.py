import pandas as pd
from typing import List

from src.data.models import PreprocessedJob

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
            print(f"Skipping invalid row: {e}")

    return jobs


def main():

    print("🚀 Starting preprocessing pipeline...")

    # Read raw CSV from dedicated CSV folder
    df_raw = pd.read_csv("src/data/csv/postings.csv")

    print("Raw dataset:", df_raw.shape)

    df_clean = clean_dataframe(df_raw)

    print("After cleaning:", df_clean.shape)

    jobs = validate_jobs(df_clean)

    print("Valid jobs:", len(jobs))

    df_valid = pd.DataFrame([job.model_dump() for job in jobs])

    df_valid.to_csv("src/data/csv/jobs_clean.csv", index=False)

    print("Cleaned dataset saved → csv/jobs_clean.csv")
    print("Final dataset shape:", df_valid.shape)


if __name__ == "__main__":
    main()
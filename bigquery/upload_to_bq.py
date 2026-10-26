"""Upload data/processed/daily_demand.csv to BigQuery.

Before running:
    gcloud auth application-default login
    set GCP_PROJECT=your-project-id          (Windows CMD)
    python bigquery/upload_to_bq.py
"""
import os
from pathlib import Path

import pandas as pd
from google.cloud import bigquery

PROJECT = os.environ["GCP_PROJECT"]
DATASET = os.environ.get("BQ_DATASET", "hotel")
TABLE = f"{PROJECT}.{DATASET}.daily_demand"
CSV = Path(__file__).resolve().parents[1] / "data" / "processed" / "daily_demand.csv"


def main():
    client = bigquery.Client(project=PROJECT)
    ds = bigquery.Dataset(f"{PROJECT}.{DATASET}")
    ds.location = os.environ.get("BQ_LOCATION", "asia-southeast1")
    client.create_dataset(ds, exists_ok=True)

    df = pd.read_csv(CSV, parse_dates=["date"])
    df["date"] = df["date"].dt.date
    job = client.load_table_from_dataframe(
        df, TABLE, job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE"))
    job.result()
    print(f"Loaded {len(df)} rows into {TABLE}")


if __name__ == "__main__":
    main()

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from database.db_connection import get_engine

RAW_CSV_PATH = PROJECT_ROOT/"data"/"Diabetes_type.csv"
TARGET_TABLE = "diabetes_raw"

RAW_COLUMNS = [
    "No. of Pregnancy",
    "Age",
    "BMI",
    "BP(Systolic)",
    "BP(Diastolic)",
    "DiabetesPedigreeFunction",
    "Insulin",
    "Skin Thickness(mm)",
    "Type-2 Diabetic",
    "Glucose",
]

def ingest_raw_data():
    engine = get_engine()
    df = pd.read_csv(RAW_CSV_PATH, usecols = RAW_COLUMNS, low_memory = False)

    df = df.dropna(subset = ["Type-2 Diabetic"])
    print(f"[Ingest] loaded {len(df)} rows from {RAW_CSV_PATH}")
    print(f"[Ingest] Columns: {list(df.columns)}")

    with engine.begin() as conn:
        conn.exec_driver_sql(f"DROP TABLE IF EXISTS {TARGET_TABLE} CASCADE")

    df.to_sql(
        TARGET_TABLE,
        con = engine,
        if_exists = 'replace',
        index = False,
        chunksize = 1000,
        method = 'multi',

    )
    print("[ingested] Raw data loaded into 'diabetes raw' table")

if __name__ == "__main__":
    ingest_raw_data()
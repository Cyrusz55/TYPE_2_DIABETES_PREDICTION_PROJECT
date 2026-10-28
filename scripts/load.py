from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sqlalchemy import DateTime, Float, Integer, String
from database.db_connection import get_engine
from database.models import create_tables
from scripts.clean import CLEAN_PATH
target_col = "Type-2 Diabetic"

# This is the table that is in database/models.py

TARGET_TABLE = "type_2_diabetes"

CHUNKSIZE = 50000


_INTEGER_COLS = [
    "no_of_pregnancy", "age", "bp_systolic", "diabetes_pedigree_function",
    "insulin", "type_2_diabetic",
]

_FLOAT_COLS = [  # for readability only - defaults handled below.
    "bmi", "bp_diastolic", "skin_thickness_mm", "glucose",
]

def _sql_dtypes():
    dtypes = {}
    for c in _INTEGER_COLS:
        dtypes[c] = Integer
    for c in _FLOAT_COLS:
        dtypes[c] = Float
    return dtypes


def load_clean_data():
    engine = get_engine()

    with engine.begin() as conn:
        conn.exec_driver_sql(f"DROP TABLE IF EXISTS {TARGET_TABLE} CASCADE")

    total = 0
    reader = pd.read_csv(CLEAN_PATH, chunksize = CHUNKSIZE)
    for i, chunk in enumerate(reader):
        chunk = chunk.dropna(subset = [target_col]) 



        chunk.to_sql(
            TARGET_TABLE,
            engine,
            if_exists = "append",
            index = False,
            method = 'multi',
            chunksize = 1000,
            dtype = _sql_dtypes(),
        )
        total += len(chunk)
        print(f"[load] chunk {i + 1}: {len(chunk)} rows (total {total})")

    print(f"[load] Done. {total} cleaned records in '{TARGET_TABLE}' table")

if __name__ == "__main__":
    df_preview = pd.read_csv(CLEAN_PATH, nrow = 5)
    print('columns:', df_preview.columns.to_list())
    load_clean_data()




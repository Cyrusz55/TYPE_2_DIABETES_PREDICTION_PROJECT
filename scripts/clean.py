from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT/"data"/"Diabetes_type.csv"
CLEAN_PATH = PROJECT_ROOT/ "data"/"clean"/"Diabetes_type_clean.csv"
target_col = "Type-2 Diabetic"

def clean_data(df: pd.DataFrame) ->pd.DataFrame:
    print(f"[clean] Starting shape: {df.shape}")

    numeric_cols = [
        "No. of Pregnancy",
            "Age",
            "BMI",
            "BP(Systolic)",
            "BP(Diastolic)",
            "DiabetesPedigreeFunction",
            "Insulin",
            "Skin Thickness(mm)",
            "Type-2 Diabetic",
            "Glucose"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors = 'coerce')

    print(f"[clean] Final shape: {df.shape}")
    return df

if __name__ == "__main__":
    df_raw = pd.read_csv(RAW_PATH)
    df = clean_data(df_raw)
    df.to_csv(CLEAN_PATH, index = False)
    print(f"[clean] Cleaned data saved to {CLEAN_PATH}")

    


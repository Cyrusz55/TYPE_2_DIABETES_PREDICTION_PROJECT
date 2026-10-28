import os
import joblib
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier


MODEL_PATH = "model/best_pipeline_XGBoost.joblib"
TARGET_COL = "Type-2 Diabetic"

# The dataset is fully numeric - there are no categorical columns, so this list
# stays empty. It is kept here because the pipeline still needs a 'cat' branch
# (an empty one), and because some columns may become categorical later.
NUM_COLS = [
    "No. of Pregnancy",
    "Age",
    "BMI",
    "BP(Systolic)",
    "BP(Diastolic)",
    "DiabetesPedigreeFunction",
    "Insulin",
    "Skin Thickness(mm)",
    "Glucose",
]
CAT_COLS = []

# Scale the minority class up to 70% of the majority class before fitting.
# This is why the pipeline is an imblearn Pipeline - SMOTETomek resamples
# INSIDE cross-validation folds, so no synthetic points leak into the test set.
SAMPLING_STRATEGY = 0.7
IMBALANCE_RATIO = 840 / 225


def get_X_y(df: pd.DataFrame):
    X = df[NUM_COLS + CAT_COLS]
    y = df[TARGET_COL]
    return X, y


def build_pipeline():
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUM_COLS),
            ("cat", categorical_pipeline, CAT_COLS),
        ]
    )

    model = ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("smote", SMOTETomek(random_state=42, sampling_strategy=SAMPLING_STRATEGY)),
            ("model", XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.03,
                scale_pos_weight=IMBALANCE_RATIO,
                subsample=0.8,
                colsample_bytree=0.7,
                min_child_weight=5,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.5,
                random_state=42,
                eval_metric="logloss",
                verbosity=0,
            )),
        ]
    )
    return model


def train_model(df: pd.DataFrame):
    X, y = get_X_y(df)
    print("Class distribution:")
    print(y.value_counts())
    print(y.value_counts(normalize=True).round(3))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    print(f"\nTraining set: {X_train.shape}")
    print(f"Test set:     {X_test.shape}")

    model = build_pipeline()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)

    print("\nTest Set Results:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred, target_names=["not diabetic", "diabetic"], zero_division=0
    ))

    os.makedirs("model", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    return model, X_test, y_test


def predict_diabetes(model, new_data: pd.DataFrame):
    return model.predict(new_data)


def load_model():
    from pathlib import Path

    # Try local first
    local_path = Path(MODEL_PATH)
    if local_path.exists():
        return joblib.load(local_path)

    # HuggingFace cache
    cache_path = Path("model/models--cyrusnx--diabetes_model/snapshots")
    if cache_path.exists():
        snapshots = list(cache_path.glob("*/best_pipeline_XGBoost.joblib"))
        if snapshots:
            return joblib.load(snapshots[0])

    raise FileNotFoundError("Model file not found")


def predict_single(input_data: dict):
    model = load_model()
    feature_cols = NUM_COLS + CAT_COLS
    row = {c: input_data.get(c, np.nan) for c in feature_cols}
    df = pd.DataFrame([row])

    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0]
    confidence = float(proba[int(pred)])
    return int(pred), confidence


if __name__ == "__main__":
    df = pd.read_csv("data/Diabetes_type.csv")
    train_model(df)

"""Create README figures from the project's real dataset and saved model."""

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Diabetes_type.csv"
MODEL_PATH = ROOT / "model" / "best_pipeline_XGBoost.joblib"
OUTPUT_DIR = ROOT / "docs" / "images"
TARGET = "Type-2 Diabetic"
FEATURES = [
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
COLORS = {"green": "#2f855a", "teal": "#0d9488", "purple": "#7c3aed"}


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUTPUT_DIR / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
        }
    )
    df = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)
    x_train, x_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.2, random_state=42, stratify=df[TARGET]
    )
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]

    labels = ["Not diabetic (0)", "Diabetic (1)"]
    counts = df[TARGET].value_counts().reindex([0, 1])
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, counts, color=[COLORS["teal"], COLORS["purple"]])
    ax.set_title("Class balance")
    ax.set_ylabel("Rows")
    ax.bar_label(bars, padding=3)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "class_balance.png")

    missing = df[FEATURES].isna().sum()
    zeros = (df[FEATURES] == 0).sum()
    fig, ax = plt.subplots(figsize=(10, 5))
    positions = np.arange(len(FEATURES))
    width = 0.38
    ax.bar(positions - width / 2, missing, width, label="Missing", color=COLORS["purple"])
    ax.bar(positions + width / 2, zeros, width, label="Zero", color=COLORS["teal"])
    ax.set_xticks(positions, FEATURES, rotation=35, ha="right")
    ax.set_title("Missing and zero values by feature")
    ax.set_ylabel("Rows")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "missing_values.png")

    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    for ax, feature in zip(axes.flat, FEATURES):
        for outcome, color, label in [(0, COLORS["teal"], "Not diabetic"), (1, COLORS["purple"], "Diabetic")]:
            values = df.loc[df[TARGET] == outcome, feature].dropna()
            ax.hist(values, bins=18, alpha=0.55, color=color, label=label, density=True)
        ax.set_title(feature)
        ax.set_ylabel("Density")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Feature distributions by outcome", y=1.01, fontsize=15)
    fig.tight_layout()
    save(fig, "feature_distributions.png")

    auc = roc_auc_score(y_test, y_proba)

    importances = model.named_steps["model"].feature_importances_
    importance = pd.Series(importances, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(importance.index, importance.values, color=COLORS["purple"])
    ax.set_title("Feature importance (XGBoost)")
    ax.set_xlabel("Importance")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "feature_importance.png")

    matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(matrix, display_labels=labels).plot(
        ax=ax, cmap="BuGn", colorbar=False, values_format="d"
    )
    ax.set_title("Confusion matrix (test set)")
    save(fig, "confusion_matrix.png")

    false_positive, true_positive, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(false_positive, true_positive, color=COLORS["teal"], linewidth=2, label=f"XGBoost (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#94a3b8", linewidth=1, label="Chance")
    ax.set_title("Receiver operating characteristic")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "roc_curve.png")


if __name__ == "__main__":
    main()

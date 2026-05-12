from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.features import DEFAULT_FEATURES, engineer_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "waze_dataset.csv"
MODEL_DIR = PROJECT_ROOT / "models"
RANDOM_STATE = 42


def find_optimal_threshold(y_true: pd.Series, probabilities) -> float:
    """Choose the threshold that maximizes Youden's J statistic."""
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, probabilities)
    youden_j = true_positive_rate - false_positive_rate
    finite_thresholds = np.isfinite(thresholds)
    return float(thresholds[finite_thresholds][youden_j[finite_thresholds].argmax()])


def train(data_path: Path = DATA_PATH, model_dir: Path = MODEL_DIR) -> dict[str, float]:
    """Train and save the Waze churn model artifacts."""
    df = pd.read_csv(data_path)
    labeled = df[df["label"].notna()].copy()
    labeled["target"] = labeled["label"].str.lower().eq("churned").astype(int)

    X = engineer_features(labeled, DEFAULT_FEATURES)
    y = labeled["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train_scaled, y_train)

    probabilities = model.predict_proba(X_test_scaled)[:, 1]
    threshold = find_optimal_threshold(y_test, probabilities)
    predicted = (probabilities >= threshold).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "threshold": threshold,
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "model.joblib")
    joblib.dump(scaler, model_dir / "scaler.joblib")

    with (model_dir / "features.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "features": DEFAULT_FEATURES,
                "threshold": threshold,
                "target_positive_class": "churned",
            },
            file,
            indent=2,
        )

    print("Training complete")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"Threshold: {threshold:.4f}")
    print(classification_report(y_test, predicted, target_names=["retained", "churned"]))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Waze churn model.")
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="Path to training CSV.")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR, help="Output artifact folder.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.data, args.model_dir)

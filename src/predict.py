from __future__ import annotations

import json
import argparse
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from src.features import engineer_features, validate_input_columns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"


@dataclass
class ChurnArtifacts:
    model: object
    scaler: object
    features: list[str]
    threshold: float


def load_artifacts(model_dir: Path = DEFAULT_MODEL_DIR) -> ChurnArtifacts:
    """Load trained model, scaler, and feature configuration."""
    config_path = model_dir / "features.json"

    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    return ChurnArtifacts(
        model=joblib.load(model_dir / "model.joblib"),
        scaler=joblib.load(model_dir / "scaler.joblib"),
        features=config["features"],
        threshold=float(config["threshold"]),
    )


def predict_churn(
    df: pd.DataFrame,
    artifacts: ChurnArtifacts | None = None,
) -> pd.DataFrame:
    """Return input rows with churn probabilities and labels appended."""
    artifacts = artifacts or load_artifacts()
    missing_columns = validate_input_columns(df)

    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required columns: {missing}")

    model_features = engineer_features(df, artifacts.features)
    scaled_features = pd.DataFrame(
        artifacts.scaler.transform(model_features),
        columns=model_features.columns,
        index=model_features.index,
    )
    probabilities = artifacts.model.predict_proba(scaled_features)[:, 1]

    predictions = df.copy()
    predictions["churn_probability"] = probabilities
    predictions["prediction"] = [
        "Churned" if probability >= artifacts.threshold else "Retained"
        for probability in probabilities
    ]
    return predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score churn risk for a CSV file.")
    parser.add_argument("input_csv", type=Path, help="CSV file to score.")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "churn_predictions.csv",
        help="Where to save predictions.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    input_df = pd.read_csv(args.input_csv)
    output_df = predict_churn(input_df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output, index=False)
    print(f"Saved predictions to {args.output}")

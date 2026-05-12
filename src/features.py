from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_INPUT_COLUMNS = [
    "sessions",
    "drives",
    "total_sessions",
    "driven_km_drives",
    "duration_minutes_drives",
    "activity_days",
    "driving_days",
    "n_days_after_onboarding",
    "device",
]

DEFAULT_FEATURES = [
    "activity_days",
    "driving_days",
    "activity_to_driving_ratio",
    "drives_per_day",
    "sessions_per_day",
    "percent_days_driving",
    "n_days_after_onboarding",
    "total_sessions",
    "duration_minutes_drives",
    "km_per_drive",
    "device_iPhone",
]


def validate_input_columns(df: pd.DataFrame) -> list[str]:
    """Return required columns missing from an inference dataframe."""
    return [column for column in REQUIRED_INPUT_COLUMNS if column not in df.columns]


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide two Series and replace invalid values with zero."""
    result = numerator.div(denominator)
    return result.replace([np.inf, -np.inf], np.nan).fillna(0)


def engineer_features(
    df: pd.DataFrame,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """Create the same model features used for training and inference."""
    selected_features = feature_names or DEFAULT_FEATURES
    features = df.copy()

    features["km_per_drive"] = safe_divide(
        features["driven_km_drives"], features["drives"]
    )
    features["drives_per_day"] = safe_divide(features["drives"], features["activity_days"])
    features["sessions_per_day"] = safe_divide(
        features["sessions"], features["activity_days"]
    )
    features["percent_days_driving"] = safe_divide(
        features["driving_days"], features["activity_days"]
    )
    features["activity_to_driving_ratio"] = safe_divide(
        features["activity_days"], features["driving_days"]
    )

    device_dummies = pd.get_dummies(features["device"], prefix="device")
    features = pd.concat([features, device_dummies], axis=1)

    for column in selected_features:
        if column not in features.columns:
            features[column] = 0

    return features[selected_features]

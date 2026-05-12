# Waze User Churn Prediction

An end-to-end machine learning project that predicts whether Waze users are likely to churn based on their app activity, driving behavior, tenure, and device type.

The project includes exploratory analysis, feature engineering, model training, saved inference artifacts, and a Streamlit app for single-user and batch churn prediction.

## Project Structure

```text
waze-churn/
|
|-- data/
|   |-- raw/
|   |   |-- waze_dataset.csv
|   |   |-- train_weather.csv
|   |   `-- test_weather.csv
|   `-- processed/
|       |-- waze_unlabeled_inference.csv
|       `-- waze_inference_predictions.csv
|
|-- notebooks/
|   |-- waze_churn_analysis.ipynb
|   `-- scratch_weather.ipynb
|
|-- src/
|   |-- __init__.py
|   |-- features.py
|   |-- train.py
|   `-- predict.py
|
|-- models/
|   |-- model.joblib
|   |-- scaler.joblib
|   `-- features.json
|
|-- app/
|   `-- app.py
|
|-- requirements.txt
|-- README.md
`-- .gitignore
```

## Problem Statement

Waze has users with known churn labels and a smaller group of users with missing labels. The goal is to build a classifier that identifies users at risk of churn so the business can prioritize retention actions.

The target variable is:

```text
label = churned or retained
```

## Model Pipeline

```text
Raw Waze data
    |
    v
Data cleaning and label filtering
    |
    v
Feature engineering
    |
    v
Train/test split with stratification
    |
    v
Standard scaling
    |
    v
Logistic Regression with class balancing
    |
    v
Threshold tuning using Youden's J statistic
    |
    v
Saved model artifacts
    |
    v
Streamlit single-user and batch predictions
```

## Feature Engineering

The reusable feature logic lives in `src/features.py`. The final model uses these features:

```text
activity_days
driving_days
activity_to_driving_ratio
drives_per_day
sessions_per_day
percent_days_driving
n_days_after_onboarding
total_sessions
duration_minutes_drives
km_per_drive
device_iPhone
```

Engineered features include:

| Feature | Formula |
|---|---|
| `km_per_drive` | `driven_km_drives / drives` |
| `drives_per_day` | `drives / activity_days` |
| `sessions_per_day` | `sessions / activity_days` |
| `percent_days_driving` | `driving_days / activity_days` |
| `activity_to_driving_ratio` | `activity_days / driving_days` |
| `device_iPhone` | One-hot encoded device flag |

Invalid divisions are safely converted to `0`, so rows with zero drives or zero activity days do not break inference.

## Training

The training pipeline is in `src/train.py`.

It performs:

1. Load labeled rows from `data/raw/waze_dataset.csv`
2. Convert `label` into a binary churn target
3. Engineer model features
4. Split data with stratification
5. Fit a `StandardScaler`
6. Train a class-balanced Logistic Regression model
7. Select an operating threshold using Youden's J statistic
8. Save artifacts to `models/`

Run training:

```bash
python -m src.train
```

This creates or updates:

```text
models/model.joblib
models/scaler.joblib
models/features.json
```

## Inference

Prediction logic lives in `src/predict.py`.

It loads the saved model artifacts, validates required columns, applies the same feature engineering and scaling used during training, then returns:

```text
churn_probability
prediction
```

Required input columns:

```text
sessions, drives, total_sessions, driven_km_drives,
duration_minutes_drives, activity_days, driving_days,
n_days_after_onboarding, device
```

Run batch inference from the command line:

```bash
python -m src.predict data/processed/waze_unlabeled_inference.csv --output data/processed/churn_predictions.csv
```

## Streamlit App

The app is located at `app/app.py`.

Run it with:

```bash
streamlit run app/app.py
```

The app supports:

- Manual single-user churn prediction
- Batch CSV upload
- Churn probability scoring
- Churn/retained label assignment
- Downloadable prediction CSV

## Setup

Create an environment and install dependencies:

```bash
pip install -r requirements.txt
```

Then run the app:

```bash
streamlit run app/app.py
```

## Current Model

The saved production artifacts are stored in `models/`. The current app uses:

- Model: `models/model.joblib`
- Scaler: `models/scaler.joblib`
- Feature config and threshold: `models/features.json`

The previous notebook analysis compared Logistic Regression, Random Forest, and XGBoost. Logistic Regression was selected because it gave strong ROC-AUC while staying simple, interpretable, and reliable for deployment.

## Notes

- The notebook is kept for analysis and experimentation.
- The `src/` package is the clean, reusable model pipeline.
- The Streamlit app does not duplicate model logic; it calls the shared prediction pipeline.
- If the model is retrained, the app automatically uses the updated artifacts in `models/`.

# Waze User Churn Prediction

A machine learning pipeline to predict user churn on the Waze navigation platform.
Built with Logistic Regression, Random Forest, and XGBoost, with semi-supervised
self-training on 700 unlabeled users to improve generalization.

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

Waze has ~700 users with no churn label and ~14,299 labeled users (18% churned, 82% retained).
The goal is to:
1. Build a churn classifier on labeled data
2. Run inference on the 700 unlabeled users
3. Use high-confidence predictions as pseudo-labels to retrain and improve the model (self-training)
4. Explain model decisions with SHAP

## Dataset

**Source:** `waze_dataset.csv`
**Size:** ~15,000 rows | 13 features + 1 label
**Target:** `label` — `churned` or `retained`
**Class balance:** ~18% churned / 82% retained (imbalanced)

### Raw Features

| Feature | Description |
|---|---|
| `sessions` | App sessions in observation period |
| `drives` | Number of drives |
| `total_sessions` | Lifetime sessions |
| `driven_km_drives` | Total km driven |
| `duration_minutes_drives` | Total drive duration (mins) |
| `activity_days` | Days app was active |
| `driving_days` | Days user drove |
| `total_navigations_fav1/2` | Navigations to saved favorites |
| `n_days_after_onboarding` | Tenure in days |
| `device` | iPhone or Android |

## Pipeline Overview

### 1. Missing Label Analysis
700 users had no `label`. Before dropping them:
- Compared km driven and device distribution vs labeled users
- Distribution was similar → exported as `waze_unlabeled_inference.csv` for later inference

### 2. EDA
- Churn distribution bar chart
- Median comparison: churned vs retained across key features
- Skewness analysis → log transform applied to 7 right-skewed features
- Outlier detection via boxplots
- Pearson correlation with churn label
- Full feature correlation heatmap
- Device vs churn rate (Chi-square test for significance)

### 3. Feature Engineering

10 engineered features created via row-wise operations (no cross-row aggregation → no leakage risk):

| Feature | Formula |
|---|---|
| `km_per_drive` | driven_km / drives |
| `km_per_hour` | driven_km / (duration_mins / 60) |
| `drives_per_day` | drives / activity_days |
| `sessions_per_day` | sessions / activity_days |
| `percent_days_driving` | driving_days / activity_days |
| `activity_to_driving_ratio` | activity_days / driving_days |
| `recency_ratio` | sessions / total_sessions *(engineered but excluded — leakage risk)* |
| `total_favs` | fav1 + fav2 navigations |
| `fav_per_drive` | total_favs / drives |
| `is_new_user` | 1 if tenure < 90 days |

### 4. Feature Selection (Mutual Information)
MI scores computed for all features. Threshold: 0.005.
Final feature set (11 features):

activity_days, driving_days, activity_to_driving_ratio,
drives_per_day, sessions_per_day, percent_days_driving,
n_days_after_onboarding, total_sessions, duration_minutes_drives,
km_per_drive, device_iPhone
CopyCopied!

### 5. Modeling

**Train/test split:** 80/20, stratified, `random_state=42`
**Imbalance handling:** `class_weight='balanced'` (LR, RF) | `scale_pos_weight` (XGBoost)
**Threshold:** Youden's J optimal threshold per model (not lazy 0.5)
**Evaluation:** ROC-AUC, PR-AUC, classification report, confusion matrix

#### Model Results (v1 — baseline)

| Model | ROC-AUC | PR-AUC | Churn Recall | Churn Precision |
|---|---|---|---|---|
| **Logistic Regression** | **0.7412** | **0.3802** | **0.83** | 0.28 |
| Random Forest | 0.7329 | 0.3592 | 0.71 | 0.30 |
| XGBoost | 0.7261 | 0.3469 | 0.65 | 0.32 |

**Winner: Logistic Regression** — confirmed by 5-fold CV (0.7526 ± 0.0151)

## Handling Class Imbalance
The dataset exhibited significant class imbalance, with retained customers representing ~82% of observations versus ~18% churned. 
To address this, two approaches were evaluated: SMOTE  and scale_pos_weight parameter

##Observations
scale_pos_weight (set to ~4.6, the ratio of retained to churned customers)encouraged the model to penalize missed churn predictions more heavily.
While this improved recall slightly, it did not meaningfully improve precision or overall F1 score, and offered less control over the training data distribution itself.

SMOTE synthetically oversampled the minority churn class in the training set to achieve a balanced distribution. 
As expected, this introduced a trade-off: precision decreased marginally (~1%), while recall improved, resulting in the model identifying a greater proportion of true churners.
More aggressive churn prediction leads to more false positives alongside more true positives.

### Decision
SMOTE was selected as the final imbalance handling strategy. Despite the minor precision trade-off, it produced more consistent recall across all three models. 
Given that the primary business objective is to **identify at-risk customers before they churn**, maximising recall was prioritised over precision.


#### 5-Fold Cross-Validation ROC-AUC

| Model | Mean | Std |
|---|---|---|
| Logistic Regression | 0.7537 | ±0.015 |
| Random Forest | 0.7409 | ±0.016 |
| XGBoost | 0.7339 | ±0.014 |

### 6. Inference on 700 Unlabeled Users
- Applied same feature engineering pipeline
- Scaled using the **training scaler only** (no refit)
- Threshold sensitivity analysis across 0.50–0.70
- Final threshold: **0.55**
- Output saved to `waze_inference_predictions.csv`

### 7. Self-Training (Semi-Supervised)

Used high-confidence predictions from the unlabeled 700 as pseudo-labels:
- **High confidence threshold:** ≥ 0.70 (churned) or ≤ 0.30 (retained)
- **Confident predictions selected:** 326 / 700 (221 retained, 105 churned)
- **Combined dataset:** 14,299 + 326 = **14,625 rows**
- Retrained LR (v2) on combined dataset

#### v1 vs v2 Comparison

| Metric | v1 | v2 | Change |
|---|---|---|---|
| ROC-AUC | 0.7412 | **0.7620** | **+0.0208** |
| F1 (Churn) | 0.4152 | **0.4455** | **+0.0303** |
| Recall (Churn) | 0.8383 | 0. 0.7717  | -0.0666 |

Self-training improved ROC-AUC by ~2.7% and F1 by ~3.3%.
Recall dropped slightly — the model became more precise, flagging fewer but more confident churners.

### 8. SHAP Explainability
Applied `shap.LinearExplainer` to the winning LR model:
- **Bar plot:** global feature importance
- **Beeswarm plot:** feature impact direction and distribution
- **Waterfall plot:** single-user explanation for highest-risk churner



## Key Findings

- **Activity and driving patterns dominate churn signal** — users who drive less frequently relative to their active days are more likely to churn
- **Logistic Regression outperforms tree models** on this dataset — the churn signal is largely linear in these engineered ratio features
- **Self-training on 326 pseudo-labeled users** provided a meaningful boost without any new labeled data
- **Device type (iPhone vs Android)** was not a statistically significant predictor (Chi-square p > 0.05)
- **Churn precision (~29–31%) across all models** reflects the difficulty of this problem — the 18/82 imbalance makes false positives inevitable at high recall



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


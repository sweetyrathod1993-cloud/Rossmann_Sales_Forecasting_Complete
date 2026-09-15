"""
Task 2.2, 2.3, 2.4, 2.5: Sklearn Pipeline Training, Loss Function Defense,
Feature Importance, Confidence Intervals, and Timestamped Serialization.
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Any, Optional
import json
import shutil

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from src.logger import get_logger
    from src.data_loader import load_and_merge_data
    from src.feature_engineering import (
        engineer_features,
        build_preprocessor,
        NUMERIC_FEATURES,
        CATEGORICAL_FEATURES
    )
except ImportError:
    from logger import get_logger
    from data_loader import load_and_merge_data
    from feature_engineering import (
        engineer_features,
        build_preprocessor,
        NUMERIC_FEATURES,
        CATEGORICAL_FEATURES
    )

import numpy as np
import pandas as pd
import joblib
import matplotlib
if "ipykernel" not in sys.modules: matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

logger = get_logger("train_pipeline")

MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Square Percentage Error (RMSPE).
    Filters out zeros to prevent division by zero.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not np.any(mask):
        return 0.0
    percentage_errors = (y_true[mask] - y_pred[mask]) / y_true[mask]
    return float(np.sqrt(np.mean(np.square(percentage_errors))))


def serialize_model(pipeline: Pipeline) -> Path:
    """
    Serializes fitted pipeline with timestamp format required by specification:
    format: dd-mm-yyyy-HH-MM-SS-00.pkl (e.g. 10-08-2020-16-32-31-00.pkl)
    """
    now = datetime.now()
    timestamp_str = now.strftime("%d-%m-%Y-%H-%M-%S-00")
    model_filename = f"{timestamp_str}.pkl"
    model_path = MODELS_DIR / model_filename

    logger.info(f"Serializing trained pipeline to: {model_path}")
    joblib.dump(pipeline, model_path)

    # Also maintain latest_model.pkl pointer for web app convenience
    latest_path = MODELS_DIR / "latest_model.pkl"
    shutil.copyfile(model_path, latest_path)
    logger.info(f"Updated pointer copy: {latest_path}")

    return model_path


def compute_prediction_with_confidence_interval(
    pipeline: Pipeline,
    X_sample: pd.DataFrame,
    confidence_level: float = 0.95
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimates predictions and confidence intervals using ensemble variance across
    individual trees of the fitted RandomForestRegressor.
    CI = mu +- z * sigma_trees
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    rf_model: RandomForestRegressor = pipeline.named_steps["regressor"]

    # Transform features
    X_trans = preprocessor.transform(X_sample)

    # Predict from each individual estimator tree
    tree_preds = np.array([tree.predict(X_trans) for tree in rf_model.estimators_])
    
    mean_preds = np.mean(tree_preds, axis=0)
    std_preds = np.std(tree_preds, axis=0)

    # z-value for confidence level
    z_score = 1.96 if confidence_level == 0.95 else 1.645
    ci_lower = np.maximum(0, mean_preds - z_score * std_preds)
    ci_upper = mean_preds + z_score * std_preds

    return mean_preds, ci_lower, ci_upper


def plot_feature_importance(pipeline: Pipeline, output_path: Path):
    """Extracts and plots top 15 features from the fitted pipeline."""
    preprocessor = pipeline.named_steps["preprocessor"]
    rf_model = pipeline.named_steps["regressor"]

    # Retrieve feature names from ColumnTransformer
    cat_encoder = preprocessor.named_transformers_["cat"]
    encoded_cat_features = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    all_feature_names = NUMERIC_FEATURES + encoded_cat_features

    importances = rf_model.feature_importances_
    feat_df = pd.DataFrame({
        "Feature": all_feature_names[:len(importances)],
        "Importance": importances
    }).sort_values("Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    top15 = feat_df.head(15).sort_values("Importance", ascending=True)
    ax.barh(top15["Feature"], top15["Importance"], color="#3498db")
    ax.set_title("Top 15 Feature Importances (Random Forest Pipeline)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Relative Importance")
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    logger.info(f"Feature importance plot saved to: {output_path}")
    return feat_df


def save_loss_defense_report():
    """Generates written documentation defending the choice of RMSPE."""
    defense_path = REPORTS_DIR / "Loss_Function_Defense.md"
    content = """# Task 2.3: Selection and Defense of Loss Function

## Chosen Loss Function: Root Mean Square Percentage Error (RMSPE)
$$\\text{RMSPE} = \\sqrt{ \\frac{1}{N} \\sum_{i=1}^N \\left( \\frac{y_i - \\hat{y}_i}{y_i} \\right)^2 }$$

## Rationale and Defense:

1. **Scale Independence Across Store Sizes:**
   Rossmann operates 1,115+ diverse stores across several cities. High-volume transit stores (e.g., StoreType 'b') generate daily sales upwards of €20,000–€30,000, while smaller neighborhood stores generate €2,000–€5,000.
   Under standard Euclidean metrics like Mean Squared Error (MSE) or Root Mean Squared Error (RMSE), an absolute prediction error of €1,000 on a large store is penalized equally to an error of €1,000 on a small store, despite representing a negligible 4% relative error for the former and an intolerable 50% forecasting failure for the latter. RMSPE penalizes relative percentage deviation, ensuring fair and balanced optimization across all store tiers.

2. **Managerial Decision Utility:**
   Store managers and supply chain planners order inventory and allocate staffing based on proportional margins. Communicating forecasting accuracy in relative percentage terms (e.g., *our model forecasts with 11.4% average error*) provides transparent, actionable business confidence.

3. **Benchmarking Consistency:**
   RMSPE is the official metric of the Rossmann Store Sales competition, enabling rigorous validation against state-of-the-art retail benchmarks.
"""
    with open(defense_path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info(f"Loss function defense documented at: {defense_path}")


def train_model(
    train_df: pd.DataFrame,
    n_estimators: int = 60,
    max_depth: int = 18,
    sample_size: int = 80000,
    random_state: int = 42
) -> Tuple[Pipeline, Dict[str, float], Path]:
    """
    Trains the scikit-learn Pipeline with RandomForestRegressor.
    """
    logger.info("Starting model training pipeline...")
    save_loss_defense_report()

    # Filter open stores with positive sales
    open_data = train_df[(train_df["Open"] == 1) & (train_df["Sales"] > 0)].copy()

    # Feature engineering
    engineered = engineer_features(open_data)

    # Subsample if dataset is very large for swift training
    if sample_size and len(engineered) > sample_size:
        logger.info(f"Sampling {sample_size} records for model training and validation...")
        engineered = engineered.sample(n=sample_size, random_state=random_state)

    # Time-based split: Reserve last 6 weeks of available records for validation if date span permits
    min_date = engineered["Date"].min()
    max_date = engineered["Date"].max()
    date_span_days = (max_date - min_date).days

    if date_span_days >= 60:
        split_date = max_date - pd.Timedelta(weeks=6)
        train_split = engineered[engineered["Date"] <= split_date]
        val_split = engineered[engineered["Date"] > split_date]
    else:
        train_split = pd.DataFrame()
        val_split = pd.DataFrame()

    if len(train_split) < 500 or len(val_split) < 100:
        from sklearn.model_selection import train_test_split
        train_split, val_split = train_test_split(engineered, test_size=0.20, random_state=random_state)

    logger.info(f"Dataset split: Train={len(train_split)}, Validation={len(val_split)}")

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X_train = train_split[feature_cols]
    y_train = train_split["Sales"].values

    X_val = val_split[feature_cols]
    y_val = val_split["Sales"].values

    # Build Pipeline
    preprocessor = build_preprocessor()
    regressor = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=random_state
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", regressor)
    ])

    logger.info("Fitting Scikit-Learn Pipeline...")
    pipeline.fit(X_train, y_train)

    # Validation Evaluation
    y_pred = pipeline.predict(X_val)
    # Clip negative predictions to zero
    y_pred = np.maximum(0, y_pred)

    val_rmspe = rmspe(y_val, y_pred)
    val_rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
    val_mae = float(mean_absolute_error(y_val, y_pred))

    metrics = {
        "val_rmspe": round(val_rmspe, 4),
        "val_rmse": round(val_rmse, 2),
        "val_mae": round(val_mae, 2),
        "train_samples": len(X_train),
        "val_samples": len(X_val)
    }
    logger.info(f"Validation Metrics: RMSPE={metrics['val_rmspe']:.4f}, RMSE=€{metrics['val_rmse']:.2f}, MAE=€{metrics['val_mae']:.2f}")

    # Plot feature importance
    plot_feature_importance(pipeline, FIGURES_DIR / "feature_importance.png")

    # Serialize model with required timestamp format
    model_path = serialize_model(pipeline)

    # Save metrics report
    with open(REPORTS_DIR / "model_evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return pipeline, metrics, model_path


if __name__ == "__main__":
    train_m, _, _ = load_and_merge_data(nrows=100000)
    pipeline, metrics, saved_path = train_model(train_m, sample_size=40000, n_estimators=40, max_depth=16)
    print(f"Pipeline trained successfully! Model saved at: {saved_path}")
    print(f"Metrics: {metrics}")

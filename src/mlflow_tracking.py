"""
Task 2.7: MLflow Tracking, Model Logging, and Serving Predictions.
Tracks experiments, logs hyperparameters, metrics, and models, and provides
inference helper for test data evaluation using MLflow.
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from src.logger import get_logger
    from src.data_loader import load_and_merge_data
    from src.train_pipeline import train_model, rmspe
    from src.feature_engineering import engineer_features, NUMERIC_FEATURES, CATEGORICAL_FEATURES
except ImportError:
    from logger import get_logger
    from data_loader import load_and_merge_data
    from train_pipeline import train_model, rmspe
    from feature_engineering import engineer_features, NUMERIC_FEATURES, CATEGORICAL_FEATURES

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

logger = get_logger("mlflow_tracking")

DB_PATH = BASE_DIR / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{DB_PATH.resolve()}")
EXPERIMENT_NAME = "Rossmann_Store_Sales_Forecasting"


def setup_mlflow_experiment(exp_name: str = EXPERIMENT_NAME) -> str:
    """Sets up or retrieves MLflow experiment."""
    mlflow.set_tracking_uri(f"sqlite:///{DB_PATH.resolve()}")
    exp = mlflow.get_experiment_by_name(exp_name)
    if exp is None:
        exp_id = mlflow.create_experiment(exp_name)
        logger.info(f"Created new MLflow experiment '{exp_name}' (ID: {exp_id})")
    else:
        exp_id = exp.experiment_id
        logger.info(f"Using existing MLflow experiment '{exp_name}' (ID: {exp_id})")
    mlflow.set_experiment(exp_name)
    return exp_id


def track_and_log_run(
    pipeline,
    metrics: Dict[str, float],
    params: Dict[str, Any],
    run_name: str = "Random_Forest_Pipeline_Run"
) -> str:
    """
    Logs parameters, metrics, artifacts, and serialized model to MLflow.
    """
    setup_mlflow_experiment()

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        logger.info(f"Logging run '{run_name}' to MLflow (Run ID: {run_id})...")

        # 1. Log Hyperparameters
        for k, v in params.items():
            mlflow.log_param(k, v)

        # 2. Log Metrics
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))

        # 3. Log Artifacts
        fig_path = BASE_DIR / "reports" / "figures" / "feature_importance.png"
        if fig_path.exists():
            mlflow.log_artifact(str(fig_path), artifact_path="figures")

        defense_path = BASE_DIR / "reports" / "Loss_Function_Defense.md"
        if defense_path.exists():
            mlflow.log_artifact(str(defense_path), artifact_path="reports")

        # 4. Log Scikit-Learn Pipeline
        logger.info("Logging Scikit-Learn Pipeline to MLflow...")
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="random_forest_pipeline"
        )

        model_uri = f"runs:/{run_id}/random_forest_pipeline"
        logger.info(f"Model logged successfully. Model URI: {model_uri}")
        return model_uri


def predict_with_mlflow(model_uri: str, test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Task 2.7 Inference:
    Loads logged model from MLflow URI and computes predictions on test data.
    """
    logger.info(f"Loading MLflow model for inference from: {model_uri}")
    mlflow.set_tracking_uri(f"sqlite:///{DB_PATH.resolve()}")
    loaded_model = mlflow.sklearn.load_model(model_uri)

    # Engineer test features
    feat_test = engineer_features(test_df)
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X_test = feat_test[feature_cols]

    # Predict
    logger.info(f"Executing batch prediction on {len(X_test)} records...")
    raw_preds = loaded_model.predict(X_test)
    preds = np.maximum(0, raw_preds)

    result_df = test_df.copy()
    result_df["Predicted_Sales"] = preds
    result_df["Predicted_Customers"] = (result_df["Predicted_Sales"] / 9.5).round().astype(int)

    logger.info("MLflow inference completed successfully.")
    return result_df


def run_mlflow_pipeline():
    """Executes model training with complete MLflow tracking and test inference."""
    logger.info("Starting MLflow pipeline experiment run...")
    train_m, test_m, _ = load_and_merge_data(nrows=40000)

    params = {
        "model_type": "RandomForestRegressor",
        "n_estimators": 35,
        "max_depth": 15,
        "min_samples_split": 5,
        "sample_size": 30000
    }

    pipeline, metrics, _ = train_model(
        train_m,
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        sample_size=params["sample_size"]
    )

    model_uri = track_and_log_run(pipeline, metrics, params)

    # Test inference demo
    sample_test = test_m.head(30)
    inference_results = predict_with_mlflow(model_uri, sample_test)
    logger.info("Sample MLflow inference results:")
    print(inference_results[["Store", "Date", "Promo", "Predicted_Sales", "Predicted_Customers"]].head(5))

    return model_uri


if __name__ == "__main__":
    run_mlflow_pipeline()

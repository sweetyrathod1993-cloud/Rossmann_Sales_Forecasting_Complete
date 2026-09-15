"""
Unified Predictor Engine for Rossmann Store Sales.
Loads latest serialized Pipeline model and provides batch/single inference,
confidence intervals, and dual predictions (Sales + Customers).
"""

import sys
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from src.logger import get_logger
    from src.feature_engineering import engineer_features, NUMERIC_FEATURES, CATEGORICAL_FEATURES
    from src.data_loader import clean_store_data, resolve_data_path
except ImportError:
    from logger import get_logger
    from feature_engineering import engineer_features, NUMERIC_FEATURES, CATEGORICAL_FEATURES
    from data_loader import clean_store_data, resolve_data_path

import numpy as np
import pandas as pd
import joblib

logger = get_logger("predictor")

MODELS_DIR = BASE_DIR / "models"


class SalesPredictor:
    def __init__(self, model_path: Optional[str] = None, store_path: Optional[str] = None):
        self.model, self.model_name = self._load_model(model_path)
        self.store_metadata = self._load_store_metadata(store_path)

    def _load_model(self, custom_path: Optional[str] = None):
        if custom_path and Path(custom_path).exists():
            target_file = Path(custom_path)
        else:
            latest_ptr = MODELS_DIR / "latest_model.pkl"
            if latest_ptr.exists():
                target_file = latest_ptr
            else:
                candidates = sorted(MODELS_DIR.glob("*.pkl"), key=lambda f: f.stat().st_mtime, reverse=True)
                if not candidates:
                    raise FileNotFoundError(f"No trained .pkl models found in {MODELS_DIR}")
                target_file = candidates[0]

        logger.info(f"Loaded forecasting pipeline from: {target_file.name}")
        pipeline = joblib.load(target_file)
        return pipeline, target_file.name

    def _load_store_metadata(self, store_path: Optional[str] = None) -> pd.DataFrame:
        p_store = resolve_data_path("store.csv", store_path)
        raw_store = pd.read_csv(p_store)
        return clean_store_data(raw_store)

    def prepare_input_dataframe(self, df: pd.DataFrame, default_store_id: int = 1) -> pd.DataFrame:
        """
        Harmonizes uploaded or generated user input columns to match pipeline requirements.
        Supports column synonyms:
        - Store_id -> Store
        - IsPromo -> Promo
        - IsHoliday -> StateHoliday
        """
        data = df.copy()

        # Rename aliases
        rename_map = {
            "Store_id": "Store",
            "store_id": "Store",
            "store": "Store",
            "IsPromo": "Promo",
            "isPromo": "Promo",
            "promo": "Promo",
            "IsHoliday": "StateHoliday",
            "isHoliday": "StateHoliday",
            "Holiday": "StateHoliday",
            "isWeekend": "IsWeekend"
        }
        data = data.rename(columns=rename_map)

        if "Store" not in data.columns:
            data["Store"] = default_store_id

        if "Date" not in data.columns:
            raise ValueError("Input dataframe must contain a 'Date' column.")

        data["Date"] = pd.to_datetime(data["Date"])

        # Merge store metadata if store attributes are not present
        if "StoreType" not in data.columns or "Assortment" not in data.columns:
            data = pd.merge(data, self.store_metadata, on="Store", how="left")

        # Fallbacks for missing flags
        if "Promo" not in data.columns:
            data["Promo"] = 0
        if "StateHoliday" not in data.columns:
            data["StateHoliday"] = "0"
        if "SchoolHoliday" not in data.columns:
            data["SchoolHoliday"] = 0
        if "Open" not in data.columns:
            # Assume open unless Sunday and no special opening
            data["Open"] = data["Date"].dt.dayofweek.apply(lambda d: 0 if d == 6 else 1)

        # Apply feature engineering
        feat_df = engineer_features(data)
        return feat_df

    def predict(self, df: pd.DataFrame, default_store_id: int = 1, compute_ci: bool = True) -> pd.DataFrame:
        """
        Generates predictions for Sales and Customers, along with confidence intervals.
        """
        prepared = self.prepare_input_dataframe(df, default_store_id=default_store_id)
        feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
        X = prepared[feature_cols]

        preprocessor = self.model.named_steps["preprocessor"]
        rf_regressor = self.model.named_steps["regressor"]

        X_trans = preprocessor.transform(X)

        if compute_ci and hasattr(rf_regressor, "estimators_") and len(rf_regressor.estimators_) > 0:
            tree_preds = np.array([tree.predict(X_trans) for tree in rf_regressor.estimators_])
            mean_preds = np.mean(tree_preds, axis=0)
            std_preds = np.std(tree_preds, axis=0)
            ci_lower = np.maximum(0, mean_preds - 1.96 * std_preds)
            ci_upper = mean_preds + 1.96 * std_preds
        else:
            mean_preds = self.model.predict(X)
            ci_lower = np.maximum(0, mean_preds * 0.85)
            ci_upper = mean_preds * 1.15

        # Enforce zero sales for closed store days
        if "Open" in prepared.columns:
            mean_preds = np.where(prepared["Open"] == 0, 0, mean_preds)
            ci_lower = np.where(prepared["Open"] == 0, 0, ci_lower)
            ci_upper = np.where(prepared["Open"] == 0, 0, ci_upper)

        mean_preds = np.maximum(0, mean_preds)

        # Customer count estimation (average historical basket size ~ €9.50)
        est_customers = np.where(mean_preds > 0, np.round(mean_preds / 9.5).astype(int), 0)

        result = prepared.copy()
        result["Predicted_Sales"] = np.round(mean_preds, 2)
        result["CI_Lower_95"] = np.round(ci_lower, 2)
        result["CI_Upper_95"] = np.round(ci_upper, 2)
        result["Predicted_Customers"] = est_customers
        return result


if __name__ == "__main__":
    predictor = SalesPredictor()
    sample_dates = pd.date_range("2015-08-01", periods=5, freq="D")
    sample_input = pd.DataFrame({
        "Date": sample_dates,
        "Store": [1] * 5,
        "IsPromo": [1, 1, 0, 0, 0],
        "IsHoliday": ["0", "0", "0", "0", "0"],
        "SchoolHoliday": [1, 1, 0, 0, 0]
    })
    preds = predictor.predict(sample_input)
    print("Predictor engine verification:")
    print(preds[["Date", "Store", "Promo", "Predicted_Sales", "CI_Lower_95", "CI_Upper_95", "Predicted_Customers"]])

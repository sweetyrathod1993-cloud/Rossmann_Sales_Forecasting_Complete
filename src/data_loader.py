"""
Data loading, validation, cleaning, and merging pipeline for Rossmann Store Sales.
Handles missing data, outlier detection, and schema harmonization.
"""

import sys
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from src.logger import get_logger
except ImportError:
    try:
        from .logger import get_logger
    except ImportError:
        from logger import get_logger

import numpy as np
import pandas as pd

logger = get_logger("data_loader")

DATA_DIR = BASE_DIR / "data"


def resolve_data_path(filename: str, custom_path: Optional[str] = None) -> Path:
    """Finds existing path for a given dataset file."""
    if custom_path and Path(custom_path).exists():
        return Path(custom_path)
    
    candidates = [
        DATA_DIR / filename,
        BASE_DIR / filename,
        BASE_DIR.parent / filename,
        BASE_DIR.parent / "Project 6" / filename,
        BASE_DIR.parent / "Project 6" / "rossmann-store-sales" / filename,
        Path(filename)
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()
            
    raise FileNotFoundError(f"Could not locate '{filename}' in expected locations: {candidates}")


def load_raw_data(
    train_path: Optional[str] = None,
    test_path: Optional[str] = None,
    store_path: Optional[str] = None,
    nrows: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads raw train, test, and store datasets."""
    p_train = resolve_data_path("train.csv", train_path)
    p_test = resolve_data_path("test.csv", test_path)
    p_store = resolve_data_path("store.csv", store_path)

    logger.info(f"Loading raw datasets: train={p_train.name}, test={p_test.name}, store={p_store.name}")
    train_df = pd.read_csv(p_train, nrows=nrows, low_memory=False)
    test_df = pd.read_csv(p_test, nrows=nrows, low_memory=False)
    store_df = pd.read_csv(p_store, low_memory=False)

    logger.info(f"Raw dimensions: train={train_df.shape}, test={test_df.shape}, store={store_df.shape}")
    return train_df, test_df, store_df


def clean_store_data(store_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans store metadata:
    - Imputes CompetitionDistance with median
    - Imputes CompetitionOpenSince with 0 (indicating no record)
    - Imputes Promo2 fields with 0 / 'None'
    """
    logger.info("Cleaning store metadata and handling missing values...")
    store = store_df.copy()

    # CompetitionDistance imputation
    median_dist = store["CompetitionDistance"].median()
    null_comp = store["CompetitionDistance"].isnull().sum()
    if null_comp > 0:
        logger.info(f"Imputing {null_comp} missing CompetitionDistance with median={median_dist:.1f}m")
        store["CompetitionDistance"] = store["CompetitionDistance"].fillna(median_dist)

    # Competitor opening timeline
    store["CompetitionOpenSinceMonth"] = store["CompetitionOpenSinceMonth"].fillna(0).astype(int)
    store["CompetitionOpenSinceYear"] = store["CompetitionOpenSinceYear"].fillna(0).astype(int)

    # Promo2 fields
    store["Promo2SinceWeek"] = store["Promo2SinceWeek"].fillna(0).astype(int)
    store["Promo2SinceYear"] = store["Promo2SinceYear"].fillna(0).astype(int)
    store["PromoInterval"] = store["PromoInterval"].fillna("None").astype(str)

    # Categoricals
    store["StoreType"] = store["StoreType"].astype(str)
    store["Assortment"] = store["Assortment"].astype(str)

    logger.info("Store data cleaning completed with 0 nulls remaining.")
    return store


def clean_sales_data(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Harmonizes types and cleans missing values in train or test sales datasets.
    """
    dataset_name = "Train" if is_train else "Test"
    logger.info(f"Cleaning {dataset_name} sales records...")
    df = df.copy()

    # Convert Date
    df["Date"] = pd.to_datetime(df["Date"])

    # Standardize StateHoliday: 0 and '0' -> '0', 'a', 'b', 'c' as strings
    df["StateHoliday"] = df["StateHoliday"].astype(str).replace({"0.0": "0", "0": "0"})

    # Handle missing 'Open' in test set (defaults to 1 unless Sunday)
    if "Open" in df.columns:
        null_open = df["Open"].isnull().sum()
        if null_open > 0:
            logger.info(f"Imputing {null_open} missing 'Open' flags in {dataset_name}...")
            if "DayOfWeek" in df.columns:
                df["Open"] = df["Open"].fillna(df["DayOfWeek"].apply(lambda d: 0 if d == 7 else 1))
            else:
                df["Open"] = df["Open"].fillna(1)
        df["Open"] = df["Open"].astype(int)

    # Ensure numeric columns
    numeric_cols = ["Store", "DayOfWeek", "Promo", "SchoolHoliday"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    if is_train:
        df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce").fillna(0)
        df["Customers"] = pd.to_numeric(df["Customers"], errors="coerce").fillna(0).astype(int)

    return df


def detect_outliers_iqr(df: pd.DataFrame, column: str = "Sales", factor: float = 3.0) -> Dict[str, Any]:
    """
    Detects extreme outliers in a column using the IQR method.
    """
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    pct = (len(outliers) / len(df)) * 100 if len(df) > 0 else 0

    logger.info(
        f"Outlier check for '{column}': Q1={q1:.1f}, Q3={q3:.1f}, Upper Threshold={upper_bound:.1f}. "
        f"Detected {len(outliers)} outliers ({pct:.2f}%)."
    )
    return {
        "column": column,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "outlier_count": len(outliers),
        "outlier_pct": pct
    }


def load_and_merge_data(
    train_path: Optional[str] = None,
    test_path: Optional[str] = None,
    store_path: Optional[str] = None,
    nrows: Optional[int] = None,
    filter_open_only: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Main loader function:
    1. Loads raw train, test, and store datasets.
    2. Cleans missing values across all sets.
    3. Merges store characteristics into train and test data.
    4. Optionally filters for active open store days (Open == 1 and Sales > 0).
    """
    raw_train, raw_test, raw_store = load_raw_data(train_path, test_path, store_path, nrows=nrows)

    clean_store = clean_store_data(raw_store)
    clean_train = clean_sales_data(raw_train, is_train=True)
    clean_test = clean_sales_data(raw_test, is_train=False)

    logger.info("Merging train data with store metadata...")
    train_merged = pd.merge(clean_train, clean_store, on="Store", how="left")

    logger.info("Merging test data with store metadata...")
    test_merged = pd.merge(clean_test, clean_store, on="Store", how="left")

    if filter_open_only and "Sales" in train_merged.columns:
        initial_len = len(train_merged)
        train_merged = train_merged[(train_merged["Open"] == 1) & (train_merged["Sales"] > 0)].copy()
        logger.info(f"Filtered open & positive sales records: {initial_len} -> {len(train_merged)}")

    logger.info(f"Final merged shapes: train={train_merged.shape}, test={test_merged.shape}")
    return train_merged, test_merged, clean_store


if __name__ == "__main__":
    train_m, test_m, store_c = load_and_merge_data(nrows=5000)
    print("Train merged sample successfully loaded:")
    print(train_m[["Store", "Date", "Sales", "Customers", "StoreType", "Assortment", "CompetitionDistance"]].head(3))
    outlier_info = detect_outliers_iqr(train_m, "Sales")
    print("Outlier info:", outlier_info)

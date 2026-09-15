"""
Task 2.1: Preprocessing & Feature Engineering for Rossmann Store Sales.
Extracts calendar features, holiday proximity (days to/from holiday),
competition age, promo2 active status, and builds sklearn ColumnTransformers.
"""

import sys
from pathlib import Path
from typing import List, Tuple, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from src.logger import get_logger
except ImportError:
    from logger import get_logger

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

logger = get_logger("feature_engineering")


def extract_holiday_dates(df: pd.DataFrame) -> np.ndarray:
    """Extracts sorted array of unique dates where a StateHoliday occurred."""
    holiday_mask = df["StateHoliday"].astype(str).isin(["a", "b", "c", "1"])
    holidays = pd.to_datetime(df.loc[holiday_mask, "Date"]).drop_duplicates().sort_values().values
    
    # Fallback if no holidays found in subset: use canonical German public holidays 2013-2015
    if len(holidays) == 0:
        holidays = pd.to_datetime([
            "2013-01-01", "2013-03-29", "2013-04-01", "2013-05-01", "2013-05-09",
            "2013-05-20", "2013-10-03", "2013-12-25", "2013-12-26",
            "2014-01-01", "2014-04-18", "2014-04-21", "2014-05-01", "2014-05-29",
            "2014-06-09", "2014-10-03", "2014-12-25", "2014-12-26",
            "2015-01-01", "2015-04-03", "2015-04-06", "2015-05-01", "2015-05-14",
            "2015-05-25", "2015-10-03", "2015-12-25", "2015-12-26"
        ]).values
    return holidays


def compute_holiday_distances(dates: pd.Series, holiday_dates: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes DaysToHoliday and DaysAfterHoliday using vectorized binary search.
    """
    dates_arr = pd.to_datetime(dates).values.astype("datetime64[D]")
    holidays_arr = holiday_dates.astype("datetime64[D]")

    # Find index of next holiday
    idx = np.searchsorted(holidays_arr, dates_arr)
    
    # Days to next holiday
    next_idx = np.clip(idx, 0, len(holidays_arr) - 1)
    days_to = (holidays_arr[next_idx] - dates_arr).astype("timedelta64[D]").astype(int)
    days_to = np.where(days_to < 0, 999, days_to)

    # Days after previous holiday
    prev_idx = np.clip(idx - 1, 0, len(holidays_arr) - 1)
    days_after = (dates_arr - holidays_arr[prev_idx]).astype("timedelta64[D]").astype(int)
    days_after = np.where(days_after < 0, 999, days_after)

    return days_to, days_after


def is_promo2_active_row(row: pd.Series) -> int:
    """Evaluates whether Promo2 is active on a given record."""
    if row.get("Promo2", 0) == 0:
        return 0
    try:
        p2_year = int(row.get("Promo2SinceYear", 0))
        p2_week = int(row.get("Promo2SinceWeek", 0))
        curr_year = int(row["Year"])
        curr_week = int(row["WeekOfYear"])
        
        if (curr_year > p2_year) or (curr_year == p2_year and curr_week >= p2_week):
            intervals = str(row.get("PromoInterval", "")).split(",")
            month_abbr = row["Date"].strftime("%b")
            return 1 if month_abbr in intervals else 0
    except Exception:
        pass
    return 0


def engineer_features(
    df: pd.DataFrame,
    holiday_dates: Optional[np.ndarray] = None,
    is_training: bool = True
) -> pd.DataFrame:
    """
    Comprehensive feature engineering pipeline:
    - Temporal features: Year, Month, Day, DayOfWeek, WeekOfYear, Quarter
    - Weekdays vs Weekends
    - Month phase: Beginning (1-10), Mid (11-20), End (21-31)
    - Days to holiday & Days after holiday
    - Competition age in months
    - Promo2 active status
    """
    logger.info("Engineering temporal, holiday distance, and promotional features...")
    data = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(data["Date"]):
        data["Date"] = pd.to_datetime(data["Date"])

    # 1. Calendar Extractions
    data["Year"] = data["Date"].dt.year
    data["Month"] = data["Date"].dt.month
    data["Day"] = data["Date"].dt.day
    data["DayOfWeek"] = data["Date"].dt.dayofweek + 1
    data["IsWeekend"] = (data["Date"].dt.dayofweek >= 5).astype(int)
    data["Quarter"] = data["Date"].dt.quarter
    data["WeekOfYear"] = data["Date"].dt.isocalendar().week.astype(int)

    # 2. Month Phase: 0 = Beginning (1-10), 1 = Mid (11-20), 2 = End (21-31)
    data["MonthPhase"] = pd.cut(
        data["Day"],
        bins=[0, 10, 20, 31],
        labels=[0, 1, 2],
        include_lowest=True
    ).astype(int)

    # 3. Holiday Distances
    if holiday_dates is None:
        holiday_dates = extract_holiday_dates(data)
    days_to, days_after = compute_holiday_distances(data["Date"], holiday_dates)
    data["DaysToHoliday"] = days_to
    data["DaysAfterHoliday"] = days_after

    # 4. Competition Age in Months
    def calc_comp_age(row):
        open_year = row.get("CompetitionOpenSinceYear", 0)
        open_month = row.get("CompetitionOpenSinceMonth", 0)
        if open_year > 0 and open_month > 0:
            age = 12 * (row["Year"] - open_year) + (row["Month"] - open_month)
            return max(0, age)
        return 0

    data["CompetitionOpenMonths"] = data.apply(calc_comp_age, axis=1)

    # 5. Promo2 Active Indicator
    data["IsPromo2Active"] = data.apply(is_promo2_active_row, axis=1)

    # 6. Ensure Categoricals as String for Encoder
    categorical_cols = ["StoreType", "Assortment", "StateHoliday"]
    for col in categorical_cols:
        if col in data.columns:
            data[col] = data[col].astype(str)

    logger.info(f"Feature engineering completed. Columns: {list(data.columns)}")
    return data


# Canonical feature column groups for modeling
NUMERIC_FEATURES = [
    "Store",
    "DayOfWeek",
    "Promo",
    "SchoolHoliday",
    "CompetitionDistance",
    "Year",
    "Month",
    "Day",
    "IsWeekend",
    "Quarter",
    "WeekOfYear",
    "MonthPhase",
    "DaysToHoliday",
    "DaysAfterHoliday",
    "CompetitionOpenMonths",
    "IsPromo2Active",
]

CATEGORICAL_FEATURES = [
    "StoreType",
    "Assortment",
    "StateHoliday",
]


def build_preprocessor() -> ColumnTransformer:
    """
    Creates modular scikit-learn ColumnTransformer:
    - Scales numeric features with StandardScaler
    - One-hot encodes categoricals
    """
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop"
    )
    return preprocessor


if __name__ == "__main__":
    from src.data_loader import load_and_merge_data
    train_m, _, _ = load_and_merge_data(nrows=1000)
    feat_df = engineer_features(train_m)
    print("Features engineered successfully. Sample:")
    print(feat_df[["Date", "DayOfWeek", "IsWeekend", "MonthPhase", "DaysToHoliday", "DaysAfterHoliday", "CompetitionOpenMonths", "IsPromo2Active"]].head(3))
    
    preprocessor = build_preprocessor()
    X_trans = preprocessor.fit_transform(feat_df)
    print(f"Preprocessed matrix shape: {X_trans.shape}")

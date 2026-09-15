import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

sys.modules.setdefault("numexpr", None)
sys.modules.setdefault("bottleneck", None)

from src.feature_engineering import (
    engineer_features,
    build_preprocessor,
    compute_holiday_distances,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES
)


def test_holiday_distances():
    dates = pd.Series(pd.to_datetime(["2015-12-23", "2015-12-25", "2015-12-27"]))
    holidays = pd.to_datetime(["2015-12-25"]).values

    days_to, days_after = compute_holiday_distances(dates, holidays)
    assert days_to[0] == 2
    assert days_to[1] == 0
    assert days_after[2] == 2


def test_engineer_features():
    df = pd.DataFrame({
        "Store": [1, 1],
        "Date": ["2015-05-05", "2015-05-24"],
        "Promo": [1, 0],
        "StateHoliday": ["0", "0"],
        "SchoolHoliday": [0, 0],
        "StoreType": ["a", "a"],
        "Assortment": ["a", "a"],
        "CompetitionDistance": [500.0, 500.0],
        "CompetitionOpenSinceMonth": [3, 3],
        "CompetitionOpenSinceYear": [2010, 2010],
        "Promo2": [1, 1],
        "Promo2SinceWeek": [10, 10],
        "Promo2SinceYear": [2014, 2014],
        "PromoInterval": ["Feb,May,Aug,Nov", "Feb,May,Aug,Nov"],
        "Open": [1, 1]
    })
    feat = engineer_features(df)
    assert "MonthPhase" in feat.columns
    assert "DaysToHoliday" in feat.columns
    assert "CompetitionOpenMonths" in feat.columns
    assert "IsPromo2Active" in feat.columns
    assert feat["MonthPhase"].iloc[0] == 0  # May 5 -> phase 0
    assert feat["MonthPhase"].iloc[1] == 2  # May 24 -> phase 2
    assert feat["IsPromo2Active"].iloc[0] == 1  # May in PromoInterval


def test_preprocessor_transformation():
    df = pd.DataFrame({
        "Store": [1, 2],
        "DayOfWeek": [1, 2],
        "Date": ["2015-05-05", "2015-05-06"],
        "Promo": [1, 0],
        "SchoolHoliday": [0, 0],
        "CompetitionDistance": [500.0, 1000.0],
        "StoreType": ["a", "b"],
        "Assortment": ["a", "c"],
        "StateHoliday": ["0", "a"],
        "CompetitionOpenSinceMonth": [0, 0],
        "CompetitionOpenSinceYear": [0, 0],
        "Promo2": [0, 0],
        "Promo2SinceWeek": [0, 0],
        "Promo2SinceYear": [0, 0],
        "PromoInterval": ["None", "None"]
    })
    feat = engineer_features(df)
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(feat)
    assert transformed.shape[0] == 2
    assert transformed.shape[1] >= len(NUMERIC_FEATURES)

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

from src.data_loader import load_and_merge_data, clean_store_data, clean_sales_data, detect_outliers_iqr


def test_clean_store_data():
    dummy_store = pd.DataFrame({
        "Store": [1, 2],
        "StoreType": ["a", "b"],
        "Assortment": ["a", "c"],
        "CompetitionDistance": [np.nan, 500.0],
        "CompetitionOpenSinceMonth": [np.nan, 9.0],
        "CompetitionOpenSinceYear": [np.nan, 2012.0],
        "Promo2": [0, 1],
        "Promo2SinceWeek": [np.nan, 14.0],
        "Promo2SinceYear": [np.nan, 2011.0],
        "PromoInterval": [np.nan, "Jan,Apr,Jul,Oct"]
    })
    cleaned = clean_store_data(dummy_store)
    assert cleaned["CompetitionDistance"].isnull().sum() == 0
    assert cleaned.loc[0, "CompetitionDistance"] == 500.0
    assert cleaned["PromoInterval"].iloc[0] == "None"
    assert cleaned["Promo2SinceWeek"].iloc[0] == 0


def test_clean_sales_data():
    dummy_sales = pd.DataFrame({
        "Store": [1, 2],
        "DayOfWeek": [1, 7],
        "Date": ["2015-07-31", "2015-08-01"],
        "Sales": ["5000", "0"],
        "Customers": ["500", "0"],
        "Open": [1, np.nan],
        "Promo": [1, 0],
        "StateHoliday": [0, "a"],
        "SchoolHoliday": [1, 0]
    })
    cleaned = clean_sales_data(dummy_sales, is_train=True)
    assert pd.api.types.is_datetime64_any_dtype(cleaned["Date"])
    assert cleaned["StateHoliday"].tolist() == ["0", "a"]
    assert cleaned["Sales"].dtype in [np.float64, np.int64]


def test_detect_outliers_iqr():
    df = pd.DataFrame({"Sales": [100, 102, 105, 101, 103, 104, 5000]})
    info = detect_outliers_iqr(df, "Sales", factor=2.0)
    assert info["outlier_count"] >= 1
    assert info["column"] == "Sales"

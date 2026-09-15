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

from src.train_pipeline import rmspe
from src.predictor import SalesPredictor


def test_rmspe_metric():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 300.0])
    # errors: +0.1, -0.05, 0.0 -> squares: 0.01, 0.0025, 0.0 -> mean: 0.0041667 -> sqrt: 0.0645497
    val = rmspe(y_true, y_pred)
    assert pytest.approx(val, rel=1e-2) == 0.0645


def test_predictor_inference():
    predictor = SalesPredictor()
    dates = pd.date_range("2015-08-01", periods=3, freq="D")
    sample_df = pd.DataFrame({
        "Date": dates,
        "Store": [1, 1, 1],
        "Promo": [1, 0, 0],
        "StateHoliday": ["0", "0", "0"],
        "SchoolHoliday": [0, 0, 0]
    })
    results = predictor.predict(sample_df)
    assert "Predicted_Sales" in results.columns
    assert "Predicted_Customers" in results.columns
    assert "CI_Lower_95" in results.columns
    assert "CI_Upper_95" in results.columns
    assert len(results) == 3
    assert (results["Predicted_Sales"] >= 0).all()

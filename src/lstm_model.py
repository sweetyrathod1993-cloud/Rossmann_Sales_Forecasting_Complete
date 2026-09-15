"""
Task 2.6: Time Series Analysis & Deep Learning LSTM Regression.
Steps:
1. Isolate Rossmann dataset into time series data.
2. Stationarity check using Augmented Dickey-Fuller (ADF) test.
3. Differencing if non-stationary.
4. Autocorrelation (ACF) and Partial Autocorrelation (PACF) analysis.
5. Sliding Window transformation into supervised learning format.
6. Scaling in the (-1, 1) range with MinMaxScaler.
7. Two-layer LSTM Regression architecture in TensorFlow/Keras.
"""

import sys
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import json

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from src.logger import get_logger
    from src.data_loader import load_and_merge_data
    from src.train_pipeline import rmspe
except ImportError:
    from logger import get_logger
    from data_loader import load_and_merge_data
    from train_pipeline import rmspe

import numpy as np
import pandas as pd
import joblib
import matplotlib
if "ipykernel" not in sys.modules: matplotlib.use("Agg")
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

logger = get_logger("lstm_model")

MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def isolate_time_series(df: pd.DataFrame, store_id: Optional[int] = 1) -> pd.Series:
    """
    Step 1: Isolates Rossmann Store Sales into a chronological daily time series.
    If store_id is provided, extracts that store; otherwise aggregates network daily total sales.
    """
    logger.info(f"Isolating time series data (Store ID: {store_id if store_id else 'All Network'})...")
    data = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(data["Date"]):
        data["Date"] = pd.to_datetime(data["Date"])

    if store_id is not None:
        data = data[data["Store"] == store_id]

    # Group by Date and sum sales
    daily_ts = data.groupby("Date")["Sales"].sum().sort_index()
    # Ensure regular daily frequency, forward filling or filling 0
    daily_ts = daily_ts.asfreq("D", fill_value=0)
    logger.info(f"Time series isolated. Total days: {len(daily_ts)}, Date Range: {daily_ts.index.min().date()} to {daily_ts.index.max().date()}")
    return daily_ts


def check_stationarity(ts: pd.Series) -> Dict[str, Any]:
    """
    Step 2: Augmented Dickey-Fuller (ADF) Stationarity test.
    """
    logger.info("Executing Augmented Dickey-Fuller (ADF) test...")
    # Drop any NaNs
    series = ts.dropna()
    adf_result = adfuller(series, autolag="AIC", result_object=False)
    
    test_stat, p_value, lags, nobs, crit_values, icbest = adf_result
    is_stationary = p_value < 0.05

    result = {
        "adf_statistic": float(test_stat),
        "p_value": float(p_value),
        "lags_used": int(lags),
        "n_observations": int(nobs),
        "critical_values": {k: float(v) for k, v in crit_values.items()},
        "is_stationary": bool(is_stationary),
        "conclusion": "Stationary (Reject H0)" if is_stationary else "Non-Stationary (Fail to Reject H0)"
    }
    logger.info(f"ADF Result: Stat={test_stat:.4f}, p-value={p_value:.4e} -> {result['conclusion']}")
    return result


def difference_series(ts: pd.Series, order: int = 1) -> pd.Series:
    """
    Step 3: Differencing to induce stationarity.
    """
    logger.info(f"Applying order-{order} differencing...")
    diff_ts = ts.diff(order).dropna()
    return diff_ts


def plot_acf_pacf(ts: pd.Series, output_path: Path, lags: int = 30):
    """
    Step 4: Autocorrelation (ACF) and Partial Autocorrelation (PACF) plots.
    """
    logger.info(f"Generating ACF and PACF plots (lags={lags})...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    
    plot_acf(ts.dropna(), lags=lags, ax=axes[0], color="#2980b9")
    axes[0].set_title("Autocorrelation Function (ACF)", fontsize=12, fontweight="bold")

    plot_pacf(ts.dropna(), lags=lags, ax=axes[1], color="#e74c3c", method="ywm")
    axes[1].set_title("Partial Autocorrelation Function (PACF)", fontsize=12, fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    logger.info(f"ACF/PACF plots saved to {output_path}")


def create_sliding_window_data(data_arr: np.ndarray, window_size: int = 14) -> Tuple[np.ndarray, np.ndarray]:
    """
    Step 5: Sliding Window for Time Series Data to convert into supervised learning matrix.
    X: [t-W, ..., t-1]
    y: [t]
    """
    X, y = [], []
    for i in range(window_size, len(data_arr)):
        X.append(data_arr[i - window_size:i, 0])
        y.append(data_arr[i, 0])
    return np.array(X), np.array(y)


def build_two_layer_lstm(window_size: int = 14) -> Sequential:
    """
    Step 7: Builds a 2-Layer LSTM Regression neural network.
    Complies with project spec: "The model should not be very deep (Two layers)
    due to computational requirements, comfortably run in google colab."
    """
    model = Sequential([
        tf.keras.layers.Input(shape=(window_size, 1)),
        LSTM(units=50, return_sequences=True),
        Dropout(0.2),
        LSTM(units=30, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.002), loss="mse")
    return model


def train_lstm_pipeline(
    df: pd.DataFrame,
    store_id: int = 1,
    window_size: int = 14,
    epochs: int = 20,
    batch_size: int = 16
) -> Dict[str, Any]:
    """
    Full workflow executing steps 1 through 7.
    """
    logger.info("--- Initiating Deep Learning LSTM Workflow ---")

    # Step 1: Isolate Time Series
    daily_ts = isolate_time_series(df, store_id=store_id)

    # Step 2: Stationarity Test on Raw Series
    raw_adf = check_stationarity(daily_ts)

    # Step 3: Differencing
    diff_ts = difference_series(daily_ts, order=1)
    diff_adf = check_stationarity(diff_ts)

    # Step 4: ACF & PACF Plotting
    plot_acf_pacf(diff_ts, FIGURES_DIR / "acf_pacf_analysis.png", lags=28)

    # Step 5 & 6: Scaling in (-1, 1) range
    logger.info("Scaling data to range (-1, 1) using MinMaxScaler...")
    values = daily_ts.values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_values = scaler.fit_transform(values)

    # Create Supervised Learning Matrix
    X, y = create_sliding_window_data(scaled_values, window_size=window_size)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Chronological Train-Test Split (Last 42 days / 6 weeks for testing)
    test_len = 42
    if len(X) > test_len + 50:
        X_train, X_test = X[:-test_len], X[-test_len:]
        y_train, y_test = y[:-test_len], y[-test_len:]
        dates_test = daily_ts.index[-test_len:]
    else:
        split = int(len(X) * 0.85)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        dates_test = daily_ts.index[-len(y_test):]

    logger.info(f"Sliding window shapes: X_train={X_train.shape}, X_test={X_test.shape}")

    # Step 7: Build & Train 2-Layer LSTM
    logger.info("Building 2-Layer LSTM network in TensorFlow/Keras...")
    lstm_model = build_two_layer_lstm(window_size=window_size)
    
    early_stop = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
    history = lstm_model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )

    # Inference & Inversion
    scaled_preds = lstm_model.predict(X_test)
    preds = scaler.inverse_transform(scaled_preds).flatten()
    actuals = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # Clip negative values
    preds = np.maximum(0, preds)

    # Metrics
    lstm_rmspe = rmspe(actuals, preds)
    lstm_rmse = float(np.sqrt(mean_squared_error(actuals, preds)))
    lstm_mae = float(mean_absolute_error(actuals, preds))

    logger.info(f"LSTM Test Evaluation: RMSPE={lstm_rmspe:.4f}, RMSE=€{lstm_rmse:.2f}, MAE=€{lstm_mae:.2f}")

    # Plot actual vs predicted sales
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(dates_test, actuals, label="Actual Sales (€)", color="#2c3e50", linewidth=2)
    ax.plot(dates_test, preds, label="LSTM 2-Layer Predicted Sales (€)", color="#e74c3c", linestyle="--", linewidth=2)
    ax.set_title(f"2-Layer LSTM Sales Forecast (Store #{store_id}) - RMSPE: {lstm_rmspe:.3f}", fontsize=13, fontweight="bold")
    ax.set_ylabel("Sales (€)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "lstm_forecast.png", dpi=200)
    plt.close(fig)

    # Save artifacts
    model_save_path = MODELS_DIR / "lstm_2layer_model.keras"
    scaler_save_path = MODELS_DIR / "lstm_scaler.pkl"
    lstm_model.save(model_save_path)
    joblib.dump(scaler, scaler_save_path)
    logger.info(f"LSTM Model saved to {model_save_path}, Scaler saved to {scaler_save_path}")

    report = {
        "store_id": store_id,
        "window_size": window_size,
        "raw_stationarity": raw_adf,
        "differenced_stationarity": diff_adf,
        "lstm_metrics": {
            "rmspe": round(lstm_rmspe, 4),
            "rmse": round(lstm_rmse, 2),
            "mae": round(lstm_mae, 2)
        }
    }
    with open(REPORTS_DIR / "lstm_analysis_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    train_m, _, _ = load_and_merge_data(nrows=150000)
    report_res = train_lstm_pipeline(train_m, store_id=1, epochs=8, window_size=14)
    print("LSTM workflow completed successfully!")
    print(json.dumps(report_res["lstm_metrics"], indent=2))

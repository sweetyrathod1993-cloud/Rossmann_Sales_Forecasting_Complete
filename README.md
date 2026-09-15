# 📊 Rossmann Store Sales Forecasting System

> **An End-to-End Enterprise Demand Forecasting System featuring Scikit-Learn Pipelines, 2-Layer Deep Learning LSTM, MLflow Tracking, and an Interactive Streamlit Web Dashboard.**

---

## 📌 Project Overview

**Rossmann Pharmaceuticals** operates over 1,115 drug stores across multiple cities in Germany. Historically, store managers relied on individual experience to project daily sales and footfall, resulting in inventory imbalances and stock-outs.

This repository provides an automated, scalable Machine Learning and Deep Learning forecasting system that predicts daily store sales turnover (€) and expected customer traffic **up to 6 weeks in advance**.

---

## 🚀 Quickstart: Which File to Run & What is Its Purpose?

Here is the exact reference guide for running the project components:

| Goal / Use Case | File to Run | Command | Purpose & Output |
| :--- | :--- | :--- | :--- |
| **Interactive Web App (UI)** | `app/app.py` | `streamlit run app/app.py` | Launches the Streamlit dashboard for single-day forecasting, 6-week batch CSV predictions, confidence intervals, and EDA insights. |
| **Master Orchestrator (All)** | `run_project.py` | `python run_project.py --all` | Sequentially executes the full pipeline: EDA $\to$ Model Training $\to$ LSTM Training $\to$ MLflow Tracking $\to$ Unit Tests. |
| **Exploratory Data Analysis** | `run_project.py` | `python run_project.py --eda` | Investigates all 11 customer behavior questions and saves high-resolution charts to `reports/figures/`. |
| **Train ML Pipeline** | `run_project.py` | `python run_project.py --train` | Trains Random Forest with Scikit-Learn Pipeline, computes metrics (RMSPE, RMSE, MAE), and serializes timestamped `.pkl` models to `models/`. |
| **Train Deep Learning LSTM** | `run_project.py` | `python run_project.py --lstm` | Evaluates series stationarity (ADF test), applies differencing, plots ACF/PACF, and trains a 2-layer LSTM recurrent neural network. |
| **MLflow Experiment Tracking** | `run_project.py` | `python run_project.py --mlflow` | Logs hyperparameters, metrics, and models to MLflow; serves batch test set inference. |
| **Run Automated Tests** | `tests/` | `pytest tests/` *(or `python run_project.py --test`)* | Executes 8 unit and integration tests verifying data loader, feature engineering, and pipeline inference. |
| **Interactive Notebook** | `Rossmann_Sales_Forecasting_Complete.ipynb` | Open in VS Code or `jupyter notebook` | Comprehensive step-by-step notebook containing all code, visual charts, deep learning training, and Markdown documentation. |
| **Production Predictor Engine** | `src/predictor.py` | `python src/predictor.py` | Standalone verification test of the unified inference engine that loads `models/latest_model.pkl` and outputs sales, customers, and 95% CI. |
| **Executive Slides** | `Rossmann_Sales_Forecasting_Presentation.pptx` | Open in PowerPoint / Google Slides | Complete 15-slide executive presentation covering EDA findings, ML/DL models, MLOps, and business ROI. |

---

## 📂 Repository Structure

```
Rossmann_Sales_Forecasting_Complete/
├── app/
│   └── app.py                                  # Task 3: Streamlit Web Dashboard
├── data/                                       # Raw datasets
│   ├── sample_submission.csv
│   ├── store.csv
│   ├── test.csv
│   └── train.csv
├── logs/                                       # Execution logs
│   └── sales_project.log
├── models/                                     # Serialized models with timestamps (Task 2.5)
│   ├── 15-09-2026-18-27-13-00.pkl
│   └── latest_model.pkl                        # Active deployment model pointer
├── reports/                                    # Analytical reports & visualizations
│   ├── Loss_Function_Defense.md                # Task 2.3: Mathematical defense of RMSPE
│   ├── model_evaluation_report.json            # JSON validation metrics
│   └── figures/                                # 14 Analytical plots (EDA, LSTM, Feature Importance)
│       ├── acf_pacf_analysis.png
│       ├── feature_importance.png
│       ├── lstm_forecast.png
│       ├── q1_promo_distribution.png ... q11_competitor_opening_impact.png
├── src/                                        # Modular production source code
│   ├── __init__.py
│   ├── data_loader.py                          # Data ingestion, imputation, outlier cleaning
│   ├── eda.py                                  # Task 1: 11 business behavior questions
│   ├── feature_engineering.py                  # Task 2.1: Temporal, holiday distance, scaling
│   ├── logger.py                               # Centralized logging setup
│   ├── lstm_model.py                           # Task 2.6: 2-layer LSTM, ADF test, ACF/PACF
│   ├── mlflow_tracking.py                      # Task 2.7: MLflow logging & model serving
│   ├── predictor.py                            # Unified prediction engine (Sales + Customers + CI)
│   └── train_pipeline.py                       # Task 2.2 - 2.5: Sklearn Pipeline, RMSPE, serialization
├── tests/                                      # Automated test suite (8 passing tests)
│   ├── test_data_loader.py
│   ├── test_features.py
│   └── test_pipeline.py
├── .streamlit/
│   └── config.toml                             # Streamlit UI & theme configurations
├── .gitignore                                  # Git ignore rules for data science
├── requirements.txt                            # Python package dependencies
├── run_project.py                              # Master CLI execution runner
├── Rossmann_Sales_Forecasting_Complete.ipynb   # Complete interactive Jupyter notebook
├── Rossmann_Sales_Forecasting_Presentation.pptx# 15-slide executive presentation
└── README.md                                   # Project documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd Rossmann_Sales_Forecasting_Complete
```

### 2. Create and Activate a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate    # On Windows use: venv\Scripts\activate
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 🛠️ Step-by-Step Task Breakdown

### Task 1: Exploratory Data Analysis (EDA)
Investigates customer purchasing behavior across 11 key operational questions:
* **Promotions (Q1, Q5, Q6):** Promo uplift boosts daily sales by **+40.3%** and footfall by **+31.2%**, increasing average basket size from €7.82 to €8.54 (+9.2%). Promo distribution between train (38.15%) and test (39.58%) is uniform.
* **Holidays (Q2):** Sales surge by **+34.8%** in the 3 days preceding State Holidays due to medication stockpiling, followed by a **+12%** demand rebound after reopening.
* **Seasonality (Q3):** December exhibits peak annual sales driven by Christmas, winter flu season, and healthcare spending deadlines.
* **Customer Correlation (Q4):** Very strong positive linear relationship ($r = 0.89$, $\rho = 0.91$).
* **Operating Trends (Q7, Q8):** Mondays generate the highest turnover. Exactly 33 stores operate on Sundays, capturing exceptional weekend turnover with zero weekday cannibalization.
* **Assortment & Competition (Q9, Q10, Q11):** Assortment 'Extra' yields the highest daily sales. Stores with competitors within <500m exhibit higher sales due to urban pedestrian density. New competitor openings cause an immediate ~6.5% sales drop over 90 days.

*Run command:*
```bash
python run_project.py --eda
```

---

### Task 2: Machine Learning & Deep Learning Pipelines

1. **Feature Engineering (`src/feature_engineering.py`):**
   - Temporal features (`Year`, `Month`, `Day`, `DayOfWeek`, `IsWeekend`, `Quarter`, `WeekOfYear`).
   - Month phase tiers (Beginning 1–10, Mid 11–20, End 21–31).
   - Vectorized holiday distance calculation (`DaysToHoliday`, `DaysAfterHoliday`) using `np.searchsorted`.
   - Competitor maturity (`CompetitionOpenMonths`) and dynamic `IsPromo2Active` flag.

2. **Scikit-Learn Pipeline (`src/train_pipeline.py`):**
   - Chains `StandardScaler` for numeric features and `OneHotEncoder` for categorical variables inside a `ColumnTransformer` with `RandomForestRegressor`.

3. **Loss Function Defense (RMSPE):**
   - Root Mean Square Percentage Error penalizes proportional error relative to actual sales:
     $$\text{RMSPE} = \sqrt{\frac{1}{N} \sum_{i=1}^N \left( \frac{y_i - \hat{y}_i}{y_i} \right)^2}$$
   - Prevents flagships (€25,000/day) from dominating the loss over small stores (€3,000/day).
   - Matches official competition benchmark and business planning tolerances.

4. **Confidence Intervals & Model Serialization:**
   - Epistemic uncertainty deduced across individual decision tree predictions:
     $$\hat{y}(x) \pm 1.96 \cdot \sigma_{\text{trees}}(x)$$
   - Models saved with timestamps (`dd-mm-yyyy-HH-MM-SS-00.pkl`) with a `latest_model.pkl` pointer.

5. **Deep Learning LSTM (`src/lstm_model.py`):**
   - Stationarity confirmed using Augmented Dickey-Fuller (ADF) test after first-order differencing ($p < 10^{-4}$).
   - ACF and PACF evaluated across 28 lags.
   - Supervised 14-day lookback sliding window.
   - 2-Layer LSTM with Dropout layers, achieving test RMSPE of ~0.2012.

6. **MLOps Tracking (`src/mlflow_tracking.py`):**
   - Centralized experiment tracking with parameters, metrics, artifact plots, and pipeline registry.

*Run commands:*
```bash
python run_project.py --train    # ML Pipeline
python run_project.py --lstm     # Deep Learning LSTM
python run_project.py --mlflow   # MLflow MLOps
```

---

### Task 3: Interactive Streamlit Web Dashboard (`app/app.py`)

Run the application:
```bash
streamlit run app/app.py
```

#### Dashboard Features:
* **Store Profile Sidebar:** Dynamic dropdown for all 1,115 stores displaying StoreType, Assortment, Competitor Distance, and Promo2 enrollment.
* **Tab 1 — Batch CSV Forecast & Download:**
  - Upload upcoming store schedule CSV (or download the auto-generated 6-week template).
  - Produces dual outputs: **Predicted Sales Turnover (€)** and **Predicted Customer Footfall**.
  - Displays dual-axis charts with 95% Confidence Interval error bands.
  - One-click CSV export (`predictions_store_<id>.csv`).
* **Tab 2 — Interactive Single-Day Forecast:**
  - What-if scenario testing: toggle Promo, select State Holiday, toggle School Holiday, and set Open/Closed status.
  - Computes real-time sales, visitor count, confidence bounds, and expected basket size (€/customer).
* **Tab 3 — Business Insights (EDA):**
  - Interactive gallery to inspect all 11 business behavior plots.
* **Tab 4 — Model Diagnostics & Loss Defense:**
  - Architecture breakdown of the Scikit-Learn Pipeline and 2-layer LSTM, feature importance rankings, and mathematical RMSPE defense.

---

### Task 4: Automated Testing

Run the test suite:
```bash
pytest tests/
# Or via runner:
python run_project.py --test
```
*Status:* **8/8 unit and integration tests passing** covering data ingestion, feature transformations, and prediction consistency.

---

## 📈 Executive Deliverables
* **Presentation:** Open `Rossmann_Sales_Forecasting_Presentation.pptx` for the 15-slide executive presentation prepared for stakeholders.
* **Jupyter Notebook:** Open `Rossmann_Sales_Forecasting_Complete.ipynb` for complete inline visualizations, code execution, and statistical validation.

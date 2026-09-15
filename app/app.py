import sys
import matplotlib.pyplot as plt
sys.modules.setdefault('numexpr', None)
sys.modules.setdefault('bottleneck', None)
"""
Task 3: Interactive Web Dashboard for Rossmann Store Sales & Customer Forecasting.
Provides single-record forecast and batch CSV upload with dual predictions
(Sales & Customers), confidence intervals, interactive visualizations, and CSV download.
"""

import sys
import io
from pathlib import Path
from datetime import date, timedelta

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

try:
    from src.predictor import SalesPredictor
except ImportError:
    from predictor import SalesPredictor

# Streamlit Page Setup
st.set_page_config(
    page_title="Rossmann Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3799;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4b6584;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f1f2f6;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_predictor():
    return SalesPredictor()


try:
    predictor = get_predictor()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Error loading forecasting model: {e}")

# Sidebar
st.sidebar.image("https://raw.githubusercontent.com/tandpfun/skill-icons/main/icons/Python-Dark.svg", width=50)
st.sidebar.title("Rossmann Pharmaceuticals")
st.sidebar.markdown("**Store Sales Prediction System**")
st.sidebar.markdown("---")

if model_loaded:
    st.sidebar.success(f"Active Model: `{predictor.model_name}`")
    all_stores = sorted(predictor.store_metadata["Store"].unique())
else:
    all_stores = list(range(1, 1116))

selected_store = st.sidebar.selectbox("Select Store ID", options=all_stores, index=0)

# Display Store Details in Sidebar
if model_loaded:
    st_meta = predictor.store_metadata[predictor.store_metadata["Store"] == selected_store].iloc[0]
    st.sidebar.markdown("### Store Profile")
    st.sidebar.write(f"- **Store Type:** `{st_meta['StoreType'].upper()}`")
    st.sidebar.write(f"- **Assortment:** `{st_meta['Assortment'].upper()}`")
    st.sidebar.write(f"- **Competitor Distance:** `{st_meta['CompetitionDistance']:,.0f} m`")
    st.sidebar.write(f"- **Promo2 Enrolled:** `{'Yes' if st_meta['Promo2'] == 1 else 'No'}`")

st.markdown('<div class="main-title">Store Sales & Customer Forecasting Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Serving machine learning & deep learning daily turnover predictions up to 6 weeks ahead</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "📁 Batch CSV Forecast & Download",
    "🎯 Single Day Prediction",
    "📊 Task 1 Business Insights (EDA)",
    "⚙️ Model Diagnostics & Loss Defense"
])

# ----------------- TAB 1: BATCH CSV FORECAST -----------------
with tab1:
    st.subheader("Upload Operational Schedule CSV")
    st.markdown("""
    Upload a CSV file containing upcoming dates and store conditions. Supported column names:
    - **Date** (e.g., `YYYY-MM-DD`)
    - **IsPromo** or `Promo` (0 or 1)
    - **IsHoliday** or `StateHoliday` (`0`, `a`=Public, `b`=Easter, `c`=Christmas)
    - **SchoolHoliday** (0 or 1)
    - Optional: `Store_id` or `Store`, `Open`
    """)

    col_up, col_demo = st.columns([3, 1])
    with col_up:
        uploaded_file = st.file_uploader("Choose CSV File", type=["csv"])
    with col_demo:
        st.markdown("**Need a sample template?**")
        # Generate demo 6-week template for selected store
        demo_dates = pd.date_range(date.today(), periods=42, freq="D")
        demo_df = pd.DataFrame({
            "Date": demo_dates.strftime("%Y-%m-%d"),
            "Store_id": selected_store,
            "IsPromo": [1 if d.weekday() < 5 and (i // 7) % 2 == 0 else 0 for i, d in enumerate(demo_dates)],
            "IsHoliday": ["0"] * 42,
            "SchoolHoliday": [0] * 42
        })
        demo_csv = demo_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download 6-Week Template CSV",
            data=demo_csv,
            file_name=f"forecast_template_store_{selected_store}.csv",
            mime="text/csv"
        )

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            st.info(f"Loaded {len(input_df)} rows from uploaded file.")

            results_df = predictor.predict(input_df, default_store_id=selected_store)

            # Filter for active store
            active_results = results_df[results_df["Store"] == selected_store].sort_values("Date").copy()
            if active_results.empty:
                active_results = results_df.sort_values("Date").copy()

            total_sales = active_results["Predicted_Sales"].sum()
            open_sales = active_results[active_results["Predicted_Sales"] > 0]["Predicted_Sales"]
            avg_daily_sales = open_sales.mean() if not open_sales.empty else 0.0
            total_customers = active_results["Predicted_Customers"].sum()
            if not active_results.empty and active_results["Predicted_Sales"].max() > 0:
                peak_day = active_results.loc[active_results["Predicted_Sales"].idxmax()]["Date"].strftime("%Y-%m-%d")
            else:
                peak_day = "N/A"

            # Metrics row
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Forecasted Sales", f"€{total_sales:,.0f}")
            m2.metric("Avg Daily Turnover (Open)", f"€{avg_daily_sales:,.0f}")
            m3.metric("Total Expected Customers", f"{total_customers:,.0f}")
            m4.metric("Peak Sales Day", peak_day)

            # Interactive Plot
            st.markdown("### Forecast Visualization")
            fig, ax1 = plt.subplots(figsize=(12, 4.5))
            ax2 = ax1.twinx()

            l1 = ax1.plot(
                active_results["Date"],
                active_results["Predicted_Sales"],
                color="#10ac84",
                linewidth=2.5,
                label="Predicted Sales (€)"
            )
            ax1.fill_between(
                active_results["Date"],
                active_results["CI_Lower_95"],
                active_results["CI_Upper_95"],
                color="#10ac84",
                alpha=0.15,
                label="95% Confidence Interval"
            )

            l2 = ax2.plot(
                active_results["Date"],
                active_results["Predicted_Customers"],
                color="#2e86de",
                linestyle="--",
                linewidth=2,
                label="Predicted Customers"
            )

            ax1.set_ylabel("Sales Turnover (€)", color="#10ac84", fontweight="bold")
            ax2.set_ylabel("Customer Footfall", color="#2e86de", fontweight="bold")
            ax1.set_title(f"Daily Forecast for Store #{selected_store} (Sales & Footfall)", fontsize=13, fontweight="bold")
            fig.autofmt_xdate()

            # Merged legend
            lines = l1 + [ax1.collections[0]] + l2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc="upper left")
            st.pyplot(fig)
            plt.close(fig)

            # Table display
            st.markdown("### Forecast Summary Table")
            display_cols = [
                "Date", "Store", "Promo", "StateHoliday",
                "Predicted_Sales", "CI_Lower_95", "CI_Upper_95", "Predicted_Customers"
            ]
            st.dataframe(active_results[display_cols], width="stretch")

            # Download CSV Button
            csv_buf = io.StringIO()
            active_results[display_cols].to_csv(csv_buf, index=False)
            st.download_button(
                label=f"⬇️ Download Predictions for Store {selected_store} (CSV)",
                data=csv_buf.getvalue(),
                file_name=f"predictions_store_{selected_store}.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Error executing batch prediction: {e}")

# ----------------- TAB 2: SINGLE DAY PREDICTION -----------------
with tab2:
    st.subheader("Interactive Single Day Forecast")
    c1, c2, c3 = st.columns(3)

    with c1:
        target_date = st.date_input("Forecast Date", value=date.today() + timedelta(days=1))
        promo_flag = st.selectbox("Promotion Active (Promo)?", options=[1, 0], format_func=lambda x: "Yes (Promo Running)" if x == 1 else "No Promo")
    with c2:
        holiday_val = st.selectbox(
            "State Holiday",
            options=["0", "a", "b", "c"],
            format_func=lambda x: {"0": "None", "a": "Public Holiday", "b": "Easter Holiday", "c": "Christmas"}[x]
        )
        school_holiday_val = st.selectbox("School Holiday?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    with c3:
        store_open_flag = st.selectbox(
            "Store Open Status",
            options=[1, 0],
            format_func=lambda x: "Open (Normal Operations)" if x == 1 else "Closed (0 Sales)"
        )

    if st.button("Generate Forecast", type="primary"):
        single_input = pd.DataFrame([{
            "Date": pd.to_datetime(target_date),
            "Store": selected_store,
            "Promo": promo_flag,
            "StateHoliday": holiday_val,
            "SchoolHoliday": school_holiday_val,
            "Open": store_open_flag
        }])

        single_pred = predictor.predict(single_input, default_store_id=selected_store).iloc[0]

        st.markdown("### Prediction Results")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Predicted Sales", f"€{single_pred['Predicted_Sales']:,.2f}")
        r2.metric("95% CI Lower Bound", f"€{single_pred['CI_Lower_95']:,.2f}")
        r3.metric("95% CI Upper Bound", f"€{single_pred['CI_Upper_95']:,.2f}")
        r4.metric("Predicted Customers", f"{single_pred['Predicted_Customers']:,} visitors")

        if single_pred["Predicted_Customers"] > 0:
            spc = single_pred["Predicted_Sales"] / single_pred["Predicted_Customers"]
            st.info(f"💡 Expected Basket Size: **€{spc:.2f}** per customer.")

# ----------------- TAB 3: TASK 1 EDA INSIGHTS -----------------
with tab3:
    st.subheader("Task 1: Exploration of Customer Purchasing Behaviour")
    st.markdown("Summary of findings addressing all 11 business questions investigated during exploratory analysis.")

    figs_dir = BASE_DIR / "reports" / "figures"
    fig_names = {
        "q1_promo_distribution.png": "1. Promotion Distribution: Train vs Test Sets",
        "q2_holiday_behavior.png": "2. Sales Behavior Before, During, and After Holidays",
        "q3_seasonality.png": "3. Seasonal Purchase Behaviors Across Calendar Months",
        "q4_sales_vs_customers.png": "4. Correlation Between Daily Sales and Customers",
        "q5_promo_effect.png": "5. Promotion Uplift on Turnover, Customers, and Basket Size",
        "q6_promo_deployment.png": "6. Strategic Promo Deployment by StoreType and Assortment",
        "q7_day_of_week_trends.png": "7. Customer Footfall & Sales by Day of Week",
        "q8_sunday_open_stores.png": "8. Weekend Sales of Sunday-Open Stores vs Standard Stores",
        "q9_assortment_effect.png": "9. Impact of Assortment Levels (Basic, Extra, Extended)",
        "q10_competition_distance.png": "10. Competitor Proximity & Urban Density Analysis",
        "q11_competitor_opening_impact.png": "11. Impact of New Competitor Openings on Existing Stores"
    }

    selected_fig = st.selectbox("Select Business Question to Explore:", list(fig_names.values()))
    # Reverse lookup
    target_filename = [k for k, v in fig_names.items() if v == selected_fig][0]
    target_img_path = figs_dir / target_filename

    if target_img_path.exists():
        st.image(str(target_img_path), width="stretch")
    else:
        st.warning(f"Plot file `{target_filename}` not found. Run `python run_project.py --eda` to generate it.")

# ----------------- TAB 4: MODEL DIAGNOSTICS & LOSS DEFENSE -----------------
with tab4:
    st.subheader("Model Diagnostics, Pipeline Architecture & Loss Defense")

    col_arch, col_eval = st.columns(2)
    with col_arch:
        st.markdown("### 🏆 Scikit-Learn Pipeline Architecture")
        st.code("""
Pipeline(steps=[
  ('preprocessor', ColumnTransformer(transformers=[
     ('num', StandardScaler(), NUMERIC_FEATURES),
     ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES)
  ])),
  ('regressor', RandomForestRegressor(n_estimators=60, max_depth=18))
])
        """, language="python")

        feat_imp_path = BASE_DIR / "reports" / "figures" / "feature_importance.png"
        if feat_imp_path.exists():
            st.image(str(feat_imp_path), caption="Top 15 Feature Importances", width="stretch")

    with col_eval:
        st.markdown("### 🧠 Deep Learning LSTM Architecture")
        st.markdown("""
        - **Model:** 2-Layer Recurrent Neural Network (LSTM)
        - **Sequence Window:** 14-day sliding lookback window
        - **Scaling:** `MinMaxScaler(feature_range=(-1, 1))`
        - **Stationarity:** Verified using Augmented Dickey-Fuller (ADF) test
        - **Differencing:** First-order differencing applied
        """)

        lstm_img_path = BASE_DIR / "reports" / "figures" / "lstm_forecast.png"
        if lstm_img_path.exists():
            st.image(str(lstm_img_path), caption="2-Layer LSTM Regression Forecast", width="stretch")

    st.markdown("---")
    st.markdown("### 🛡️ Task 2.3 Defense of Chosen Loss Function: RMSPE")
    st.markdown("""
    $$\\text{RMSPE} = \\sqrt{ \\frac{1}{N} \\sum_{i=1}^N \\left( \\frac{y_i - \\hat{y}_i}{y_i} \\right)^2 }$$

    1. **Scale Neutrality:** Stores with €25,000/day turnover would dominate standard squared error losses (MSE/RMSE), sacrificing smaller stores with €3,000/day turnover. RMSPE ensures equal percentage penalty across all store volumes.
    2. **Operational Interpretability:** Store managers understand percentage accuracy immediately (e.g. within 11% error margin).
    3. **Benchmark Alignment:** Directly matches the official Rossmann competition objective function.
    """)

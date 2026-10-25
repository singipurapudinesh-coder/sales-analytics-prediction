# ⭐ End-to-End Sales Analytics & Prediction System

A complete enterprise-grade sales analytics and machine learning prediction platform. This system ingests raw multi-channel sales data, cleans and validates it with Python (Pandas & NumPy), stores it in a normalized Star Schema SQL database (SQLite & MySQL compatible via SQLAlchemy), computes advanced business KPIs and customer intelligence (RFM segmentation, Cohort retention, Pareto 80/20 analysis), and forecasts future revenue using Scikit-Learn models with interactive Plotly visualizations in a modern Streamlit web dashboard.

---

## 🚀 Key Highlights & Architecture

```
                                      ┌────────────────────────────────────────┐
                                      │       1. Raw Sales Data Ingestion      │
                                      │   (Multi-channel, dirty data, noise)   │
                                      └──────────────────┬─────────────────────┘
                                                         │
                                                         ▼
                                      ┌────────────────────────────────────────┐
                                      │    2. Python / Pandas ETL Cleaner      │
                                      │ (Deduplication, Imputation, Anomaly    │
                                      │  Detection, KPI Feature Engineering)   │
                                      └──────────────────┬─────────────────────┘
                                                         │
                                                         ▼
                                      ┌────────────────────────────────────────┐
                                      │   3. Relational SQL Database Layer     │
                                      │   (Star Schema: dim_products,          │
                                      │    dim_customers, fact_sales)          │
                                      └───────┬────────────────────────┬───────┘
                                              │                        │
                      ┌───────────────────────┴──────┐  ┌──────────────┴────────────────────────┐
                      ▼                              ▼  ▼                                       ▼
        ┌────────────────────────────┐    ┌────────────────────────────┐    ┌────────────────────────────┐
        │  4. Business Analytics     │    │  5. Machine Learning       │    │  6. Interactive UI         │
        │ • Executive KPIs & MoM     │    │ • Time-Series Forecasting  │    │ • Streamlit Web Dashboard  │
        │ • RFM Customer Segments    │    │ • Ridge / RF / HistGBDT    │    │ • Plotly Visualizations    │
        │ • Cohort Retention Matrix  │    │ • 95% Confidence Intervals │    │ • SQL Query Lab & Console  │
        │ • Pareto 80/20 Distribution│    │ • "What-If" Levers & PED   │    │ • Real-time Data Filters   │
        └────────────────────────────┘    └────────────────────────────┘    └────────────────────────────┘
```

---

## 🛠️ Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Data Cleaning & ETL** | Python 3, Pandas, NumPy |
| **Database & Storage** | SQL, SQLAlchemy, SQLite (local zero-config), MySQL (remote connector) |
| **Predictive Machine Learning** | Scikit-learn (HistGradientBoosting, Random Forest, Ridge Regression) |
| **Interactive Visualizations** | Plotly Express & Plotly Graph Objects |
| **User Interface** | Streamlit (Multi-tab responsive layout) |
| **Testing** | Pytest, NumPy/Pandas testing assertions |

---

## 📂 Project Directory Structure

```
sales-analytics-system/
├── app.py                      # Main interactive Streamlit application
├── run.py                      # CLI runner for all pipeline stages
├── requirements.txt            # Python dependencies
├── data/
│   ├── raw/                    # Raw transaction datasets (with intentional anomalies)
│   ├── processed/              # Cleaned & feature-engineered datasets
│   └── database/               # Local SQLite database (sales_analytics.db)
├── models/                     # Serialized scikit-learn forecast models
├── src/
│   ├── __init__.py
│   ├── data_generator.py       # Realistic enterprise sales dataset generator
│   ├── cleaner.py              # Robust ETL cleaning and feature engineering
│   ├── database.py             # SQLAlchemy Star Schema models & query interface
│   ├── analytics.py            # Executive KPIs, RFM, Cohorts, and Pareto analysis
│   ├── predictor.py            # ML forecasting, benchmarking, & What-If simulator
│   └── visuals.py              # High-contrast corporate Plotly charts
└── tests/
    └── test_pipeline.py        # Automated test suite
```

---

## ⚡ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the End-to-End Pipeline
Execute data generation, cleaning, SQL storage, and model training in one command:
```bash
python run.py --all
```

Or run individual pipeline stages via CLI flags:
```bash
python run.py --generate   # Generate fresh raw data with real-world noise
python run.py --clean      # Run the Pandas ETL cleaning pipeline
python run.py --load-db    # Ingest cleaned data into the SQL database
python run.py --train      # Train Scikit-learn forecasting models
```

### 3. Launch the Interactive Dashboard
```bash
streamlit run app.py
```
Or via the CLI runner:
```bash
python run.py --serve
```

---

## 🎯 Dashboard Capabilities

### 1. 📊 Executive Overview
- Real-time KPIs: **Total Net Revenue**, **Net Profit & Margin %**, **Units Sold**, **Average Order Value (AOV)**, **Repeat Customer Rate**, and **Return Rate %**.
- Interactive Revenue & Profit Trajectory with a 7-day rolling average.
- Revenue breakdown by Category & Geographic Region.
- Top 10 Best-Selling Products table with profit margins.

### 2. 🧹 ETL & Data Quality Studio
- Data Quality Health Score indicator (0-100%).
- Full audit log displaying: rows ingested, duplicate transactions removed, negative quantities corrected, null values imputed, and invalid promotion discounts clamped.
- Side-by-side interactive viewer: Raw Dirty Data vs Cleaned & Normalized Data.
- In-browser file uploader to ingest and clean your own sales CSV files.

### 3. 📈 Deep-Dive Customer & Sales Analytics
- **RFM Customer Segmentation**: Recency, Frequency, and Monetary scoring assigning customers into behavioral clusters (*Champions*, *Loyal Customers*, *At Risk*, *Lost*, etc.) visualized via an interactive Treemap.
- **Cohort Retention Heatmap**: Month-over-month customer retention percentages tracking lifetime repurchase habits.
- **Pareto 80/20 Analysis**: Cumulative revenue curves distinguishing the vital 20% product drivers from long-tail inventory.
- **Omnichannel Performance**: Order volumes, margins, and return rates per sales channel (*Online Store*, *Mobile App*, *Marketplace*, *B2B Portal*, *Direct Sales*).

### 4. 🤖 AI Sales Forecasting & Predictive Modeling
- Automated feature engineering: Multi-day lags ($t-1$, $t-7$, $t-14$, $t-30$), rolling statistics, day-of-week seasonality, and quarterly cycles.
- Compares and benchmarks **Ridge Regression**, **Random Forest Regressor**, and **Hist Gradient Boosting**.
- Automated model evaluation metrics: **RMSE**, **MAE**, **R² Score**, and **MAPE**.
- Multi-step recursive forecasting for 14 to 90 days ahead with **95% Confidence Intervals** (uncertainty cone).
- Permutation & tree feature importance chart showing key sales drivers.

### 5. 🎯 "What-If" Strategic Scenario Simulator
- Dynamic interactive sliders to adjust:
  - **Price Adjustment %** (e.g. $+5\%$)
  - **Promotional Discount Change %** (e.g. $-2\%$)
  - **Price Elasticity of Demand (PED)**
  - **Marketing Campaign Uplift %**
- Real-time financial projections showing projected Net Revenue, Net Profit, and Margin deltas.

### 6. 🗄️ SQL Explorer & Query Lab
- Visual Star Schema inspector displaying `dim_products`, `dim_customers`, and `fact_sales` tables with live row counts and datatypes.
- Pre-built 1-click analytical queries (Monthly Run Rate, High Margin Products, Customer Lifetime Value, Regional Delivery Efficiency).
- Interactive SQL Editor allowing custom SQL execution against SQLite or MySQL with instant table preview and CSV download.

---

## 🧪 Running Automated Tests

Run the complete test suite to verify data cleaning, database operations, analytics, and machine learning components:
```bash
pytest tests/test_pipeline.py -v
```


"""
End-to-End Sales Analytics & Prediction System
Streamlit Web Application
An interactive executive intelligence platform combining ETL, SQL, Analytics, ML Forecasting, and Scenario Simulation.
"""

import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st

# Add src to python path for clean modular imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.data_generator import generate_raw_sales_data
from src.cleaner import SalesDataCleaner, clean_sales_file
from src.database import SalesDatabase, get_default_database
from src.analytics import SalesAnalytics
from src.predictor import SalesForecaster, ScenarioSimulator, save_forecaster_model, load_forecaster_model
from src.visuals import (
    plot_sales_and_profit_timeline,
    plot_category_distribution,
    plot_regional_performance,
    plot_rfm_treemap,
    plot_cohort_heatmap,
    plot_pareto_analysis,
    plot_forecast_timeline,
    plot_feature_importances,
    plot_data_quality_gauge,
)

# Page configuration
st.set_page_config(
    page_title="Sales Analytics & Prediction System",
    page_icon="⭐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for executive corporate styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 10px;
    }
    .kpi-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 4px;
    }
    .kpi-delta {
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 4px;
    }
    .delta-positive { color: #10B981; }
    .delta-negative { color: #EF4444; }
    .badge-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        background-color: #EFF6FF;
        color: #2563EB;
    }
</style>
""", unsafe_allow_html=True)


# Paths setup
DATA_DIR = os.path.join(current_dir, "data")
RAW_CSV_PATH = os.path.join(DATA_DIR, "raw", "raw_sales_data.csv")
PROCESSED_CSV_PATH = os.path.join(DATA_DIR, "processed", "cleaned_sales_data.csv")
MODELS_DIR = os.path.join(current_dir, "models")


# -------------------------------------------------------------
# Caching Data & Model Ingestion
# -------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data():
    """Load or generate baseline datasets."""
    # If cleaned data exists, load it
    if os.path.exists(PROCESSED_CSV_PATH):
        df_clean = pd.read_csv(PROCESSED_CSV_PATH)
        df_clean["order_date"] = pd.to_datetime(df_clean["order_date"])
        df_clean["ship_date"] = pd.to_datetime(df_clean["ship_date"])
    else:
        # Generate and clean on initial run
        if not os.path.exists(RAW_CSV_PATH):
            generate_raw_sales_data(num_records=8000, output_path=RAW_CSV_PATH)
        df_clean, _ = clean_sales_file(RAW_CSV_PATH, PROCESSED_CSV_PATH)

    # Load raw sample if exists
    df_raw = pd.read_csv(RAW_CSV_PATH) if os.path.exists(RAW_CSV_PATH) else df_clean.copy()
    return df_raw, df_clean


@st.cache_resource(show_spinner=False)
def get_db_instance():
    """Return database connection instance."""
    db = get_default_database()
    return db


# Initialize data
df_raw, df_clean = load_data()
db = get_db_instance()


# -------------------------------------------------------------
# Sidebar Configuration & Filters
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🏢 Enterprise Sales Intelligence")
    st.caption("End-to-End Analytics & Prediction")
    st.divider()

    selected_view = st.radio(
        "Navigation",
        [
            "📊 Executive Overview",
            "🧹 ETL & Data Quality",
            "📈 Deep-Dive Analytics",
            "🤖 AI Sales Forecasting",
            "🎯 What-If Simulator",
            "🗄️ SQL Explorer & Lab",
        ],
        index=0,
    )

    st.divider()
    st.markdown("#### 🔍 Global Data Filters")

    min_date = df_clean["order_date"].min().date()
    max_date = df_clean["order_date"].max().date()

    date_range = st.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    all_categories = sorted(list(df_clean["category"].dropna().unique()))
    selected_categories = st.multiselect("Category Filter", options=all_categories, default=all_categories)

    all_regions = sorted(list(df_clean["region"].dropna().unique()))
    selected_regions = st.multiselect("Region Filter", options=all_regions, default=all_regions)

    all_segments = sorted(list(df_clean["customer_segment"].dropna().unique()))
    selected_segments = st.multiselect("Customer Segment", options=all_segments, default=all_segments)

    st.divider()
    st.caption("Engine: **Python 3.14 | Pandas | SQLite/MySQL | Scikit-learn**")


# Apply global filters to dataframe
filtered_df = df_clean.copy()

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_d, end_d = date_range
    filtered_df = filtered_df[
        (filtered_df["order_date"].dt.date >= start_d) & (filtered_df["order_date"].dt.date <= end_d)
    ]

if selected_categories:
    filtered_df = filtered_df[filtered_df["category"].isin(selected_categories)]

if selected_regions:
    filtered_df = filtered_df[filtered_df["region"].isin(selected_regions)]

if selected_segments:
    filtered_df = filtered_df[filtered_df["customer_segment"].isin(selected_segments)]

# Safeguard if filters yield empty data
if filtered_df.empty:
    st.warning("⚠️ No data matches the selected filters. Please adjust your criteria in the sidebar.")
    st.stop()

analytics = SalesAnalytics(filtered_df)


# =============================================================
# VIEW 1: EXECUTIVE OVERVIEW
# =============================================================
if selected_view == "📊 Executive Overview":
    st.markdown('<div class="main-header">Executive Sales Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Real-time enterprise revenue, profit margins, and volume analytics</div>', unsafe_allow_html=True)

    kpis = analytics.calculate_executive_kpis()

    # KPI Metric Cards Row
    k1, k2, k3, k4, k5 = st.columns(5)

    with k1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Net Revenue</div>
            <div class="kpi-value">${kpis['total_revenue']:,.0f}</div>
            <div class="kpi-delta {'delta-positive' if kpis['mom_growth_pct'] >= 0 else 'delta-negative'}">
                {'▲' if kpis['mom_growth_pct'] >= 0 else '▼'} {kpis['mom_growth_pct']}% MoM
            </div>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Net Profit</div>
            <div class="kpi-value">${kpis['total_profit']:,.0f}</div>
            <div class="kpi-delta delta-positive">Margin: {kpis['profit_margin_pct']}%</div>
        </div>
        """, unsafe_allow_html=True)

    with k3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Orders</div>
            <div class="kpi-value">{kpis['total_orders']:,}</div>
            <div class="kpi-delta">Units: {kpis['total_units_sold']:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with k4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Average Order Value</div>
            <div class="kpi-value">${kpis['avg_order_value']:,.2f}</div>
            <div class="kpi-delta">Avg Promo: {kpis['avg_discount_pct']}%</div>
        </div>
        """, unsafe_allow_html=True)

    with k5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Customer Retention</div>
            <div class="kpi-value">{kpis['repeat_customer_rate_pct']}%</div>
            <div class="kpi-delta {'delta-negative' if kpis['return_rate_pct'] > 5 else 'delta-positive'}">
                Return Rate: {kpis['return_rate_pct']}%
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # Timeline Chart
    st.plotly_chart(plot_sales_and_profit_timeline(filtered_df), use_container_width=True)

    # Category and Regional Split
    col_left, col_right = st.columns([1, 1])

    with col_left:
        cat_df = filtered_df.groupby("category").agg(
            total_revenue=("net_sales", "sum"),
            total_profit=("profit", "sum"),
        ).reset_index()
        st.plotly_chart(plot_category_distribution(cat_df), use_container_width=True)

    with col_right:
        reg_df = filtered_df.groupby("region").agg(
            total_revenue=("net_sales", "sum"),
            total_profit=("profit", "sum"),
        ).reset_index()
        st.plotly_chart(plot_regional_performance(reg_df), use_container_width=True)

    # Top Products Table
    st.markdown("### 🏆 Top 10 Revenue Generating Products")
    top_prods = filtered_df.groupby(["product_id", "product_name", "category"]).agg(
        units_sold=("quantity", "sum"),
        revenue=("net_sales", "sum"),
        profit=("profit", "sum"),
        avg_margin=("profit_margin_pct", "mean"),
    ).reset_index().sort_values(by="revenue", ascending=False).head(10)

    top_prods["revenue"] = top_prods["revenue"].apply(lambda x: f"${x:,.2f}")
    top_prods["profit"] = top_prods["profit"].apply(lambda x: f"${x:,.2f}")
    top_prods["avg_margin"] = top_prods["avg_margin"].apply(lambda x: f"{x:.1f}%")
    st.dataframe(top_prods, use_container_width=True, hide_index=True)


# =============================================================
# VIEW 2: ETL & DATA QUALITY STUDIO
# =============================================================
elif selected_view == "🧹 ETL & Data Quality":
    st.markdown('<div class="main-header">ETL Pipeline & Data Quality Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Automated data extraction, cleaning audit, anomaly repair, and schema validation</div>', unsafe_allow_html=True)

    cleaner = SalesDataCleaner()
    # Re-run cleaner to get audit log
    _, audit_log = cleaner.clean_pipeline(df_raw)

    q_col1, q_col2, q_col3, q_col4 = st.columns([1.2, 1, 1, 1])
    with q_col1:
        st.plotly_chart(plot_data_quality_gauge(audit_log["data_quality_score_pct"]), use_container_width=True)
    with q_col2:
        st.metric("Raw Rows Ingested", f"{audit_log['initial_rows']:,}")
        st.metric("Duplicates Removed", f"{audit_log['duplicates_removed']:,}")
    with q_col3:
        st.metric("Clean Rows Retained", f"{audit_log['final_rows']:,}")
        st.metric("Negative Qty Fixed", f"{audit_log['outlier_quantities_fixed']:,}")
    with q_col4:
        st.metric("Null Values Handled", f"{audit_log['initial_null_cells']:,}")
        st.metric("Invalid Promos Fixed", f"{audit_log['outlier_discounts_fixed']:,}")

    st.divider()

    st.markdown("### 🔄 Interactive ETL Pipeline Runner")
    etl_col1, etl_col2 = st.columns(2)

    with etl_col1:
        st.markdown("**Regenerate Synthetic Dataset**")
        st.caption("Generate a fresh transaction stream with configurable anomalies to test the cleaning pipeline.")
        record_count = st.slider("Record Volume", min_value=1000, max_value=20000, value=8500, step=500)
        inject_noise = st.checkbox("Inject Data Quality Anomalies (Missing, Outliers, Dirty Strings)", value=True)
        if st.button("🚀 Generate New Raw Dataset", type="primary"):
            with st.spinner("Generating raw transactions with realistic seasonality and anomalies..."):
                generate_raw_sales_data(
                    num_records=record_count,
                    inject_anomalies=inject_noise,
                    output_path=RAW_CSV_PATH,
                )
                clean_sales_file(RAW_CSV_PATH, PROCESSED_CSV_PATH)
                st.cache_data.clear()
                st.success("New raw dataset generated and cleaned successfully! Reloading...")
                st.rerun()

    with etl_col2:
        st.markdown("**Upload External Sales Data**")
        st.caption("Upload a custom CSV file to run through our cleaning and feature engineering pipeline.")
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
        if uploaded_file is not None:
            if st.button("📥 Ingest and Clean Uploaded File"):
                try:
                    df_uploaded = pd.read_csv(uploaded_file)
                    df_up_cleaned, up_audit = cleaner.clean_pipeline(df_uploaded)
                    df_up_cleaned.to_csv(PROCESSED_CSV_PATH, index=False)
                    st.cache_data.clear()
                    st.success(f"Successfully processed {len(df_up_cleaned)} uploaded transactions!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")

    st.divider()
    st.markdown("### 🔍 Raw vs Cleaned Dataset Comparison")
    diff_tab1, diff_tab2 = st.tabs(["Cleaned & Feature-Engineered Data", "Raw Dirty Data Preview"])
    with diff_tab1:
        st.caption("Standardized ISO dates, clean numbers, engineered KPIs (Gross Sales, Net Sales, Profit, Delivery Days).")
        st.dataframe(df_clean.head(50), use_container_width=True)
    with diff_tab2:
        st.caption("Uncleaned raw feed containing string whitespace, currency formats ($), NaNs, and negative entries.")
        st.dataframe(df_raw.head(50), use_container_width=True)


# =============================================================
# VIEW 3: DEEP-DIVE ANALYTICS
# =============================================================
elif selected_view == "📈 Deep-Dive Analytics":
    st.markdown('<div class="main-header">Deep-Dive Customer & Sales Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">RFM Customer Segmentation, Cohort Retention Heatmap, and Pareto 80/20 Distribution</div>', unsafe_allow_html=True)

    analytics_tab1, analytics_tab2, analytics_tab3, analytics_tab4 = st.tabs([
        "👥 RFM Customer Segmentation",
        "📅 Cohort Retention Heatmap",
        "⚖️ Pareto 80/20 Product Analysis",
        "🌐 Channel Attribution",
    ])

    with analytics_tab1:
        rfm_df, segment_summary = analytics.compute_rfm_segmentation()
        st.plotly_chart(plot_rfm_treemap(segment_summary), use_container_width=True)
        
        st.markdown("#### Segment Distribution & Economic Value")
        st.dataframe(
            segment_summary.rename(columns={
                "segment": "Customer Segment",
                "customer_count": "Customers",
                "avg_recency": "Avg Recency (Days)",
                "avg_frequency": "Avg Orders",
                "total_revenue": "Total Spend ($)",
                "revenue_share_pct": "Spend Share (%)",
                "avg_revenue_per_cust": "Avg Spend / Customer ($)",
            }),
            use_container_width=True,
            hide_index=True,
        )

    with analytics_tab2:
        st.markdown("#### Customer Retention Matrix Across Acquisition Cohorts")
        st.caption("Percentage of customers who return to make subsequent purchases 1 to 12 months after their first transaction.")
        cohort_matrix = analytics.compute_cohort_retention()
        st.plotly_chart(plot_cohort_heatmap(cohort_matrix), use_container_width=True)

    with analytics_tab3:
        prod_pareto, pareto_summary = analytics.compute_pareto_analysis()
        p1, p2, p3 = st.columns(3)
        p1.metric("Total Catalog Products", f"{pareto_summary['total_products']:,}")
        p2.metric("Core 80% Revenue Drivers", f"{pareto_summary['vital_products_count']} products")
        p3.metric("Catalog Share Driving 80% Rev", f"{pareto_summary['vital_products_pct']}%")
        st.plotly_chart(plot_pareto_analysis(prod_pareto), use_container_width=True)

    with analytics_tab4:
        channel_df = analytics.compute_channel_attribution()
        st.markdown("#### Omnichannel Sales & Profit Efficiency")
        st.dataframe(
            channel_df.rename(columns={
                "sales_channel": "Channel",
                "order_count": "Orders",
                "units_sold": "Units Sold",
                "total_revenue": "Net Revenue ($)",
                "total_profit": "Net Profit ($)",
                "profit_margin_pct": "Margin (%)",
                "return_rate_pct": "Return Rate (%)",
                "avg_discount_pct": "Avg Promo (%)",
                "revenue_share_pct": "Revenue Share (%)",
            }),
            use_container_width=True,
            hide_index=True,
        )


# =============================================================
# VIEW 4: AI SALES FORECASTING
# =============================================================
elif selected_view == "🤖 AI Sales Forecasting":
    st.markdown('<div class="main-header">AI Sales Forecasting & Predictive Modeling</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Automated time-series feature engineering, multi-model benchmarking, and future demand forecasting with 95% confidence intervals</div>', unsafe_allow_html=True)

    forecaster = SalesForecaster()
    daily_featured = forecaster.prepare_time_series_features(filtered_df)

    f_col1, f_col2, f_col3 = st.columns([1, 1, 1])
    with f_col1:
        forecast_horizon = st.slider("Forecast Horizon (Days Ahead)", min_value=14, max_value=90, value=30, step=7)
    with f_col2:
        test_period = st.slider("Evaluation Test Window (Days)", min_value=14, max_value=60, value=45, step=7)
    with f_col3:
        st.write("")
        st.write("")
        retrain_clicked = st.button("⚡ Train & Retrain Models", type="primary")

    with st.spinner("Training ML Regressors (Ridge, Random Forest, Hist Gradient Boosting)..."):
        metrics_df, test_preds = forecaster.train_and_evaluate(daily_featured, test_days=test_period)
        future_forecast = forecaster.forecast_future(daily_featured, horizon_days=forecast_horizon)
        feature_imp = forecaster.get_feature_importances()
        save_forecaster_model(forecaster, MODELS_DIR)

    # Model Evaluation Benchmark Table
    st.markdown("#### 🏆 Model Benchmark & Performance Comparison")
    st.caption(f"Best Performing Model Selected: **{forecaster.best_model_name}**")
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    # Forecast Plot
    st.plotly_chart(
        plot_forecast_timeline(
            daily_featured,
            future_forecast,
            test_preds_df=test_preds,
            best_model_name=forecaster.best_model_name,
        ),
        use_container_width=True,
    )

    # Feature Importance Plot
    col_imp, col_table = st.columns([1, 1])
    with col_imp:
        st.plotly_chart(plot_feature_importances(feature_imp), use_container_width=True)
    with col_table:
        st.markdown("#### 📅 Future Forecast Data Table")
        future_forecast_display = future_forecast.copy()
        future_forecast_display["date"] = future_forecast_display["date"].dt.strftime("%Y-%m-%d")
        future_forecast_display.rename(columns={
            "date": "Date",
            "forecast_sales": "Forecast Revenue ($)",
            "lower_bound": "95% Lower ($)",
            "upper_bound": "95% Upper ($)",
        }, inplace=True)
        st.dataframe(future_forecast_display, use_container_width=True, height=350, hide_index=True)
        csv_data = future_forecast_display.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Forecast CSV", data=csv_data, file_name="sales_forecast.csv", mime="text/csv")


# =============================================================
# VIEW 5: WHAT-IF SCENARIO SIMULATOR
# =============================================================
elif selected_view == "🎯 What-If Simulator":
    st.markdown('<div class="main-header">"What-If" Strategic Scenario Simulator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Simulate business outcomes of price adjustments, promotional discounts, elasticity, and marketing campaigns</div>', unsafe_allow_html=True)

    sim_col1, sim_col2 = st.columns([1, 1])

    with sim_col1:
        st.markdown("#### 🎛️ Scenario Levers & Assumptions")
        price_delta = st.slider("Price Adjustment (%)", min_value=-25.0, max_value=30.0, value=5.0, step=1.0)
        discount_delta = st.slider("Discount Promo Adjustment (%)", min_value=-15.0, max_value=20.0, value=-2.0, step=1.0)
        elasticity = st.slider("Price Elasticity of Demand (PED)", min_value=-2.5, max_value=-0.2, value=-1.2, step=0.1)
        marketing_uplift = st.slider("Marketing Campaign Volume Uplift (%)", min_value=0.0, max_value=40.0, value=8.0, step=1.0)

    sim_results = ScenarioSimulator.simulate(
        filtered_df,
        price_change_pct=price_delta,
        discount_change_pct=discount_delta,
        elasticity_coefficient=elasticity,
        marketing_uplift_pct=marketing_uplift,
    )

    with sim_col2:
        st.markdown("#### 📊 Projected Business Impact")
        
        r1, r2 = st.columns(2)
        with r1:
            st.metric(
                "Projected Net Revenue",
                f"${sim_results['simulated_revenue']:,.0f}",
                delta=f"{sim_results['revenue_delta_pct']:+.1f}% vs Baseline",
            )
            st.metric(
                "Projected Total Units",
                f"{sim_results['simulated_units']:,}",
                delta=f"{((sim_results['simulated_units'] - sim_results['baseline_units']) / sim_results['baseline_units'] * 100):+.1f}%",
            )
        with r2:
            st.metric(
                "Projected Net Profit",
                f"${sim_results['simulated_profit']:,.0f}",
                delta=f"{sim_results['profit_delta_pct']:+.1f}% vs Baseline",
            )
            st.metric(
                "Projected Profit Margin",
                f"{sim_results['simulated_margin_pct']:.1f}%",
                delta=f"{(sim_results['simulated_margin_pct'] - sim_results['baseline_margin_pct']):+.1f}% pts",
            )

    st.divider()

    # Visual comparison chart
    compare_df = pd.DataFrame([
        {"Metric": "Revenue ($)", "Scenario": "Baseline", "Amount": sim_results["baseline_revenue"]},
        {"Metric": "Revenue ($)", "Scenario": "Simulated", "Amount": sim_results["simulated_revenue"]},
        {"Metric": "Profit ($)", "Scenario": "Baseline", "Amount": sim_results["baseline_profit"]},
        {"Metric": "Profit ($)", "Scenario": "Simulated", "Amount": sim_results["simulated_profit"]},
    ])

    import plotly.express as px
    fig_sim = px.bar(
        compare_df,
        x="Metric",
        y="Amount",
        color="Scenario",
        barmode="group",
        color_discrete_map={"Baseline": "#64748B", "Simulated": "#10B981"},
        text_auto="$,.0f",
        title="Baseline vs. Simulated Financial Performance",
    )
    st.plotly_chart(fig_sim, use_container_width=True)


# =============================================================
# VIEW 6: SQL EXPLORER & QUERY LAB
# =============================================================
elif selected_view == "🗄️ SQL Explorer & Lab":
    st.markdown('<div class="main-header">SQL Database & Query Lab</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Interactive SQL editor, Star Schema inspector, and analytical queries across SQLite & MySQL</div>', unsafe_allow_html=True)

    # Ingest data into DB if tables are empty
    table_meta = db.get_table_metadata()
    if not table_meta or all(m["row_count"] == 0 for m in table_meta.values()):
        with st.spinner("Populating Star Schema tables in SQL database..."):
            db.load_cleaned_dataframe(df_clean)
            table_meta = db.get_table_metadata()

    # Schema Inspector
    st.markdown("#### 📐 Relational Star Schema")
    meta_cols = st.columns(len(table_meta))
    for i, (tbl_name, meta) in enumerate(table_meta.items()):
        with meta_cols[i % len(meta_cols)]:
            st.markdown(f"**`{tbl_name}`**")
            st.caption(f"{meta['row_count']:,} records")
            with st.expander("Columns"):
                for col in meta["columns"]:
                    st.text(f"• {col['name']} ({col['type']})")

    st.divider()

    # Pre-built Analytical SQL Queries
    PRESET_QUERIES = {
        "Monthly Revenue & Profit Run Rate": """SELECT 
    SUBSTR(order_date, 1, 7) AS year_month,
    COUNT(order_id) AS total_orders,
    SUM(quantity) AS total_units_sold,
    ROUND(SUM(net_sales), 2) AS total_net_revenue,
    ROUND(SUM(profit), 2) AS total_profit,
    ROUND(AVG(profit_margin_pct), 2) AS avg_profit_margin_pct
FROM fact_sales
GROUP BY SUBSTR(order_date, 1, 7)
ORDER BY year_month DESC;""",

        "Top 10 High-Margin Products": """SELECT 
    p.product_name,
    p.category,
    p.subcategory,
    SUM(f.quantity) AS units_sold,
    ROUND(SUM(f.net_sales), 2) AS total_revenue,
    ROUND(SUM(f.profit), 2) AS total_profit,
    ROUND((SUM(f.profit) / SUM(f.net_sales)) * 100, 2) AS profit_margin_pct
FROM fact_sales f
JOIN dim_products p ON f.product_id = p.product_id
GROUP BY p.product_id, p.product_name, p.category, p.subcategory
ORDER BY total_profit DESC
LIMIT 10;""",

        "Customer Segment Lifetime Value": """SELECT 
    c.customer_segment,
    COUNT(DISTINCT c.customer_id) AS total_customers,
    COUNT(f.order_id) AS total_orders,
    ROUND(SUM(f.net_sales), 2) AS total_revenue,
    ROUND(SUM(f.net_sales) / COUNT(DISTINCT c.customer_id), 2) AS revenue_per_customer,
    ROUND(AVG(f.discount_rate) * 100, 2) AS avg_discount_pct
FROM fact_sales f
JOIN dim_customers c ON f.customer_id = c.customer_id
GROUP BY c.customer_segment
ORDER BY total_revenue DESC;""",

        "Regional Delivery & Shipping Efficiency": """SELECT 
    c.region,
    COUNT(f.order_id) AS order_volume,
    ROUND(AVG(f.delivery_days), 1) AS avg_delivery_days,
    ROUND(AVG(f.shipping_cost), 2) AS avg_shipping_cost,
    ROUND(SUM(f.net_sales), 2) AS total_sales,
    ROUND(SUM(CASE WHEN LOWER(f.order_status) = 'returned' THEN 1 ELSE 0 END) * 100.0 / COUNT(f.order_id), 2) AS return_rate_pct
FROM fact_sales f
JOIN dim_customers c ON f.customer_id = c.customer_id
GROUP BY c.region
ORDER BY total_sales DESC;""",
    }

    st.markdown("#### ⚡ SQL Query Lab")
    preset_choice = st.selectbox("Select Preset Analytical Query", options=["-- Custom Query --"] + list(PRESET_QUERIES.keys()))

    initial_sql = PRESET_QUERIES[preset_choice] if preset_choice != "-- Custom Query --" else "SELECT * FROM fact_sales LIMIT 10;"

    sql_input = st.text_area("SQL Editor", value=initial_sql, height=180)

    if st.button("▶️ Execute Query", type="primary"):
        try:
            with st.spinner("Executing query..."):
                query_result = db.execute_query(sql_input)
                st.success(f"Query returned {len(query_result)} rows successfully!")
                st.dataframe(query_result, use_container_width=True)
                
                # CSV Download option
                csv_bytes = query_result.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Query Results as CSV", data=csv_bytes, file_name="query_results.csv", mime="text/csv")
        except Exception as e:
            st.error(f"SQL Execution Error: {str(e)}")

    st.divider()
    with st.expander("🔌 MySQL External Database Connection Setup"):
        st.caption("You can connect this system to a remote MySQL server by providing the connection credentials below.")
        mysql_user = st.text_input("MySQL User", value="root")
        mysql_pass = st.text_input("MySQL Password", type="password", value="")
        mysql_host = st.text_input("MySQL Host", value="localhost")
        mysql_port = st.text_input("MySQL Port", value="3306")
        mysql_db = st.text_input("MySQL Database Name", value="sales_analytics")
        if st.button("Test MySQL Connection"):
            try:
                uri = f"mysql+mysqlconnector://{mysql_user}:{mysql_pass}@{mysql_host}:{mysql_port}/{mysql_db}"
                test_db = SalesDatabase(db_uri=uri)
                st.success("Successfully connected to MySQL database!")
            except Exception as e:
                st.warning(f"Could not connect to MySQL ({str(e)}). Falling back seamlessly to local SQLite engine.")


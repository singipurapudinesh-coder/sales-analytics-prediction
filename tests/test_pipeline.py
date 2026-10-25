"""
End-to-End Sales Analytics & Prediction System
Unit & Integration Test Suite
Verifies data generation, ETL cleaning, SQL storage, analytical computations, and ML forecasting.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_generator import generate_raw_sales_data
from src.cleaner import SalesDataCleaner
from src.database import SalesDatabase
from src.analytics import SalesAnalytics
from src.predictor import SalesForecaster, ScenarioSimulator


@pytest.fixture(scope="session")
def sample_raw_data():
    """Generate small synthetic raw dataset for fast test execution."""
    df = generate_raw_sales_data(
        num_records=600,
        start_date="2024-01-01",
        end_date="2024-08-31",
        inject_anomalies=True,
        random_state=42,
    )
    return df


@pytest.fixture(scope="session")
def cleaned_data(sample_raw_data):
    """Clean the raw test dataset."""
    cleaner = SalesDataCleaner()
    df_clean, audit = cleaner.clean_pipeline(sample_raw_data)
    return df_clean, audit


def test_data_generation(sample_raw_data):
    """Test raw sales data generation properties."""
    assert isinstance(sample_raw_data, pd.DataFrame)
    assert len(sample_raw_data) >= 600
    expected_cols = [
        "order_id", "order_date", "customer_id", "product_id",
        "category", "subcategory", "unit_price", "quantity", "discount_rate"
    ]
    for col in expected_cols:
        assert col in sample_raw_data.columns


def test_data_cleaning_and_features(cleaned_data):
    """Test data cleaning, anomaly handling, and feature engineering."""
    df_clean, audit = cleaned_data
    
    # Assert features exist
    engineered_cols = [
        "gross_sales", "discount_amount", "net_sales", "cogs", "profit",
        "profit_margin_pct", "delivery_days", "order_year", "order_month"
    ]
    for col in engineered_cols:
        assert col in df_clean.columns

    # Assert no nulls in critical numeric columns
    assert df_clean["net_sales"].isna().sum() == 0
    assert df_clean["profit"].isna().sum() == 0
    assert (df_clean["quantity"] <= 0).sum() == 0
    assert (df_clean["discount_rate"] < 0).sum() == 0
    assert (df_clean["discount_rate"] > 0.75).sum() == 0

    # Mathematical consistency
    computed_net = (df_clean["gross_sales"] - df_clean["discount_amount"]).round(2)
    assert np.allclose(df_clean["net_sales"], computed_net, atol=0.05)

    # Audit log check
    assert audit["final_rows"] > 0
    assert audit["data_quality_score_pct"] >= 85.0


def test_database_operations(cleaned_data, tmp_path):
    """Test SQLite / SQL Database ingestion and queries."""
    df_clean, _ = cleaned_data
    test_db_path = os.path.join(tmp_path, "test_sales.db")
    db = SalesDatabase(db_uri=f"sqlite:///{test_db_path}")

    # Load data
    load_result = db.load_cleaned_dataframe(df_clean)
    assert load_result["sales_loaded"] > 0
    assert load_result["products_loaded"] > 0
    assert load_result["customers_loaded"] > 0

    # Query KPIs
    kpi_dict = db.get_kpi_summary()
    assert kpi_dict["total_orders"] > 0
    assert kpi_dict["total_net_revenue"] > 0

    # Arbitrary SQL execution
    custom_query_df = db.execute_query("SELECT COUNT(*) as cnt FROM fact_sales")
    assert custom_query_df.iloc[0]["cnt"] == len(df_clean)


def test_analytics_kpis_and_rfm(cleaned_data):
    """Test analytics module: KPIs, RFM segmentation, Pareto."""
    df_clean, _ = cleaned_data
    analytics = SalesAnalytics(df_clean)

    # Executive KPIs
    kpis = analytics.calculate_executive_kpis()
    assert kpis["total_revenue"] > 0
    assert kpis["total_orders"] > 0
    assert 0 <= kpis["repeat_customer_rate_pct"] <= 100

    # RFM
    rfm_df, segment_summary = analytics.compute_rfm_segmentation()
    assert not rfm_df.empty
    assert "Champions" in segment_summary["segment"].values or len(segment_summary) > 0

    # Pareto
    prod_pareto, pareto_summary = analytics.compute_pareto_analysis()
    assert pareto_summary["total_products"] > 0
    assert "Vital 80% Revenue Drivers" in prod_pareto["pareto_class"].values


def test_predictor_and_simulation(cleaned_data):
    """Test ML forecasting model training, recursive forecasting, and what-if simulation."""
    df_clean, _ = cleaned_data
    forecaster = SalesForecaster()
    daily_featured = forecaster.prepare_time_series_features(df_clean)
    assert len(daily_featured) > 20

    # Train and evaluate
    metrics_df, test_preds = forecaster.train_and_evaluate(daily_featured, test_days=14)
    assert not metrics_df.empty
    assert forecaster.best_model is not None

    # Future forecast
    future_df = forecaster.forecast_future(daily_featured, horizon_days=7)
    assert len(future_df) == 7
    assert "forecast_sales" in future_df.columns
    assert (future_df["upper_bound"] >= future_df["forecast_sales"]).all()
    assert (future_df["forecast_sales"] >= future_df["lower_bound"]).all()

    # What-If Simulation
    sim_res = ScenarioSimulator.simulate(
        df_clean,
        price_change_pct=5.0,
        discount_change_pct=-2.0,
        elasticity_coefficient=-1.0,
    )
    assert "simulated_revenue" in sim_res
    assert "simulated_profit" in sim_res


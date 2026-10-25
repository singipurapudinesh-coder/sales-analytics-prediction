"""
End-to-End Sales Analytics & Prediction System
CLI Pipeline Runner
Allows executing any stage of the lifecycle: data generation, ETL cleaning,
database ingestion, machine learning model training, or starting the Streamlit app.
"""

import os
import sys
import argparse
import subprocess

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.data_generator import generate_raw_sales_data
from src.cleaner import clean_sales_file
from src.database import get_default_database
from src.predictor import SalesForecaster, save_forecaster_model

DATA_DIR = os.path.join(current_dir, "data")
RAW_PATH = os.path.join(DATA_DIR, "raw", "raw_sales_data.csv")
PROC_PATH = os.path.join(DATA_DIR, "processed", "cleaned_sales_data.csv")
MODELS_DIR = os.path.join(current_dir, "models")


def step_generate():
    print("\n--- STEP 1: Generating Raw Sales Data ---")
    df = generate_raw_sales_data(
        num_records=8500,
        inject_anomalies=True,
        output_path=RAW_PATH,
    )
    print(f"Generated {len(df)} transactions with intentional anomalies at {RAW_PATH}")


def step_clean():
    print("\n--- STEP 2: Running ETL Cleaning Pipeline ---")
    if not os.path.exists(RAW_PATH):
        step_generate()
    df_clean, audit = clean_sales_file(RAW_PATH, PROC_PATH)
    print(f"Cleaned {len(df_clean)} records. Audit report:")
    for k, v in audit.items():
        print(f"  • {k}: {v}")


def step_db():
    print("\n--- STEP 3: Ingesting into SQL Database ---")
    if not os.path.exists(PROC_PATH):
        step_clean()
    import pandas as pd
    df_clean = pd.read_csv(PROC_PATH)
    db = get_default_database()
    loaded = db.load_cleaned_dataframe(df_clean)
    print(f"Database loaded successfully: {loaded}")
    kpis = db.get_kpi_summary()
    print("Database KPIs:", kpis)


def step_train():
    print("\n--- STEP 4: Training Predictive Forecasting Models ---")
    if not os.path.exists(PROC_PATH):
        step_clean()
    import pandas as pd
    df_clean = pd.read_csv(PROC_PATH)
    forecaster = SalesForecaster()
    daily_featured = forecaster.prepare_time_series_features(df_clean)
    metrics, _ = forecaster.train_and_evaluate(daily_featured, test_days=45)
    print("Model Evaluation Benchmark:")
    print(metrics.to_string(index=False))
    future_df = forecaster.forecast_future(daily_featured, horizon_days=30)
    print(f"\nGenerated 30-day forecast. Sample (first 5 days):")
    print(future_df.head().to_string(index=False))
    save_forecaster_model(forecaster, MODELS_DIR)


def step_serve():
    print("\n--- Launching Streamlit Dashboard ---")
    app_path = os.path.join(current_dir, "app.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path], check=True)


def main():
    parser = argparse.ArgumentParser(description="End-to-End Sales Analytics & Prediction System CLI")
    parser.add_argument("--generate", action="store_true", help="Generate raw sales data")
    parser.add_argument("--clean", action="store_true", help="Run ETL cleaning & feature engineering")
    parser.add_argument("--load-db", action="store_true", help="Ingest cleaned data into SQL database")
    parser.add_argument("--train", action="store_true", help="Train ML forecasting models")
    parser.add_argument("--all", action="store_true", help="Run complete end-to-end pipeline")
    parser.add_argument("--serve", action="store_true", help="Launch Streamlit web application")

    args = parser.parse_args()

    if args.generate:
        step_generate()
    elif args.clean:
        step_clean()
    elif args.load_db:
        step_db()
    elif args.train:
        step_train()
    elif args.all:
        step_generate()
        step_clean()
        step_db()
        step_train()
        print("\nAll pipeline steps completed successfully!")
    elif args.serve:
        step_serve()
    else:
        # Default behavior: run all steps
        step_generate()
        step_clean()
        step_db()
        step_train()
        print("\nPipeline setup complete. Run 'python run.py --serve' to launch the web dashboard.")


if __name__ == "__main__":
    main()


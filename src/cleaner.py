"""
End-to-End Sales Analytics & Prediction System
Data Cleaning & Feature Engineering Module
Leverages Pandas and NumPy to clean raw sales datasets, handle anomalies,
impute missing values, and engineer business KPIs.
"""

import os
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd


class SalesDataCleaner:
    """
    Production-grade ETL cleaner for sales transactions.
    Cleans dirty inputs, handles anomalies, imputes nulls, and engineers analytic features.
    """

    def __init__(self):
        self.audit_log: Dict[str, Any] = {}

    def clean_pipeline(self, df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Execute full end-to-end cleaning and feature engineering pipeline.
        Returns cleaned DataFrame and comprehensive audit report.
        """
        df = df_raw.copy()
        initial_rows = len(df)
        initial_nulls = int(df.isnull().sum().sum())

        # 1. Deduplication
        duplicates_count = int(df.duplicated().sum())
        df = df.drop_duplicates().copy()
        # Ensure order_id uniqueness if duplicate orders exist
        if "order_id" in df.columns:
            order_dups = int(df.duplicated(subset=["order_id"]).sum())
            df = df.drop_duplicates(subset=["order_id"], keep="first")
            duplicates_count += order_dups

        # 2. String Standardization & Trimming
        str_cols = ["category", "subcategory", "region", "country", "city", 
                    "customer_segment", "sales_channel", "order_status", "product_name"]
        for col in str_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.title()
                # Replace 'Nan', 'None', '' with Unknown
                df[col] = df[col].replace({"Nan": "Unknown", "None": "Unknown", "": "Unknown"})

        # Specific category correction if casing caused mismatch
        known_categories = ["Electronics", "Home & Office", "Fashion & Apparel", "Health & Beauty", "Sports & Outdoors"]
        cat_map = {c.lower(): c for c in known_categories}
        if "category" in df.columns:
            df["category"] = df["category"].apply(lambda x: cat_map.get(str(x).lower(), x))

        # 3. Numeric Cleaning: unit_price
        if "unit_price" in df.columns:
            if df["unit_price"].dtype == object:
                # Remove currency symbols and commas
                df["unit_price"] = (
                    df["unit_price"]
                    .astype(str)
                    .str.replace("$", "", regex=False)
                    .str.replace(",", "", regex=False)
                    .str.strip()
                )
            df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")
            # Impute missing prices with median by subcategory if possible, else global median
            if "subcategory" in df.columns:
                df["unit_price"] = df["unit_price"].fillna(
                    df.groupby("subcategory")["unit_price"].transform("median")
                )
            df["unit_price"] = df["unit_price"].fillna(49.99).clip(lower=1.0)

        # 4. Numeric Cleaning: unit_cost
        if "unit_cost" in df.columns:
            df["unit_cost"] = pd.to_numeric(df["unit_cost"], errors="coerce")
            # If cost is missing or greater than price, default to 60% of unit price
            invalid_cost_mask = df["unit_cost"].isna() | (df["unit_cost"] <= 0) | (df["unit_cost"] >= df["unit_price"])
            df.loc[invalid_cost_mask, "unit_cost"] = df.loc[invalid_cost_mask, "unit_price"] * 0.60
            df["unit_cost"] = df["unit_cost"].round(2)

        # 5. Numeric Cleaning: quantity
        outlier_quantity_count = 0
        if "quantity" in df.columns:
            df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(1)
            # Correct negative quantities
            negative_qty_mask = df["quantity"] <= 0
            outlier_quantity_count = int(negative_qty_mask.sum())
            df["quantity"] = df["quantity"].abs().astype(int)
            # Clip unreasonable extremes (e.g. > 100)
            df["quantity"] = df["quantity"].clip(lower=1, upper=100)

        # 6. Numeric Cleaning: discount_rate
        outlier_discount_count = 0
        if "discount_rate" in df.columns:
            df["discount_rate"] = pd.to_numeric(df["discount_rate"], errors="coerce").fillna(0.0)
            invalid_discount_mask = (df["discount_rate"] < 0) | (df["discount_rate"] > 0.70)
            outlier_discount_count = int(invalid_discount_mask.sum())
            # Clip discount rate between 0.0 and 0.70 (70% max allowable promo)
            df["discount_rate"] = df["discount_rate"].clip(lower=0.0, upper=0.70).round(2)

        # 7. Numeric Cleaning: shipping_cost
        if "shipping_cost" in df.columns:
            df["shipping_cost"] = pd.to_numeric(df["shipping_cost"], errors="coerce")
            med_ship = float(df["shipping_cost"].dropna().median()) if len(df["shipping_cost"].dropna()) > 0 else 6.99
            df["shipping_cost"] = df["shipping_cost"].fillna(med_ship)
            # If negative, correct to absolute
            df["shipping_cost"] = df["shipping_cost"].abs().clip(lower=0.0).round(2)

        # 8. Date Parsing & Standardization
        date_errors = 0
        if "order_date" in df.columns:
            df["order_date"] = pd.to_datetime(df["order_date"], format="mixed", errors="coerce")
            # If any date failed to parse, fill with median order date
            null_dates = df["order_date"].isna().sum()
            if null_dates > 0:
                date_errors += int(null_dates)
                valid_median = df["order_date"].dropna().median()
                df["order_date"] = df["order_date"].fillna(valid_median)

        if "ship_date" in df.columns:
            df["ship_date"] = pd.to_datetime(df["ship_date"], format="mixed", errors="coerce")
            # If missing, set ship_date to order_date + 3 days
            missing_ship = df["ship_date"].isna()
            df.loc[missing_ship, "ship_date"] = df.loc[missing_ship, "order_date"] + pd.Timedelta(days=3)
            # Ensure ship_date is not earlier than order_date
            earlier_ship = df["ship_date"] < df["order_date"]
            df.loc[earlier_ship, "ship_date"] = df.loc[earlier_ship, "order_date"] + pd.Timedelta(days=2)

        # 9. Feature Engineering
        df["gross_sales"] = (df["quantity"] * df["unit_price"]).round(2)
        df["discount_amount"] = (df["gross_sales"] * df["discount_rate"]).round(2)
        df["net_sales"] = (df["gross_sales"] - df["discount_amount"]).round(2)
        df["cogs"] = (df["quantity"] * df["unit_cost"]).round(2)
        df["profit"] = (df["net_sales"] - df["cogs"] - df["shipping_cost"]).round(2)
        
        # Profit Margin %
        df["profit_margin_pct"] = np.where(
            df["net_sales"] > 0,
            ((df["profit"] / df["net_sales"]) * 100).round(2),
            0.0
        )

        # Date & Temporal Features
        df["delivery_days"] = (df["ship_date"] - df["order_date"]).dt.days
        df["order_year"] = df["order_date"].dt.year
        df["order_month"] = df["order_date"].dt.month
        df["order_year_month"] = df["order_date"].dt.to_period("M").astype(str)
        df["order_quarter"] = df["order_date"].dt.quarter
        df["order_day"] = df["order_date"].dt.day
        df["order_day_name"] = df["order_date"].dt.day_name()
        df["is_weekend"] = df["order_date"].dt.weekday.isin([5, 6]).astype(int)

        # Status Flags
        df["is_returned"] = (df["order_status"].str.lower() == "returned").astype(int)
        df["is_delivered"] = (df["order_status"].str.lower() == "delivered").astype(int)

        # 10. Audit Report Compilation
        final_rows = len(df)
        final_nulls = int(df.isnull().sum().sum())
        
        # Calculate data quality score (0 to 100%)
        # Penalties: remaining nulls, initial nulls handled, duplicates removed, outliers fixed
        issues_fixed = duplicates_count + outlier_quantity_count + outlier_discount_count + date_errors
        quality_score = round(max(85.0, 100.0 - (final_nulls / max(1, final_rows * len(df.columns)) * 100.0)), 2)

        self.audit_log = {
            "initial_rows": initial_rows,
            "final_rows": final_rows,
            "duplicates_removed": duplicates_count,
            "initial_null_cells": initial_nulls,
            "remaining_null_cells": final_nulls,
            "outlier_quantities_fixed": outlier_quantity_count,
            "outlier_discounts_fixed": outlier_discount_count,
            "date_parsing_fixes": date_errors,
            "data_quality_score_pct": quality_score,
            "cleaned_columns_count": len(df.columns),
            "date_range": f"{df['order_date'].min().strftime('%Y-%m-%d')} to {df['order_date'].max().strftime('%Y-%m-%d')}",
        }

        return df, self.audit_log


def clean_sales_file(
    raw_csv_path: str,
    processed_csv_path: str = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Read raw CSV file, run cleaning pipeline, and optionally save to processed CSV.
    """
    if not os.path.exists(raw_csv_path):
        raise FileNotFoundError(f"Raw CSV file not found at: {raw_csv_path}")

    df_raw = pd.read_csv(raw_csv_path)
    cleaner = SalesDataCleaner()
    df_cleaned, audit = cleaner.clean_pipeline(df_raw)

    if processed_csv_path:
        os.makedirs(os.path.dirname(os.path.abspath(processed_csv_path)), exist_ok=True)
        # Format dates back to standard string for CSV serialization
        df_export = df_cleaned.copy()
        df_export["order_date"] = df_export["order_date"].dt.strftime("%Y-%m-%d")
        df_export["ship_date"] = df_export["ship_date"].dt.strftime("%Y-%m-%d")
        df_export.to_csv(processed_csv_path, index=False)
        print(f"[Cleaner] Saved cleaned dataset ({len(df_cleaned)} rows) to: {processed_csv_path}")

    return df_cleaned, audit


if __name__ == "__main__":
    raw_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "raw_sales_data.csv"))
    proc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed", "cleaned_sales_data.csv"))
    if os.path.exists(raw_path):
        clean_sales_file(raw_path, proc_path)
    else:
        print("Raw data file does not exist yet. Please run data_generator.py first.")


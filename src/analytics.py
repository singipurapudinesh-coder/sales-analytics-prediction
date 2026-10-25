"""
End-to-End Sales Analytics & Prediction System
Business Analytics & KPI Engine
Calculates executive KPIs, MoM/YoY growth, RFM customer segmentation,
cohort retention matrices, Pareto 80/20 product analysis, and channel attribution.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd


class SalesAnalytics:
    """
    Computes analytical metrics, customer segmentation, cohorts, and business intelligence.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(self.df["order_date"]):
            self.df["order_date"] = pd.to_datetime(self.df["order_date"])
        if "ship_date" in self.df.columns and not pd.api.types.is_datetime64_any_dtype(self.df["ship_date"]):
            self.df["ship_date"] = pd.to_datetime(self.df["ship_date"])

    def calculate_executive_kpis(self) -> Dict[str, Any]:
        """Compute high-level executive KPIs and month-over-month comparisons."""
        df = self.df
        total_revenue = float(df["net_sales"].sum())
        total_profit = float(df["profit"].sum())
        total_units = int(df["quantity"].sum())
        total_orders = int(df["order_id"].nunique())
        unique_customers = int(df["customer_id"].nunique())
        avg_order_value = float(total_revenue / max(1, total_orders))
        profit_margin_pct = float((total_profit / max(1.0, total_revenue)) * 100)
        avg_discount_pct = float(df["discount_rate"].mean() * 100)

        # Repeat customer rate
        orders_per_cust = df.groupby("customer_id")["order_id"].nunique()
        repeat_customers = int((orders_per_cust > 1).sum())
        repeat_rate_pct = float((repeat_customers / max(1, unique_customers)) * 100)

        # Return rate
        returned_orders = int((df["order_status"].str.lower() == "returned").sum())
        return_rate_pct = float((returned_orders / max(1, total_orders)) * 100)

        # Month-over-Month (MoM) Growth for recent month vs prior month
        df_monthly = (
            df.set_index("order_date")
            .resample("ME")["net_sales"]
            .sum()
            .reset_index()
        )
        if len(df_monthly) >= 2:
            current_month_sales = df_monthly.iloc[-1]["net_sales"]
            prev_month_sales = df_monthly.iloc[-2]["net_sales"]
            mom_growth_pct = float(
                ((current_month_sales - prev_month_sales) / max(1.0, prev_month_sales)) * 100
            )
        else:
            mom_growth_pct = 0.0

        return {
            "total_revenue": total_revenue,
            "total_profit": total_profit,
            "profit_margin_pct": round(profit_margin_pct, 2),
            "total_units_sold": total_units,
            "total_orders": total_orders,
            "unique_customers": unique_customers,
            "avg_order_value": round(avg_order_value, 2),
            "avg_discount_pct": round(avg_discount_pct, 2),
            "repeat_customer_rate_pct": round(repeat_rate_pct, 2),
            "return_rate_pct": round(return_rate_pct, 2),
            "mom_growth_pct": round(mom_growth_pct, 2),
        }

    def compute_rfm_segmentation(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate Recency, Frequency, and Monetary (RFM) metrics for each customer,
        assign quintile scores (1-5), and categorize into actionable behavioral segments.
        """
        df = self.df
        reference_date = df["order_date"].max() + pd.Timedelta(days=1)

        rfm = df.groupby("customer_id").agg({
            "order_date": lambda x: (reference_date - x.max()).days,
            "order_id": "nunique",
            "net_sales": "sum",
            "customer_name": "first",
            "customer_segment": "first",
            "region": "first",
        }).reset_index()

        rfm.rename(columns={
            "order_date": "recency_days",
            "order_id": "frequency",
            "net_sales": "monetary",
        }, inplace=True)

        rfm["monetary"] = rfm["monetary"].round(2)

        # Quintile scores (handling duplicate bin edges gracefully with rank)
        rfm["r_score"] = pd.qcut(rfm["recency_days"].rank(method="first"), q=5, labels=[5, 4, 3, 2, 1]).astype(int)
        rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
        rfm["m_score"] = pd.qcut(rfm["monetary"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)

        rfm["rfm_combined"] = rfm["r_score"].astype(str) + rfm["f_score"].astype(str) + rfm["m_score"].astype(str)

        # Segment Assignment Logic
        def assign_segment(row):
            r = row["r_score"]
            fm = (row["f_score"] + row["m_score"]) / 2.0

            if r >= 4 and fm >= 4:
                return "Champions"
            elif r >= 3 and fm >= 3:
                return "Loyal Customers"
            elif r >= 4 and fm < 3:
                return "Recent Customers"
            elif r >= 3 and fm < 3:
                return "Promising"
            elif r == 3 and fm >= 3:
                return "Needs Attention"
            elif r <= 2 and fm >= 4:
                return "At Risk"
            elif r <= 2 and fm >= 2:
                return "Hibernating"
            else:
                return "Lost"

        rfm["segment"] = rfm.apply(assign_segment, axis=1)

        # Summary statistics by segment
        segment_summary = rfm.groupby("segment").agg(
            customer_count=("customer_id", "count"),
            avg_recency=("recency_days", "mean"),
            avg_frequency=("frequency", "mean"),
            total_revenue=("monetary", "sum"),
            avg_revenue_per_cust=("monetary", "mean"),
        ).reset_index()

        segment_summary["revenue_share_pct"] = (
            (segment_summary["total_revenue"] / segment_summary["total_revenue"].sum()) * 100
        ).round(2)
        segment_summary["avg_recency"] = segment_summary["avg_recency"].round(1)
        segment_summary["avg_frequency"] = segment_summary["avg_frequency"].round(1)
        segment_summary["total_revenue"] = segment_summary["total_revenue"].round(2)
        segment_summary["avg_revenue_per_cust"] = segment_summary["avg_revenue_per_cust"].round(2)

        segment_summary.sort_values(by="total_revenue", ascending=False, inplace=True)

        return rfm, segment_summary

    def compute_cohort_retention(self) -> pd.DataFrame:
        """
        Perform monthly cohort retention analysis.
        Tracks percentage of customers who return to make a purchase in subsequent months.
        """
        df = self.df.copy()
        df["order_month_dt"] = df["order_date"].dt.to_period("M")
        
        # Determine each customer's cohort (month of first purchase)
        df["cohort_month"] = df.groupby("customer_id")["order_month_dt"].transform("min")

        # Calculate month difference
        def get_cohort_index(row):
            return (row["order_month_dt"].year - row["cohort_month"].year) * 12 + (
                row["order_month_dt"].month - row["cohort_month"].month
            )

        df["cohort_index"] = df.apply(get_cohort_index, axis=1)

        # Aggregate unique customers per cohort and cohort index
        cohort_data = df.groupby(["cohort_month", "cohort_index"])["customer_id"].nunique().reset_index()

        # Pivot to matrix
        cohort_pivot = cohort_data.pivot(index="cohort_month", columns="cohort_index", values="customer_id")
        cohort_size = cohort_pivot.iloc[:, 0]

        # Calculate retention percentage
        retention_matrix = cohort_pivot.divide(cohort_size, axis=0) * 100.0
        retention_matrix.index = retention_matrix.index.astype(str)

        # Limit to first 12 months for readable heatmap
        cols = [c for c in retention_matrix.columns if c <= 12]
        retention_matrix = retention_matrix[cols].round(1)

        return retention_matrix

    def compute_pareto_analysis(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Perform Pareto 80/20 analysis on product sales.
        Identifies the vital 20% of products generating 80% of total revenue.
        """
        df = self.df
        prod_sales = df.groupby(["product_id", "product_name", "category"]).agg(
            total_revenue=("net_sales", "sum"),
            units_sold=("quantity", "sum"),
            total_profit=("profit", "sum"),
        ).reset_index()

        prod_sales = prod_sales.sort_values(by="total_revenue", ascending=False).reset_index(drop=True)
        prod_sales["cumulative_revenue"] = prod_sales["total_revenue"].cumsum()
        total_rev = prod_sales["total_revenue"].sum()
        prod_sales["cumulative_revenue_pct"] = (prod_sales["cumulative_revenue"] / total_rev) * 100

        total_prods = len(prod_sales)
        prod_sales["product_rank"] = prod_sales.index + 1
        prod_sales["cumulative_product_pct"] = (prod_sales["product_rank"] / total_prods) * 100

        # Classify Pareto category
        prod_sales["pareto_class"] = np.where(
            prod_sales["cumulative_revenue_pct"] <= 80.0,
            "Vital 80% Revenue Drivers",
            "Long Tail 20% Products",
        )

        vital_count = int((prod_sales["pareto_class"] == "Vital 80% Revenue Drivers").sum())
        vital_pct = round((vital_count / max(1, total_prods)) * 100, 2)

        summary = {
            "total_products": total_prods,
            "vital_products_count": vital_count,
            "vital_products_pct": vital_pct,
            "pareto_principle_verified": vital_pct <= 35.0,  # Typically 15-30% products yield 80%
        }

        return prod_sales, summary

    def compute_channel_attribution(self) -> pd.DataFrame:
        """Evaluate sales channels across volume, revenue, profitability, and return rates."""
        df = self.df
        channel_df = df.groupby("sales_channel").agg(
            order_count=("order_id", "count"),
            units_sold=("quantity", "sum"),
            total_revenue=("net_sales", "sum"),
            total_profit=("profit", "sum"),
            avg_discount=("discount_rate", "mean"),
            return_count=("is_returned", "sum"),
        ).reset_index()

        channel_df["profit_margin_pct"] = (
            (channel_df["total_profit"] / channel_df["total_revenue"]) * 100
        ).round(2)
        channel_df["return_rate_pct"] = (
            (channel_df["return_count"] / channel_df["order_count"]) * 100
        ).round(2)
        channel_df["avg_discount_pct"] = (channel_df["avg_discount"] * 100).round(2)
        channel_df["revenue_share_pct"] = (
            (channel_df["total_revenue"] / channel_df["total_revenue"].sum()) * 100
        ).round(2)

        channel_df.sort_values(by="total_revenue", ascending=False, inplace=True)
        return channel_df


if __name__ == "__main__":
    import os
    proc_csv = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed", "cleaned_sales_data.csv"))
    if os.path.exists(proc_csv):
        df = pd.read_csv(proc_csv)
        analytics = SalesAnalytics(df)
        kpis = analytics.calculate_executive_kpis()
        print("Executive KPIs:", kpis)
    else:
        print("Cleaned data file not found. Run cleaner.py first.")


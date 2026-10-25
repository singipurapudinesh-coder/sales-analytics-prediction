"""
End-to-End Sales Analytics & Prediction System
Database Management Module
Implements SQLAlchemy engine for SQLite (zero-config local) & MySQL (remote enterprise).
Manages Star-Schema relational tables, data ingestion, and analytical SQL queries.
"""

import os
from typing import Dict, Any, List, Optional
import pandas as pd
from sqlalchemy import (
    create_engine,
    Column,
    String,
    Float,
    Integer,
    Date,
    MetaData,
    Table,
    text,
    inspect,
)
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class ProductDim(Base):
    __tablename__ = "dim_products"
    product_id = Column(String(50), primary_key=True)
    product_name = Column(String(255))
    category = Column(String(100))
    subcategory = Column(String(100))
    unit_cost = Column(Float)
    unit_price = Column(Float)


class CustomerDim(Base):
    __tablename__ = "dim_customers"
    customer_id = Column(String(50), primary_key=True)
    customer_name = Column(String(255))
    customer_segment = Column(String(100))
    region = Column(String(100))
    country = Column(String(100))
    city = Column(String(100))


class SalesFact(Base):
    __tablename__ = "fact_sales"
    order_id = Column(String(50), primary_key=True)
    order_date = Column(Date, index=True)
    ship_date = Column(Date)
    customer_id = Column(String(50), index=True)
    product_id = Column(String(50), index=True)
    sales_channel = Column(String(100))
    quantity = Column(Integer)
    unit_price = Column(Float)
    unit_cost = Column(Float)
    discount_rate = Column(Float)
    gross_sales = Column(Float)
    discount_amount = Column(Float)
    net_sales = Column(Float)
    cogs = Column(Float)
    profit = Column(Float)
    profit_margin_pct = Column(Float)
    shipping_cost = Column(Float)
    order_status = Column(String(50))
    delivery_days = Column(Integer)


class SalesDatabase:
    """
    SQL Database interface supporting SQLite and MySQL.
    """

    def __init__(self, db_uri: Optional[str] = None):
        if not db_uri:
            # Default to local SQLite database in data/database/sales_analytics.db
            db_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "data", "database")
            )
            os.makedirs(db_dir, exist_ok=True)
            db_path = os.path.join(db_dir, "sales_analytics.db")
            self.db_uri = f"sqlite:///{db_path}"
        else:
            self.db_uri = db_uri

        self.engine = create_engine(self.db_uri, echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)
        self.create_tables()

    def create_tables(self):
        """Create tables in the database if they don't exist."""
        Base.metadata.create_all(self.engine)

    def load_cleaned_dataframe(self, df_clean: pd.DataFrame) -> Dict[str, int]:
        """
        Normalize and insert cleaned sales dataframe into Star Schema tables:
        dim_products, dim_customers, fact_sales.
        """
        df = df_clean.copy()

        # Format datetime columns to Date objects or strings
        if not pd.api.types.is_datetime64_any_dtype(df["order_date"]):
            df["order_date"] = pd.to_datetime(df["order_date"])
        if not pd.api.types.is_datetime64_any_dtype(df["ship_date"]):
            df["ship_date"] = pd.to_datetime(df["ship_date"])

        # 1. Populate dim_products
        product_cols = ["product_id", "product_name", "category", "subcategory", "unit_cost", "unit_price"]
        df_prod = (
            df[product_cols]
            .drop_duplicates(subset=["product_id"])
            .copy()
        )

        # 2. Populate dim_customers
        customer_cols = ["customer_id", "customer_name", "customer_segment", "region", "country", "city"]
        df_cust = (
            df[customer_cols]
            .drop_duplicates(subset=["customer_id"])
            .copy()
        )

        # 3. Populate fact_sales
        fact_cols = [
            "order_id", "order_date", "ship_date", "customer_id", "product_id",
            "sales_channel", "quantity", "unit_price", "unit_cost", "discount_rate",
            "gross_sales", "discount_amount", "net_sales", "cogs", "profit",
            "profit_margin_pct", "shipping_cost", "order_status", "delivery_days"
        ]
        df_fact = df[fact_cols].drop_duplicates(subset=["order_id"]).copy()

        # Use connection transaction for clean reload
        with self.engine.begin() as conn:
            # Upsert / Replace tables
            df_prod.to_sql("dim_products", con=conn, if_exists="replace", index=False)
            df_cust.to_sql("dim_customers", con=conn, if_exists="replace", index=False)
            df_fact.to_sql("fact_sales", con=conn, if_exists="replace", index=False)

            # Create an aggregated view/summary table for lightning-fast queries
            summary_sql = """
            CREATE TABLE IF NOT EXISTS sales_summary_daily AS
            SELECT 
                f.order_date,
                p.category,
                c.region,
                c.customer_segment,
                f.sales_channel,
                COUNT(f.order_id) as total_orders,
                SUM(f.quantity) as total_units_sold,
                SUM(f.net_sales) as total_net_sales,
                SUM(f.profit) as total_profit,
                AVG(f.profit_margin_pct) as avg_profit_margin,
                SUM(f.discount_amount) as total_discounts
            FROM fact_sales f
            LEFT JOIN dim_products p ON f.product_id = p.product_id
            LEFT JOIN dim_customers c ON f.customer_id = c.customer_id
            GROUP BY f.order_date, p.category, c.region, c.customer_segment, f.sales_channel;
            """
            conn.execute(text("DROP TABLE IF EXISTS sales_summary_daily;"))
            conn.execute(text(summary_sql))

        return {
            "products_loaded": len(df_prod),
            "customers_loaded": len(df_cust),
            "sales_loaded": len(df_fact),
        }

    def execute_query(self, sql_query: str, params: Optional[dict] = None) -> pd.DataFrame:
        """
        Execute arbitrary SQL query and return results as a Pandas DataFrame.
        """
        with self.engine.connect() as conn:
            result = pd.read_sql_query(text(sql_query), con=conn, params=params or {})
        return result

    def get_table_metadata(self) -> Dict[str, Any]:
        """
        Inspect database schema and return table names, column details, and row counts.
        """
        inspector = inspect(self.engine)
        table_names = inspector.get_table_names()
        metadata = {}

        for table in table_names:
            columns = inspector.get_columns(table)
            col_info = [{"name": c["name"], "type": str(c["type"])} for c in columns]
            
            # Row count
            try:
                row_count = self.execute_query(f"SELECT COUNT(*) as count FROM {table}").iloc[0]["count"]
            except Exception:
                row_count = 0

            metadata[table] = {
                "columns": col_info,
                "row_count": int(row_count),
            }

        return metadata

    # -------------------------------------------------------------
    # High-Performance Analytical SQL Queries
    # -------------------------------------------------------------
    def get_kpi_summary(self) -> Dict[str, float]:
        """Fetch primary business KPIs using SQL aggregation."""
        query = """
        SELECT 
            COUNT(order_id) as total_orders,
            COUNT(DISTINCT customer_id) as total_customers,
            COALESCE(SUM(quantity), 0) as total_units_sold,
            COALESCE(SUM(gross_sales), 0) as total_gross_sales,
            COALESCE(SUM(net_sales), 0) as total_net_revenue,
            COALESCE(SUM(profit), 0) as total_net_profit,
            COALESCE(AVG(net_sales), 0) as avg_order_value,
            COALESCE(AVG(profit_margin_pct), 0) as avg_profit_margin,
            COALESCE(AVG(discount_rate) * 100, 0) as avg_discount_rate_pct,
            COALESCE(SUM(CASE WHEN LOWER(order_status) = 'returned' THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(order_id), 0), 0) as return_rate_pct
        FROM fact_sales;
        """
        df = self.execute_query(query)
        if df.empty:
            return {}
        return df.iloc[0].to_dict()

    def get_monthly_sales_trend(self) -> pd.DataFrame:
        """Fetch monthly sales, profit, and order volumes."""
        query = """
        SELECT 
            SUBSTR(order_date, 1, 7) as year_month,
            COUNT(order_id) as orders_count,
            SUM(net_sales) as total_revenue,
            SUM(profit) as total_profit,
            AVG(profit_margin_pct) as avg_margin,
            SUM(quantity) as total_units
        FROM fact_sales
        GROUP BY SUBSTR(order_date, 1, 7)
        ORDER BY year_month ASC;
        """
        return self.execute_query(query)

    def get_category_performance(self) -> pd.DataFrame:
        """Category-level revenue, profit, and units sold."""
        query = """
        SELECT 
            p.category,
            COUNT(f.order_id) as order_count,
            SUM(f.quantity) as units_sold,
            ROUND(SUM(f.net_sales), 2) as total_revenue,
            ROUND(SUM(f.profit), 2) as total_profit,
            ROUND(AVG(f.profit_margin_pct), 2) as avg_margin_pct
        FROM fact_sales f
        JOIN dim_products p ON f.product_id = p.product_id
        GROUP BY p.category
        ORDER BY total_revenue DESC;
        """
        return self.execute_query(query)

    def get_regional_performance(self) -> pd.DataFrame:
        """Regional revenue, profit, and average delivery time."""
        query = """
        SELECT 
            c.region,
            COUNT(f.order_id) as order_count,
            ROUND(SUM(f.net_sales), 2) as total_revenue,
            ROUND(SUM(f.profit), 2) as total_profit,
            ROUND(AVG(f.delivery_days), 1) as avg_delivery_days,
            ROUND(AVG(f.profit_margin_pct), 2) as avg_margin_pct
        FROM fact_sales f
        JOIN dim_customers c ON f.customer_id = c.customer_id
        GROUP BY c.region
        ORDER BY total_revenue DESC;
        """
        return self.execute_query(query)

    def get_top_products(self, limit: int = 10) -> pd.DataFrame:
        """Top performing products by revenue."""
        query = f"""
        SELECT 
            p.product_id,
            p.product_name,
            p.category,
            SUM(f.quantity) as total_units_sold,
            ROUND(SUM(f.net_sales), 2) as total_revenue,
            ROUND(SUM(f.profit), 2) as total_profit
        FROM fact_sales f
        JOIN dim_products p ON f.product_id = p.product_id
        GROUP BY p.product_id, p.product_name, p.category
        ORDER BY total_revenue DESC
        LIMIT {limit};
        """
        return self.execute_query(query)


def get_default_database() -> SalesDatabase:
    """Helper to obtain default initialized SalesDatabase instance."""
    return SalesDatabase()


if __name__ == "__main__":
    db = get_default_database()
    print("Database initialized successfully at:", db.db_uri)
    meta = db.get_table_metadata()
    print("Current tables:", list(meta.keys()))


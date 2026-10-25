"""
End-to-End Sales Analytics & Prediction System
Data Generator Module: Creates realistic enterprise sales transactions with intentional dirty data
for testing and demonstrating the ETL & Cleaning pipeline.
"""

import os
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


CATEGORIES = {
    "Electronics": {
        "subcategories": {
            "Smartphones": {"price_range": (399, 1199), "cost_margin": 0.35},
            "Laptops & PCs": {"price_range": (699, 2199), "cost_margin": 0.28},
            "Audio & Headphones": {"price_range": (49, 349), "cost_margin": 0.45},
            "Accessories": {"price_range": (15, 89), "cost_margin": 0.55},
            "Smart Home": {"price_range": (39, 299), "cost_margin": 0.38},
        },
        "seasonality_q4_boost": 1.45,
    },
    "Home & Office": {
        "subcategories": {
            "Ergonomic Chairs": {"price_range": (149, 599), "cost_margin": 0.40},
            "Standing Desks": {"price_range": (249, 799), "cost_margin": 0.35},
            "Desk Organization": {"price_range": (19, 79), "cost_margin": 0.50},
            "Printers & Scanners": {"price_range": (89, 399), "cost_margin": 0.25},
            "Lighting & Lamps": {"price_range": (29, 149), "cost_margin": 0.45},
        },
        "seasonality_q4_boost": 1.15,
    },
    "Fashion & Apparel": {
        "subcategories": {
            "Men's Casual Wear": {"price_range": (25, 120), "cost_margin": 0.55},
            "Women's Footwear": {"price_range": (45, 180), "cost_margin": 0.52},
            "Outerwear & Jackets": {"price_range": (79, 320), "cost_margin": 0.48},
            "Athletic Wear": {"price_range": (30, 110), "cost_margin": 0.50},
            "Bags & Luggage": {"price_range": (40, 260), "cost_margin": 0.45},
        },
        "seasonality_q4_boost": 1.35,
    },
    "Health & Beauty": {
        "subcategories": {
            "Skincare Essentials": {"price_range": (18, 95), "cost_margin": 0.60},
            "Hair Care & Styling": {"price_range": (14, 65), "cost_margin": 0.58},
            "Wellness & Vitamins": {"price_range": (20, 80), "cost_margin": 0.50},
            "Personal Grooming": {"price_range": (25, 140), "cost_margin": 0.42},
            "Organic Cosmetics": {"price_range": (22, 110), "cost_margin": 0.55},
        },
        "seasonality_q4_boost": 1.25,
    },
    "Sports & Outdoors": {
        "subcategories": {
            "Fitness Equipment": {"price_range": (35, 450), "cost_margin": 0.38},
            "Camping & Hiking": {"price_range": (29, 350), "cost_margin": 0.42},
            "Cycling & Scooters": {"price_range": (120, 850), "cost_margin": 0.32},
            "Water Sports": {"price_range": (45, 290), "cost_margin": 0.40},
            "Outdoor Games": {"price_range": (20, 110), "cost_margin": 0.48},
        },
        "seasonality_q4_boost": 1.10,
    },
}

REGIONS = {
    "North America": {
        "countries": ["United States", "Canada"],
        "cities": ["New York", "San Francisco", "Chicago", "Toronto", "Austin", "Seattle", "Vancouver"],
    },
    "Europe": {
        "countries": ["United Kingdom", "Germany", "France", "Netherlands", "Spain"],
        "cities": ["London", "Berlin", "Paris", "Amsterdam", "Madrid", "Munich", "Dublin"],
    },
    "Asia-Pacific": {
        "countries": ["Japan", "Singapore", "Australia", "India", "South Korea"],
        "cities": ["Tokyo", "Singapore", "Sydney", "Mumbai", "Seoul", "Melbourne", "Bangalore"],
    },
    "Latin America": {
        "countries": ["Brazil", "Mexico", "Chile", "Colombia"],
        "cities": ["Sao Paulo", "Mexico City", "Santiago", "Bogota", "Rio de Janeiro"],
    },
}

CUSTOMER_SEGMENTS = ["Consumer", "Corporate", "Small Business", "Enterprise"]
SALES_CHANNELS = ["Online Store", "Mobile App", "Marketplace", "B2B Portal", "Direct Sales"]
ORDER_STATUSES = ["Delivered", "Shipped", "Processing", "Cancelled", "Returned"]
ORDER_STATUS_PROBS = [0.84, 0.06, 0.04, 0.03, 0.03]


def generate_raw_sales_data(
    num_records: int = 8000,
    start_date: str = "2023-01-01",
    end_date: str = "2025-12-31",
    inject_anomalies: bool = True,
    random_state: int = 42,
    output_path: str = None,
) -> pd.DataFrame:
    """
    Generate synthetic raw sales transaction data with realistic business patterns,
    seasonality, customer cohorts, and optional dirty data anomalies.
    """
    random.seed(random_state)
    np.random.seed(random_state)

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    total_days = (end_dt - start_dt).days

    # Customer pool (realistic repeat purchase behavior)
    num_customers = max(200, int(num_records * 0.18))
    customer_ids = [f"CUST-{1000 + i}" for i in range(num_customers)]
    customer_names = [f"Customer {1000 + i}" for i in range(num_customers)]
    customer_profiles = {}
    for cid, cname in zip(customer_ids, customer_names):
        region = random.choice(list(REGIONS.keys()))
        country = random.choice(REGIONS[region]["countries"])
        city = random.choice(REGIONS[region]["cities"])
        segment = random.choices(CUSTOMER_SEGMENTS, weights=[0.55, 0.25, 0.15, 0.05])[0]
        customer_profiles[cid] = {
            "name": cname,
            "region": region,
            "country": country,
            "city": city,
            "segment": segment,
        }

    records = []

    for i in range(num_records):
        order_id = f"ORD-{20230000 + i}"
        
        # Pick date with seasonal weighting (higher weight towards Q4 and weekends)
        day_offset = random.randint(0, total_days)
        order_date = start_dt + timedelta(days=day_offset)
        
        # Seasonality factors
        month = order_date.month
        is_q4 = month in [10, 11, 12]
        is_weekend = order_date.weekday() >= 5
        
        # Select category
        category = random.choice(list(CATEGORIES.keys()))
        cat_info = CATEGORIES[category]
        
        # Select subcategory
        subcat = random.choice(list(cat_info["subcategories"].keys()))
        subcat_info = cat_info["subcategories"][subcat]
        
        # Price and cost
        p_min, p_max = subcat_info["price_range"]
        unit_price = round(random.uniform(p_min, p_max), 2)
        base_margin = subcat_info["cost_margin"]
        unit_cost = round(unit_price * (1 - base_margin + random.uniform(-0.04, 0.04)), 2)

        # Quantity depends on segment
        customer_id = random.choice(customer_ids)
        c_profile = customer_profiles[customer_id]
        segment = c_profile["segment"]
        
        if segment == "Enterprise":
            qty = random.randint(5, 40)
        elif segment == "Corporate":
            qty = random.randint(2, 15)
        elif segment == "Small Business":
            qty = random.randint(1, 8)
        else:
            qty = random.choices([1, 2, 3, 4, 5], weights=[0.55, 0.25, 0.12, 0.05, 0.03])[0]

        # Discount
        if is_q4:
            discount = random.choices([0.0, 0.10, 0.15, 0.20, 0.25, 0.30], weights=[0.3, 0.2, 0.2, 0.15, 0.1, 0.05])[0]
        else:
            discount = random.choices([0.0, 0.05, 0.10, 0.15, 0.20], weights=[0.6, 0.18, 0.12, 0.07, 0.03])[0]

        # Shipping details
        shipping_days = random.randint(1, 7)
        ship_date = order_date + timedelta(days=shipping_days)
        base_shipping = 4.99 + (qty * 1.50)
        shipping_cost = round(base_shipping + random.uniform(0, 5), 2)

        # Channel & status
        channel = random.choices(SALES_CHANNELS, weights=[0.42, 0.26, 0.16, 0.10, 0.06])[0]
        order_status = random.choices(ORDER_STATUSES, weights=ORDER_STATUS_PROBS)[0]

        product_name = f"{subcat} Model {random.choice(['Pro', 'Plus', 'Ultra', 'Standard', 'Max', 'Lite'])}"
        product_id = f"PRD-{abs(hash(f'{category}_{subcat}_{product_name}')) % 9000 + 1000}"

        records.append({
            "order_id": order_id,
            "order_date": order_date.strftime("%Y-%m-%d"),
            "ship_date": ship_date.strftime("%Y-%m-%d"),
            "customer_id": customer_id,
            "customer_name": c_profile["name"],
            "customer_segment": segment,
            "region": c_profile["region"],
            "country": c_profile["country"],
            "city": c_profile["city"],
            "product_id": product_id,
            "product_name": product_name,
            "category": category,
            "subcategory": subcat,
            "sales_channel": channel,
            "unit_cost": unit_cost,
            "unit_price": unit_price,
            "quantity": qty,
            "discount_rate": discount,
            "shipping_cost": shipping_cost,
            "order_status": order_status,
        })

    df = pd.DataFrame(records)

    # -------------------------------------------------------------
    # Inject Realistic Dirty Data & Anomalies if enabled
    # -------------------------------------------------------------
    if inject_anomalies:
        n = len(df)
        
        # 1. Missing Values (NaNs)
        missing_discount_idx = np.random.choice(n, size=int(n * 0.04), replace=False)
        df.loc[missing_discount_idx, "discount_rate"] = np.nan
        
        missing_region_idx = np.random.choice(n, size=int(n * 0.02), replace=False)
        df.loc[missing_region_idx, "region"] = np.nan
        
        missing_shipcost_idx = np.random.choice(n, size=int(n * 0.03), replace=False)
        df.loc[missing_shipcost_idx, "shipping_cost"] = np.nan

        # 2. Inconsistent String Casing and Whitespace
        messy_str_idx = np.random.choice(n, size=int(n * 0.06), replace=False)
        for idx in messy_str_idx:
            curr_cat = str(df.loc[idx, "category"])
            choice = random.choice(["upper", "lower", "spaces"])
            if choice == "upper":
                df.loc[idx, "category"] = curr_cat.upper()
            elif choice == "lower":
                df.loc[idx, "category"] = curr_cat.lower()
            else:
                df.loc[idx, "category"] = f"  {curr_cat}  "

        # 3. Currency Symbols in unit_price for some records (e.g. '$ 1,299.00')
        df["unit_price"] = df["unit_price"].astype(object)
        curr_price_idx = np.random.choice(n, size=int(n * 0.05), replace=False)
        for idx in curr_price_idx:
            val = df.loc[idx, "unit_price"]
            df.loc[idx, "unit_price"] = f"${val:,.2f}"

        # 4. Inconsistent Date Formats
        df["order_date"] = df["order_date"].astype(object)
        date_messy_idx = np.random.choice(n, size=int(n * 0.05), replace=False)
        for idx in date_messy_idx:
            raw_d = df.loc[idx, "order_date"]
            dt = datetime.strptime(str(raw_d), "%Y-%m-%d")
            format_choice = random.choice(["slash", "written"])
            if format_choice == "slash":
                df.loc[idx, "order_date"] = dt.strftime("%d/%m/%Y")
            else:
                df.loc[idx, "order_date"] = dt.strftime("%b %d, %Y")

        # 5. Outliers and Erroneous Values
        outlier_idx = np.random.choice(n, size=int(n * 0.015), replace=False)
        for i, idx in enumerate(outlier_idx):
            if i % 3 == 0:
                # Negative quantity anomaly
                df.loc[idx, "quantity"] = -random.randint(1, 5)
            elif i % 3 == 1:
                # Extreme discount > 100%
                df.loc[idx, "discount_rate"] = random.choice([1.25, 1.50, -0.20])
            else:
                # Impossible negative shipping
                df.loc[idx, "shipping_cost"] = -15.0

        # 6. Duplicate Rows
        duplicate_samples = df.sample(n=int(n * 0.02), random_state=random_state)
        df = pd.concat([df, duplicate_samples], ignore_index=True)

    # Shuffle to ensure realistic disorder
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[DataGenerator] Successfully generated {len(df)} raw records saved to: {output_path}")

    return df


if __name__ == "__main__":
    default_dest = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "raw_sales_data.csv"))
    generate_raw_sales_data(num_records=8500, output_path=default_dest)

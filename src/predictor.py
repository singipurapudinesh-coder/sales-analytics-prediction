"""
End-to-End Sales Analytics & Prediction System
Machine Learning & Predictive Sales Modeling Module
Implements Time-Series Sales Forecasting, Model Benchmarking (Ridge, Random Forest, Gradient Boosting),
Residual Diagnostics, Recursive Multi-step Future Forecasting, and What-If Scenario Simulation.
"""

import os
from typing import Dict, Any, Tuple, List, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class SalesForecaster:
    """
    Time-Series Sales Forecasting engine with automated feature engineering,
    multi-model benchmarking, recursive future forecasting, and confidence intervals.
    """

    def __init__(self, freq: str = "D"):
        self.freq = freq
        self.models: Dict[str, Any] = {}
        self.best_model_name: Optional[str] = None
        self.best_model: Optional[Any] = None
        self.feature_names: List[str] = []
        self.metrics_comparison: pd.DataFrame = pd.DataFrame()
        self.daily_sales_series: Optional[pd.DataFrame] = None
        self.residual_std: float = 0.0

    def prepare_time_series_features(self, df_clean: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate sales into uniform daily series and construct lag, rolling, and calendar features.
        """
        df = df_clean.copy()
        if not pd.api.types.is_datetime64_any_dtype(df["order_date"]):
            df["order_date"] = pd.to_datetime(df["order_date"])

        # Aggregate daily net sales
        daily = (
            df.groupby("order_date")
            .agg(
                sales=("net_sales", "sum"),
                orders=("order_id", "count"),
                units=("quantity", "sum"),
                avg_discount=("discount_rate", "mean"),
            )
            .reset_index()
        )

        daily = daily.sort_values(by="order_date").set_index("order_date")

        # Fill potential missing dates with 0 sales to maintain contiguous time series
        full_date_idx = pd.date_range(start=daily.index.min(), end=daily.index.max(), freq="D")
        daily = daily.reindex(full_date_idx).fillna({
            "sales": 0.0,
            "orders": 0,
            "units": 0,
            "avg_discount": 0.0,
        })
        daily.index.name = "date"
        daily = daily.reset_index()

        self.daily_sales_series = daily.copy()

        # Feature Engineering: Lags
        for lag in [1, 2, 7, 14, 21, 28, 30]:
            daily[f"lag_{lag}"] = daily["sales"].shift(lag)

        # Feature Engineering: Rolling Statistics
        for window in [7, 14, 30]:
            daily[f"rolling_mean_{window}"] = daily["sales"].shift(1).rolling(window=window).mean()
            daily[f"rolling_std_{window}"] = daily["sales"].shift(1).rolling(window=window).std().fillna(0)

        # Feature Engineering: Calendar & Seasonality
        daily["day_of_week"] = daily["date"].dt.dayofweek
        daily["day_of_month"] = daily["date"].dt.day
        daily["day_of_year"] = daily["date"].dt.dayofyear
        daily["month"] = daily["date"].dt.month
        daily["quarter"] = daily["date"].dt.quarter
        daily["is_weekend"] = daily["day_of_week"].isin([5, 6]).astype(int)
        daily["is_month_end"] = daily["date"].dt.is_month_end.astype(int)
        
        # Trend index
        daily["time_step"] = np.arange(len(daily))

        # Drop rows where lag features are NaN (first 30 days)
        daily_featured = daily.dropna().reset_index(drop=True)
        return daily_featured

    def train_and_evaluate(
        self,
        daily_featured: pd.DataFrame,
        test_days: int = 45,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Train Ridge, Random Forest, and HistGradientBoosting models on chronological train/test split.
        Returns metrics comparison table and test set predictions.
        """
        feature_cols = [
            c for c in daily_featured.columns
            if c not in ["date", "sales", "orders", "units", "avg_discount"]
        ]
        self.feature_names = feature_cols

        X = daily_featured[feature_cols]
        y = daily_featured["sales"]

        # Chronological train/test split
        split_idx = len(daily_featured) - test_days
        if split_idx <= 60:
            test_days = max(14, int(len(daily_featured) * 0.15))
            split_idx = len(daily_featured) - test_days

        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        test_dates = daily_featured.iloc[split_idx:]["date"].values

        candidate_models = {
            "Hist Gradient Boosting": HistGradientBoostingRegressor(
                max_iter=150, min_samples_leaf=15, random_state=42
            ),
            "Random Forest": RandomForestRegressor(
                n_estimators=100, max_depth=12, random_state=42, n_jobs=-1
            ),
            "Ridge Regression": Ridge(alpha=10.0),
        }

        results = []
        predictions_dict = {"date": test_dates, "actual_sales": y_test.values}

        for name, model in candidate_models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            preds = np.clip(preds, 0, None)  # Sales cannot be negative
            
            predictions_dict[f"pred_{name}"] = preds

            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
            mae = float(mean_absolute_error(y_test, preds))
            r2 = float(r2_score(y_test, preds))
            # Safe MAPE calculation
            non_zero_mask = y_test > 0
            if non_zero_mask.sum() > 0:
                mape = float(np.mean(np.abs((y_test[non_zero_mask] - preds[non_zero_mask]) / y_test[non_zero_mask])) * 100)
            else:
                mape = 0.0

            results.append({
                "Model": name,
                "RMSE ($)": round(rmse, 2),
                "MAE ($)": round(mae, 2),
                "R² Score": round(r2, 4),
                "MAPE (%)": round(mape, 2),
            })
            self.models[name] = model

        metrics_df = pd.DataFrame(results).sort_values(by="RMSE ($)", ascending=True).reset_index(drop=True)
        self.metrics_comparison = metrics_df

        # Select best model
        self.best_model_name = metrics_df.iloc[0]["Model"]
        self.best_model = self.models[self.best_model_name]

        # Calculate residual standard deviation of best model for confidence bounds
        best_preds = predictions_dict[f"pred_{self.best_model_name}"]
        residuals = y_test.values - best_preds
        self.residual_std = float(np.std(residuals))

        test_preds_df = pd.DataFrame(predictions_dict)
        return metrics_df, test_preds_df

    def forecast_future(
        self,
        daily_featured: pd.DataFrame,
        horizon_days: int = 30,
    ) -> pd.DataFrame:
        """
        Perform recursive multi-step future forecasting for the next N days
        with upper and lower confidence intervals.
        """
        if self.best_model is None:
            raise ValueError("Model has not been trained yet. Call train_and_evaluate first.")

        history_df = daily_featured.copy()
        last_date = history_df["date"].max()
        future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, horizon_days + 1)]

        # Maintain rolling series for recursive updates
        sales_buffer = list(history_df["sales"].values)
        time_step_counter = history_df["time_step"].max()

        forecast_records = []

        for f_date in future_dates:
            time_step_counter += 1
            
            # Construct feature vector for today
            row_features = {}
            for lag in [1, 2, 7, 14, 21, 28, 30]:
                row_features[f"lag_{lag}"] = sales_buffer[-lag] if len(sales_buffer) >= lag else sales_buffer[-1]

            for window in [7, 14, 30]:
                window_slice = sales_buffer[-window:]
                row_features[f"rolling_mean_{window}"] = float(np.mean(window_slice))
                row_features[f"rolling_std_{window}"] = float(np.std(window_slice))

            row_features["day_of_week"] = f_date.dayofweek
            row_features["day_of_month"] = f_date.day
            row_features["day_of_year"] = f_date.dayofyear
            row_features["month"] = f_date.month
            row_features["quarter"] = f_date.quarter
            row_features["is_weekend"] = int(f_date.dayofweek in [5, 6])
            row_features["is_month_end"] = int(f_date.is_month_end)
            row_features["time_step"] = time_step_counter

            # Convert to DataFrame matching training feature columns
            X_curr = pd.DataFrame([row_features])[self.feature_names]
            
            # Predict sales
            predicted_val = float(np.clip(self.best_model.predict(X_curr)[0], 0, None))
            
            # Append prediction to buffer for next iterations
            sales_buffer.append(predicted_val)

            # Increasing uncertainty cone over future horizon
            day_idx = len(forecast_records) + 1
            uncertainty_multiplier = 1.0 + (0.015 * day_idx)
            margin_of_error = 1.96 * self.residual_std * uncertainty_multiplier

            forecast_records.append({
                "date": f_date,
                "forecast_sales": round(predicted_val, 2),
                "lower_bound": round(max(0.0, predicted_val - margin_of_error), 2),
                "upper_bound": round(predicted_val + margin_of_error, 2),
            })

        return pd.DataFrame(forecast_records)

    def get_feature_importances(self) -> pd.DataFrame:
        """Extract feature importances from the best model."""
        if self.best_model is None:
            return pd.DataFrame()

        if hasattr(self.best_model, "feature_importances_"):
            importances = self.best_model.feature_importances_
        elif hasattr(self.best_model, "coef_"):
            importances = np.abs(self.best_model.coef_)
        else:
            return pd.DataFrame()

        imp_df = pd.DataFrame({
            "Feature": self.feature_names,
            "Importance": importances,
        }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

        # Normalize to percentage
        imp_df["Importance (%)"] = (imp_df["Importance"] / imp_df["Importance"].sum() * 100).round(2)
        return imp_df


class ScenarioSimulator:
    """
    Interactive 'What-If' Sales & Profit Scenario Simulator.
    Simulates price elasticity, promotional discounts, and seasonal campaign uplifts.
    """

    @staticmethod
    def simulate(
        baseline_df: pd.DataFrame,
        price_change_pct: float = 0.0,
        discount_change_pct: float = 0.0,
        elasticity_coefficient: float = -1.2,
        marketing_uplift_pct: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Simulate impact on Revenue, Units, and Profit:
        - Price Elasticity of Demand: % Change in Quantity = Elasticity * % Change in Price
        - Net Sales = Quantity * New Price * (1 - New Discount)
        - Net Profit = Net Sales - COGS - Shipping
        """
        df = baseline_df.copy()

        # Baseline sums
        base_revenue = float(df["net_sales"].sum())
        base_profit = float(df["profit"].sum())
        base_units = int(df["quantity"].sum())
        base_margin = float((base_profit / max(1.0, base_revenue)) * 100)

        # Elasticity effect on demand
        quantity_price_effect = 1.0 + (elasticity_coefficient * (price_change_pct / 100.0))
        marketing_effect = 1.0 + (marketing_uplift_pct / 100.0)
        overall_quantity_multiplier = max(0.2, quantity_price_effect * marketing_effect)

        sim_units = int(round(base_units * overall_quantity_multiplier))

        # Adjusted price & discount
        price_multiplier = 1.0 + (price_change_pct / 100.0)
        sim_unit_price = df["unit_price"] * price_multiplier
        sim_discount_rate = np.clip(df["discount_rate"] + (discount_change_pct / 100.0), 0.0, 0.70)

        # Recompute simulated rows
        sim_qty = np.maximum(1, np.round(df["quantity"] * overall_quantity_multiplier))
        sim_gross = sim_qty * sim_unit_price
        sim_net = sim_gross * (1 - sim_discount_rate)
        sim_cogs = sim_qty * df["unit_cost"]
        sim_profit = sim_net - sim_cogs - df["shipping_cost"]

        sim_revenue = float(sim_net.sum())
        sim_total_profit = float(sim_profit.sum())
        sim_margin = float((sim_total_profit / max(1.0, sim_revenue)) * 100)

        rev_delta_pct = ((sim_revenue - base_revenue) / max(1.0, base_revenue)) * 100
        profit_delta_pct = ((sim_total_profit - base_profit) / max(1.0, abs(base_profit))) * 100

        return {
            "baseline_revenue": round(base_revenue, 2),
            "simulated_revenue": round(sim_revenue, 2),
            "revenue_delta_pct": round(rev_delta_pct, 2),
            "baseline_profit": round(base_profit, 2),
            "simulated_profit": round(sim_total_profit, 2),
            "profit_delta_pct": round(profit_delta_pct, 2),
            "baseline_units": base_units,
            "simulated_units": sim_units,
            "baseline_margin_pct": round(base_margin, 2),
            "simulated_margin_pct": round(sim_margin, 2),
        }


def save_forecaster_model(forecaster: SalesForecaster, model_dir: str):
    """Save trained forecaster object and artifacts to disk."""
    os.makedirs(model_dir, exist_ok=True)
    joblib_path = os.path.join(model_dir, "sales_forecaster.joblib")
    joblib.dump(forecaster, joblib_path)
    print(f"[Predictor] Model saved to: {joblib_path}")


def load_forecaster_model(model_dir: str) -> Optional[SalesForecaster]:
    """Load pre-trained forecaster model from disk."""
    joblib_path = os.path.join(model_dir, "sales_forecaster.joblib")
    if os.path.exists(joblib_path):
        return joblib.load(joblib_path)
    return None


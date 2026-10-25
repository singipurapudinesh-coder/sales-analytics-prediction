"""
End-to-End Sales Analytics & Prediction System
Visualization Engine using Plotly
Generates executive-grade interactive charts: Cohort Heatmaps, RFM Treemaps,
Pareto 80/20 Curves, Forecast vs Actual with Confidence Bands, and Model Diagnostics.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Color themes
PALETTE = {
    "primary": "#3B82F6",      # Blue
    "secondary": "#10B981",    # Emerald
    "accent": "#F59E0B",       # Amber
    "danger": "#EF4444",       # Crimson
    "purple": "#8B5CF6",       # Violet
    "dark": "#1E293B",         # Slate Dark
    "light": "#F8FAFC",        # Off White
    "grid": "#E2E8F0",         # Border
}


def apply_custom_layout(fig: go.Figure, title: str = "", height: int = 400) -> go.Figure:
    """Apply uniform corporate aesthetics to any Plotly figure."""
    fig.update_layout(
        title={
            "text": f"<b>{title}</b>",
            "font": {"size": 16, "family": "Inter, Roboto, sans-serif"},
            "x": 0.02,
            "xanchor": "left",
        },
        height=height,
        margin={"l": 40, "r": 40, "t": 60, "b": 40},
        plot_bgcolor="rgba(248, 250, 252, 0.5)",
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, sans-serif", "color": "#1E293B"},
        hoverlabel={"bgcolor": "#1E293B", "font_color": "#FFFFFF", "font_size": 12},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    fig.update_xaxes(showgrid=True, gridcolor=PALETTE["grid"], zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=PALETTE["grid"], zeroline=False)
    return fig


def plot_sales_and_profit_timeline(df: pd.DataFrame) -> go.Figure:
    """Dual-axis timeline of Net Sales, Net Profit, and 7-day Moving Average."""
    df_sorted = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_sorted["order_date"]):
        df_sorted["order_date"] = pd.to_datetime(df_sorted["order_date"])

    daily = df_sorted.groupby("order_date").agg(
        revenue=("net_sales", "sum"),
        profit=("profit", "sum"),
    ).reset_index().sort_values(by="order_date")

    daily["ma7"] = daily["revenue"].rolling(7, min_periods=1).mean()

    fig = go.Figure()

    # Revenue Area
    fig.add_trace(go.Scatter(
        x=daily["order_date"],
        y=daily["revenue"],
        name="Daily Revenue ($)",
        mode="lines",
        line={"color": "rgba(59, 130, 246, 0.4)", "width": 1.5},
        fill="tozeroy",
        fillcolor="rgba(59, 130, 246, 0.08)",
    ))

    # 7-day Moving Average Trend
    fig.add_trace(go.Scatter(
        x=daily["order_date"],
        y=daily["ma7"],
        name="7-Day Moving Avg",
        mode="lines",
        line={"color": PALETTE["primary"], "width": 3},
    ))

    # Profit Line
    fig.add_trace(go.Scatter(
        x=daily["order_date"],
        y=daily["profit"],
        name="Net Profit ($)",
        mode="lines",
        line={"color": PALETTE["secondary"], "width": 2, "dash": "solid"},
    ))

    return apply_custom_layout(fig, title="Revenue & Profit Trajectory Over Time", height=420)


def plot_category_distribution(cat_df: pd.DataFrame) -> go.Figure:
    """Donut chart for Category Revenue Share and Margins."""
    fig = px.pie(
        cat_df,
        names="category",
        values="total_revenue",
        hole=0.45,
        color_discrete_sequence=[PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"], PALETTE["purple"], "#EC4899"],
    )
    fig.update_traces(
        textposition="outside",
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>Revenue: $%{value:,.2f}<br>Share: %{percent}",
    )
    return apply_custom_layout(fig, title="Revenue Share by Product Category", height=380)


def plot_regional_performance(region_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart for regional sales revenue and profit margin."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=region_df["region"],
        y=region_df["total_revenue"],
        name="Revenue ($)",
        marker_color=PALETTE["primary"],
        text=[f"${v:,.0f}" for v in region_df["total_revenue"]],
        textposition="auto",
    ))
    fig.add_trace(go.Bar(
        x=region_df["region"],
        y=region_df["total_profit"],
        name="Profit ($)",
        marker_color=PALETTE["secondary"],
        text=[f"${v:,.0f}" for v in region_df["total_profit"]],
        textposition="auto",
    ))

    fig.update_layout(barmode="group")
    return apply_custom_layout(fig, title="Regional Revenue & Profit Breakdown", height=380)


def plot_rfm_treemap(segment_summary: pd.DataFrame) -> go.Figure:
    """Interactive Treemap showing RFM Customer Segments and Spend."""
    fig = px.treemap(
        segment_summary,
        path=["segment"],
        values="total_revenue",
        color="avg_frequency",
        color_continuous_scale="Viridis",
        hover_data=["customer_count", "avg_recency", "avg_revenue_per_cust"],
    )
    fig.update_traces(
        hovertemplate="<b>%{label}</b><br>Total Revenue: $%{value:,.2f}<br>Customers: %{customdata[0]}<br>Avg Frequency: %{color:.1f} orders<br>Avg Recency: %{customdata[1]:.0f} days",
    )
    return apply_custom_layout(fig, title="RFM Customer Segmentation Treemap", height=420)


def plot_cohort_heatmap(cohort_matrix: pd.DataFrame) -> go.Figure:
    """Annotated heatmap for monthly cohort retention percentages."""
    fig = px.imshow(
        cohort_matrix,
        labels={"x": "Cohort Month Index", "y": "First Purchase Cohort", "color": "Retention %"},
        color_continuous_scale="Blues",
        text_auto=".0f",
        aspect="auto",
    )
    fig.update_xaxes(title="Months After First Purchase")
    fig.update_yaxes(title="Cohort Initial Month")
    return apply_custom_layout(fig, title="Customer Cohort Retention Rate (%)", height=460)


def plot_pareto_analysis(prod_sales: pd.DataFrame) -> go.Figure:
    """Dual-axis Pareto 80/20 chart: Product revenue bars with cumulative percentage curve."""
    top_30 = prod_sales.head(30).copy()
    fig = go.Figure()

    # Product Revenue Bars
    fig.add_trace(go.Bar(
        x=top_30["product_name"],
        y=top_30["total_revenue"],
        name="Revenue ($)",
        marker_color=PALETTE["primary"],
        yaxis="y",
    ))

    # Cumulative Percentage Line
    fig.add_trace(go.Scatter(
        x=top_30["product_name"],
        y=top_30["cumulative_revenue_pct"],
        name="Cumulative %",
        mode="lines+markers",
        line={"color": PALETTE["danger"], "width": 3},
        yaxis="y2",
    ))

    # 80% Horizontal Threshold Line
    fig.add_hline(
        y=80.0,
        line_dash="dot",
        line_color="#475569",
        annotation_text="80% Revenue Cutoff",
        annotation_position="bottom right",
        secondary_y=True if hasattr(fig, "secondary_y") else False,
    )

    fig.update_layout(
        yaxis={"title": "Revenue ($)"},
        yaxis2={"title": "Cumulative Revenue %", "overlaying": "y", "side": "right", "range": [0, 105]},
        xaxis={"tickangle": -45},
    )
    return apply_custom_layout(fig, title="Pareto 80/20 Analysis (Top 30 Products)", height=450)


def plot_forecast_timeline(
    history_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    test_preds_df: Optional[pd.DataFrame] = None,
    best_model_name: str = "Forecast",
) -> go.Figure:
    """
    Comprehensive Time-Series plot showing Historical Data, Test Set Predictions,
    Future Forecast, and Upper/Lower Confidence Cone.
    """
    fig = go.Figure()

    # 1. Historical Actual Sales (Recent 120 days for clarity)
    hist_tail = history_df.tail(120).copy()
    fig.add_trace(go.Scatter(
        x=hist_tail["date"],
        y=hist_tail["sales"],
        name="Historical Actual",
        mode="lines",
        line={"color": "#64748B", "width": 2},
    ))

    # 2. Test Set Prediction if available
    if test_preds_df is not None and not test_preds_df.empty:
        pred_col = [c for c in test_preds_df.columns if "pred" in c and best_model_name in c]
        if not pred_col:
            pred_col = [c for c in test_preds_df.columns if "pred" in c]
        if pred_col:
            fig.add_trace(go.Scatter(
                x=test_preds_df["date"],
                y=test_preds_df[pred_col[0]],
                name=f"Test Prediction ({best_model_name})",
                mode="lines",
                line={"color": PALETTE["accent"], "width": 2.5, "dash": "dash"},
            ))

    # 3. Future Forecast Cone (Confidence Interval)
    if forecast_df is not None and not forecast_df.empty:
        # Upper bound
        fig.add_trace(go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["upper_bound"],
            name="95% Confidence Upper",
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        ))
        # Lower bound with fill
        fig.add_trace(go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["lower_bound"],
            name="Confidence Interval",
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(16, 185, 129, 0.18)",
            hoverinfo="skip",
        ))
        # Forecast Mean
        fig.add_trace(go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["forecast_sales"],
            name="Future Forecast",
            mode="lines+markers",
            line={"color": PALETTE["secondary"], "width": 3},
            marker={"size": 4},
        ))

    return apply_custom_layout(fig, title="Sales Demand Forecast & 95% Confidence Cone", height=450)


def plot_feature_importances(imp_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart showing top machine learning predictors."""
    top_10 = imp_df.head(10).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=top_10["Importance (%)"],
        y=top_10["Feature"],
        orientation="h",
        marker_color=PALETTE["purple"],
        text=[f"{v:.1f}%" for v in top_10["Importance (%)"]],
        textposition="auto",
    ))
    return apply_custom_layout(fig, title="Feature Importance / Predictive Driver Weights", height=380)


def plot_data_quality_gauge(score: float) -> go.Figure:
    """Indicator gauge chart for overall data quality score."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": "Data Quality Health Score", "font": {"size": 18}},
        number={"suffix": "%", "font": {"size": 36, "color": PALETTE["primary"]}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": PALETTE["primary"]},
            "steps": [
                {"range": [0, 60], "color": "#FEE2E2"},
                {"range": [60, 85], "color": "#FEF3C7"},
                {"range": [85, 100], "color": "#D1FAE5"},
            ],
            "threshold": {
                "line": {"color": PALETTE["secondary"], "width": 4},
                "thickness": 0.75,
                "value": score,
            },
        },
    ))
    return apply_custom_layout(fig, title="", height=280)


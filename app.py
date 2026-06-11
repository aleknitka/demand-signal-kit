import streamlit as st
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta

from demand_signal_kit.config import DataConfig, FeatureConfig, TrainingConfig
from demand_signal_kit.data.loader import load_data
from demand_signal_kit.data.sample_generator import generate_sample_data
from demand_signal_kit.data.splitter import train_test_split, future_dates
from demand_signal_kit.data.schema import DataSchema, validate_dataframe
from demand_signal_kit.features.pipeline import build_features
from demand_signal_kit.models.registry import get_model, list_models
from demand_signal_kit.evaluation.metrics import evaluate_forecast, compare_models

NUMERIC_TYPES = {pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                 pl.Float32, pl.Float64}


def get_numeric_feature_cols(df: pl.DataFrame, exclude: tuple[str, ...] = ()) -> list[str]:
    return [c for c in df.columns if c not in exclude and df[c].dtype in NUMERIC_TYPES]

st.set_page_config(
    page_title="Mimo Forecasting",
    page_icon="📊",
    layout="wide",
)

# ── Sidebar Configuration ──────────────────────────────────────────────────────

with st.sidebar:
    st.header("Configuration")

    st.subheader("Data Source")
    source_type = st.radio(
        "Source",
        ["Generate Sample", "Upload File", "PostgreSQL"],
        label_visibility="collapsed",
    )

    st.subheader("Column Mapping")
    date_col = st.text_input("Date column", "date")
    product_col = st.text_input("Product column", "product_id")
    target_col = st.text_input("Target column", "quantity")

    st.subheader("Feature Engineering")
    lag_periods = st.multiselect(
        "Lag periods (days)", [7, 14, 21, 28, 30, 60, 90], default=[7, 14, 28]
    )
    rolling_windows = st.multiselect(
        "Rolling windows (days)", [7, 14, 28], default=[7, 14, 28]
    )
    rolling_stats = st.multiselect(
        "Rolling stats", ["mean", "std", "min", "max"], default=["mean", "std"]
    )
    include_cyclical = st.checkbox("Cyclical features (sin/cos)", True)
    include_holidays = st.checkbox("Holiday features", True)
    holiday_country = st.text_input("Holiday country code", "US")

    st.subheader("Training")
    available_models = list_models()
    selected_models = st.multiselect("Models", available_models, default=available_models)
    horizon = st.slider("Forecast horizon (days)", 7, 90, 14)
    test_size = st.slider("Test size", 0.05, 0.5, 0.2, 0.05)

    st.divider()
    if st.button("Reset All", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ── Session State Init ─────────────────────────────────────────────────────────

for key, default in [
    ("raw_df", None),
    ("feature_df", None),
    ("train_df", None),
    ("test_df", None),
    ("trained_models", {}),
    ("predictions", {}),
    ("forecasts", {}),
    ("comparison_df", None),
    ("feature_cols", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Helper ─────────────────────────────────────────────────────────────────────


def get_feat_config() -> FeatureConfig:
    return FeatureConfig(
        lag_periods=lag_periods,
        rolling_windows=rolling_windows,
        rolling_stats=rolling_stats,
        include_cyclical=include_cyclical,
        include_holidays=include_holidays,
        holiday_country=holiday_country,
    )


def load_uploaded_data(uploaded) -> pl.DataFrame:
    if uploaded.name.endswith(".csv"):
        df = pl.read_csv(uploaded, try_parse_dates=True)
    else:
        df = pl.read_parquet(uploaded)
    schema = DataSchema(date_column=date_col, product_column=product_col, target_column=target_col)
    return validate_dataframe(df, schema)


# ── Main Tabs ──────────────────────────────────────────────────────────────────

st.title("Mimo Forecasting Dashboard")

tab_data, tab_features, tab_train, tab_forecast, tab_compare = st.tabs(
    ["Data", "Features", "Train", "Forecast", "Compare"]
)

# ── TAB: Data ──────────────────────────────────────────────────────────────────

with tab_data:
    st.header("Data Source")

    if source_type == "Generate Sample":
        if st.button("Load Sample Dataset (5 products, 3 years)", use_container_width=True):
            sample_path = "data/sample_demand.csv"
            import os
            if os.path.exists(sample_path):
                with st.spinner("Loading sample dataset..."):
                    df = pl.read_csv(sample_path, try_parse_dates=True)
                    st.session_state.raw_df = df
                    st.session_state.feature_df = None
                    st.session_state.trained_models = {}
                    st.session_state.predictions = {}
                    st.session_state.forecasts = {}
                    st.session_state.comparison_df = None
                st.success(f"Loaded {len(df):,} rows — 5 product categories with seasonal patterns, holidays, and promotions")
            else:
                st.warning("Sample file not found. Generating fresh data...")

        st.divider()
        st.caption("Or generate custom data:")
        col1, col2, col3 = st.columns(3)
        with col1:
            n_products = st.number_input("Products", 1, 20, 5)
        with col2:
            start = st.date_input("Start date", value=__import__("datetime").date(2022, 1, 1))
        with col3:
            end = st.date_input("End date", value=__import__("datetime").date(2024, 12, 31))

        if st.button("Generate Sample Data", use_container_width=True):
            with st.spinner("Generating..."):
                df = generate_sample_data(
                    n_products=n_products,
                    start_date=start,
                    end_date=end,
                )
                st.session_state.raw_df = df
                st.session_state.feature_df = None
                st.session_state.trained_models = {}
                st.session_state.predictions = {}
                st.session_state.forecasts = {}
                st.session_state.comparison_df = None
            st.success(f"Generated {len(df):,} rows")

    elif source_type == "Upload File":
        uploaded = st.file_uploader("Upload CSV or Parquet", type=["csv", "parquet"])
        if uploaded:
            try:
                df = load_uploaded_data(uploaded)
                st.session_state.raw_df = df
                st.session_state.feature_df = None
                st.session_state.trained_models = {}
                st.session_state.predictions = {}
                st.session_state.forecasts = {}
                st.session_state.comparison_df = None
                st.success(f"Loaded {len(df):,} rows")
            except Exception as e:
                st.error(f"Failed to load: {e}")

    elif source_type == "PostgreSQL":
        conn_str = st.text_input("Connection string", "postgresql://user:pass@host:5432/db")
        table = st.text_input("Table or SQL query", "sales_data")
        if st.button("Load from Database", use_container_width=True):
            try:
                config = DataConfig(
                    source_type="postgres",
                    connection_string=conn_str,
                    source_path=table,
                    date_column=date_col,
                    product_column=product_col,
                    target_column=target_col,
                )
                df = load_data(config)
                st.session_state.raw_df = df
                st.session_state.feature_df = None
                st.session_state.trained_models = {}
                st.session_state.predictions = {}
                st.session_state.forecasts = {}
                st.session_state.comparison_df = None
                st.success(f"Loaded {len(df):,} rows")
            except Exception as e:
                st.error(f"Failed to load: {e}")

    # Display loaded data
    raw_df = st.session_state.raw_df
    if raw_df is not None:
        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{len(raw_df):,}")
        c2.metric("Columns", raw_df.shape[1])
        c3.metric("Products", raw_df[product_col].n_unique())
        c4.metric(
            "Date Range",
            f"{raw_df[date_col].min()} → {raw_df[date_col].max()}",
        )

        st.subheader("Preview")
        st.dataframe(raw_df.head(200).to_pandas(), use_container_width=True)

        st.subheader("Time Series")
        pdf = raw_df.to_pandas()
        fig = px.line(
            pdf,
            x=date_col,
            y=target_col,
            color=product_col,
            title="Daily Demand by Product",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Distribution")
        fig2 = px.histogram(
            pdf,
            x=target_col,
            color=product_col,
            barmode="overlay",
            nbins=50,
            title="Demand Distribution",
        )
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, use_container_width=True)

# ── TAB: Features ──────────────────────────────────────────────────────────────

with tab_features:
    st.header("Feature Engineering")

    raw_df = st.session_state.raw_df
    if raw_df is None:
        st.info("Load data in the Data tab first.")
    else:
        if st.button("Build Features", use_container_width=True):
            with st.spinner("Building features..."):
                feat_config = get_feat_config()
                feature_df = build_features(
                    raw_df,
                    feat_config,
                    date_column=date_col,
                    target_column=target_col,
                    product_column=product_col,
                )
                st.session_state.feature_df = feature_df
                feature_cols = get_numeric_feature_cols(
                    feature_df, exclude=(date_col, product_col, target_col)
                )
                st.session_state.feature_cols = feature_cols
            st.success(f"Built {len(feature_cols)} features")

        feature_df = st.session_state.feature_df
        if feature_df is not None:
            feature_cols = st.session_state.feature_cols
            st.metric("Total Features", len(feature_cols))

            st.subheader("Feature Columns")
            st.code(", ".join(feature_cols), language=None)

            st.subheader("Feature Preview")
            st.dataframe(
                feature_df.head(100).to_pandas(),
                use_container_width=True,
            )

            st.subheader("Feature Statistics")
            stats = feature_df.select(feature_cols).describe()
            st.dataframe(stats.to_pandas(), use_container_width=True)

            st.subheader("Correlation with Target")
            corrs = (
                feature_df.select(feature_cols + [target_col])
                .corr()
                .tail(1)
                .drop(target_col)
                .to_pandas()
                .T
                .sort_values(by=target_col, ascending=False)
            )
            corrs.columns = ["correlation"]
            fig = px.bar(
                corrs.reset_index(),
                x="index",
                y="correlation",
                title="Feature Correlation with Target",
            )
            fig.update_layout(height=400, xaxis_title="Feature", yaxis_title="Correlation")
            st.plotly_chart(fig, use_container_width=True)

# ── TAB: Train ─────────────────────────────────────────────────────────────────

with tab_train:
    st.header("Model Training")

    feature_df = st.session_state.feature_df
    if feature_df is None:
        st.info("Build features in the Features tab first.")
    elif not selected_models:
        st.warning("Select at least one model in the sidebar.")
    else:
        if st.button("Train Models", use_container_width=True):
            train_df, test_df = train_test_split(
                feature_df, date_column=date_col, test_size=test_size
            )
            st.session_state.train_df = train_df
            st.session_state.test_df = test_df

            feature_cols = st.session_state.feature_cols
            progress = st.progress(0)
            trained_models = {}
            predictions = {}

            for i, model_name in enumerate(selected_models):
                progress.progress(
                    i / len(selected_models),
                    text=f"Training {model_name}...",
                )
                try:
                    forecaster = get_model(model_name)()
                    forecaster.fit(
                        train_df, date_col, target_col, product_col, feature_cols
                    )
                    preds = forecaster.predict(
                        test_df, date_col, product_col, feature_cols
                    )
                    merged = test_df.select([date_col, product_col, target_col]).join(
                        preds, on=[date_col, product_col]
                    )
                    trained_models[model_name] = forecaster
                    predictions[model_name] = merged
                except Exception as e:
                    st.error(f"Failed to train {model_name}: {e}")

            progress.progress(1.0, text="Done!")
            st.session_state.trained_models = trained_models
            st.session_state.predictions = predictions

            if predictions:
                comparison = compare_models(
                    predictions,
                    date_column=date_col,
                    target_column=target_col,
                )
                st.session_state.comparison_df = comparison

        # Show results
        predictions = st.session_state.predictions
        comparison_df = st.session_state.comparison_df

        if comparison_df is not None:
            st.divider()
            st.subheader("Model Comparison")
            st.dataframe(comparison_df.to_pandas(), use_container_width=True)

            fig = px.bar(
                comparison_df.to_pandas(),
                x="model",
                y=["MAE", "RMSE", "MAPE", "SMAPE"],
                barmode="group",
                title="Metrics by Model",
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        if predictions:
            st.subheader("Actual vs Predicted")
            products = sorted(
                st.session_state.test_df[product_col].unique().to_list()
            )
            sel_products = st.multiselect(
                "Select products", products, default=products[:3], key="train_products"
            )

            test_df = st.session_state.test_df
            for model_name, pred_df in predictions.items():
                st.markdown(f"**{model_name}**")
                pdf = (
                    test_df.select([date_col, product_col, target_col])
                    .join(pred_df, on=[date_col, product_col], suffix="_pred")
                    .to_pandas()
                )
                pdf = pdf[pdf[product_col].isin(sel_products)]

                fig = go.Figure()
                for pid in sel_products:
                    mask = pdf[product_col] == pid
                    fig.add_trace(
                        go.Scatter(
                            x=pdf.loc[mask, date_col],
                            y=pdf.loc[mask, target_col],
                            name=f"{pid} actual",
                            mode="lines",
                        )
                    )
                    pred_col = f"{target_col}_pred" if f"{target_col}_pred" in pdf.columns else "forecast"
                    fig.add_trace(
                        go.Scatter(
                            x=pdf.loc[mask, date_col],
                            y=pdf.loc[mask, pred_col],
                            name=f"{pid} predicted",
                            mode="lines",
                            line=dict(dash="dash"),
                        )
                    )
                fig.update_layout(height=350, title=f"{model_name}: Actual vs Predicted")
                st.plotly_chart(fig, use_container_width=True)

# ── TAB: Forecast ──────────────────────────────────────────────────────────────

with tab_forecast:
    st.header("Future Forecast")

    trained_models = st.session_state.trained_models
    raw_df = st.session_state.raw_df

    if not trained_models:
        st.info("Train models in the Train tab first.")
    elif raw_df is None:
        st.info("Load data in the Data tab first.")
    else:
        model_name = st.selectbox(
            "Select model", list(trained_models.keys()), key="forecast_model"
        )
        forecast_horizon = st.slider(
            "Forecast horizon (days)", 7, 90, horizon, key="forecast_horizon"
        )

        products = raw_df[product_col].unique().to_list()
        sel_products = st.multiselect(
            "Select products", products, default=products, key="forecast_products"
        )

        if st.button("Generate Forecast", use_container_width=True):
            with st.spinner("Generating forecast..."):
                last_date = raw_df[date_col].max().isoformat()
                future = future_dates(last_date, forecast_horizon, sel_products)
                feat_config = get_feat_config()
                combined = build_features(
                    pl.concat([raw_df, future]),
                    feat_config,
                    date_column=date_col,
                    target_column=target_col,
                    product_column=product_col,
                )
                future_feat = combined.filter(pl.col(date_col) > last_date)

                feature_cols = st.session_state.feature_cols
                forecaster = trained_models[model_name]
                forecast = forecaster.predict(
                    future_feat, date_col, product_col, feature_cols
                )
                st.session_state.forecasts[model_name] = forecast

            st.success(f"Generated {forecast_horizon}-day forecast")

        forecast_result = st.session_state.forecasts.get(model_name)
        if forecast_result is not None:
            st.subheader("Forecast Chart")
            recent_days = 60
            cutoff = raw_df[date_col].max() - timedelta(days=recent_days)
            recent = raw_df.filter(pl.col(date_col) > cutoff)

            fig = go.Figure()
            for pid in sel_products:
                hist = recent.filter(pl.col(product_col) == pid).to_pandas()
                pred = forecast_result.filter(pl.col(product_col) == pid).to_pandas()

                fig.add_trace(
                    go.Scatter(
                        x=hist[date_col],
                        y=hist[target_col],
                        name=f"{pid} actual",
                        mode="lines",
                        line=dict(color="rgba(0,123,255,0.7)"),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=pred[date_col],
                        y=pred["forecast"],
                        name=f"{pid} forecast",
                        mode="lines",
                        line=dict(dash="dash", color="rgba(255,99,132,0.9)"),
                    )
                )

            fig.update_layout(
                height=450,
                title=f"{model_name} — {forecast_horizon}-Day Forecast",
                xaxis_title="Date",
                yaxis_title=target_col,
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Forecast Data")
            st.dataframe(forecast_result.to_pandas(), use_container_width=True)

# ── TAB: Compare ───────────────────────────────────────────────────────────────

with tab_compare:
    st.header("Model Comparison")

    comparison_df = st.session_state.comparison_df
    predictions = st.session_state.predictions

    if comparison_df is None:
        st.info("Train multiple models in the Train tab to compare.")
    else:
        st.subheader("Metrics Overview")
        st.dataframe(comparison_df.to_pandas(), use_container_width=True)

        # Grouped bar chart
        pdf = comparison_df.to_pandas()
        metrics = ["MAE", "RMSE", "MAPE", "SMAPE"]
        fig = go.Figure()
        for metric in metrics:
            fig.add_trace(
                go.Bar(
                    name=metric,
                    x=pdf["model"],
                    y=pdf[metric],
                )
            )
        fig.update_layout(
            barmode="group",
            height=400,
            title="All Metrics by Model",
            yaxis_title="Value",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Radar chart
        st.subheader("Metric Radar")
        fig_radar = go.Figure()
        for _, row in pdf.iterrows():
            values = [row[m] for m in metrics]
            values.append(values[0])  # close the polygon
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=values,
                    theta=metrics + [metrics[0]],
                    name=row["model"],
                    fill="toself",
                    opacity=0.6,
                )
            )
        fig_radar.update_layout(
            height=400,
            polar=dict(radialaxis=dict(visible=True)),
            title="Model Comparison Radar",
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # Per-product breakdown
        if predictions:
            st.subheader("Per-Product Breakdown")
            test_df = st.session_state.test_df
            products = sorted(test_df[product_col].unique().to_list())
            sel_product = st.selectbox("Select product", products, key="compare_product")

            product_metrics = []
            for model_name, pred_df in predictions.items():
                mask = test_df[product_col] == sel_product
                actual = test_df.filter(mask)[target_col]
                predicted = pred_df.filter(pred_df[product_col] == sel_product)[
                    "forecast"
                ]
                if len(predicted) > 0:
                    m = evaluate_forecast(actual, predicted)
                    m["model"] = model_name
                    product_metrics.append(m)

            if product_metrics:
                prod_df = pl.DataFrame(product_metrics)
                st.dataframe(prod_df.to_pandas(), use_container_width=True)

                fig = px.bar(
                    prod_df.to_pandas(),
                    x="model",
                    y=metrics,
                    barmode="group",
                    title=f"Metrics for {sel_product}",
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)

        # Residual plot
        if predictions:
            st.subheader("Residual Analysis")
            test_df = st.session_state.test_df
            model_name = st.selectbox(
                "Model for residuals",
                list(predictions.keys()),
                key="residual_model",
            )
            pred_df = predictions[model_name]
            merged = (
                test_df.select([date_col, product_col, target_col])
                .join(pred_df, on=[date_col, product_col])
            )
            merged = merged.with_columns(
                (pl.col(target_col) - pl.col("forecast")).alias("residual")
            )
            pdf = merged.to_pandas()

            fig = px.scatter(
                pdf,
                x=date_col,
                y="residual",
                color=product_col,
                title=f"{model_name}: Residuals Over Time",
                opacity=0.6,
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

"""
app.py
======
Streamlit dashboard for the Real Estate Price Estimator & Market Insights.

The dashboard is scoped to a **market** (US metro) selected in the sidebar.
Market-level values and trends come from **live Zillow Research data** (with a
committed snapshot fallback); listing-level analytics and price estimates come
from a model trained on synthetic listings anchored to real neighbourhood
values.

Run with::

    streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split

import config
import market_data
import model as model_lib
from data_loader import get_location_stats, load_or_create_data

# --------------------------------------------------------------------------- #
# Page setup
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Real Estate Price Estimator & Market Insights",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PLOTLY_TEMPLATE = "plotly_white"
FONT_FAMILY = "Inter, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif"


def inject_css() -> None:
    """Inject the dashboard's light, corporate styling."""
    st.markdown(
        """
        <style>
        :root {
            --ink: #0F172A;
            --muted: #64748B;
            --border: #E2E8F0;
            --primary: #4F46E5;
            --surface: #FFFFFF;
        }
        html, body, [class*="css"] { font-family: Inter, -apple-system, "Segoe UI", Roboto, Arial, sans-serif; }
        .block-container { padding-top: 2.0rem; padding-bottom: 3rem; max-width: 1500px; }

        .hero {
            background: linear-gradient(120deg, #4F46E5 0%, #6366F1 45%, #0EA5E9 100%);
            border-radius: 18px; padding: 26px 32px; color: #FFFFFF;
            box-shadow: 0 10px 30px rgba(79, 70, 229, 0.22); margin-bottom: 22px;
        }
        .hero h1 { margin: 0; font-size: 1.7rem; font-weight: 700; letter-spacing: -0.02em; }
        .hero p { margin: 6px 0 0 0; opacity: 0.92; font-size: 0.95rem; }
        .hero .live-badge {
            display: inline-block; margin-top: 12px; padding: 4px 12px; border-radius: 999px;
            background: rgba(255,255,255,0.18); font-size: 0.74rem; font-weight: 600;
            border: 1px solid rgba(255,255,255,0.35);
        }

        .kpi-card {
            background: var(--surface); border: 1px solid var(--border); border-radius: 14px;
            padding: 16px 18px; box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06); height: 100%;
        }
        .kpi-label { font-size: 0.72rem; letter-spacing: 0.07em; text-transform: uppercase; color: var(--muted); font-weight: 600; }
        .kpi-value { font-size: 1.55rem; font-weight: 700; color: var(--ink); margin-top: 6px; }
        .kpi-sub { font-size: 0.78rem; color: var(--muted); margin-top: 3px; }

        .result-card {
            background: linear-gradient(135deg, #EEF2FF 0%, #E0F2FE 100%);
            border: 1px solid #C7D2FE; border-radius: 16px; padding: 20px 22px;
        }
        .result-label { font-size: 0.74rem; letter-spacing: 0.07em; text-transform: uppercase; color: #3730A3; font-weight: 700; }
        .result-value { font-size: 2.0rem; font-weight: 800; color: #1E1B4B; margin: 4px 0; }
        .result-range { font-size: 0.92rem; color: #3730A3; font-weight: 600; }
        .result-note { font-size: 0.8rem; color: #475569; margin-top: 8px; }

        .pill { display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: 0.74rem; font-weight: 600; }
        .pill-up { background: #DCFCE7; color: #166534; }
        .pill-down { background: #FEE2E2; color: #991B1B; }
        .pill-neutral { background: #E2E8F0; color: #334155; }

        section[data-testid="stSidebar"] { border-right: 1px solid var(--border); }
        .footer { color: var(--muted); font-size: 0.78rem; margin-top: 36px; text-align: center; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def money(value: float, decimals: int = 0) -> str:
    """Format a number as compact dollars (e.g. ``$1.2M``)."""
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:,.2f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:,.1f}K"
    return f"${value:,.{decimals}f}"


def dollars(value: float) -> str:
    """Format a number as full dollars (e.g. ``$1,234,567``)."""
    return f"${value:,.0f}"


def kpi_card(label: str, value: str, sub: str = "") -> None:
    """Render a single KPI card."""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_figure(fig: go.Figure, height: int = 420) -> go.Figure:
    """Apply the shared visual theme to a Plotly figure."""
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=height,
        font=dict(family=FONT_FAMILY, size=13, color=config.COLORS["ink"]),
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        colorway=config.CHART_SEQUENCE,
    )
    fig.update_xaxes(showgrid=False, linecolor=config.COLORS["border"])
    fig.update_yaxes(gridcolor=config.COLORS["border"], zeroline=False)
    return fig


def market_short_name(market: str) -> str:
    """``"Atlanta, GA"`` -> ``"Atlanta"``."""
    return str(market).split(",")[0].strip()


# --------------------------------------------------------------------------- #
# Data & model loading (cached)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Generating and preparing the housing dataset...")
def get_data() -> pd.DataFrame:
    """Load (or synthesise) the cleaned housing dataset."""
    return load_or_create_data()


@st.cache_resource(show_spinner="Training the price model...")
def get_model() -> tuple:
    """Load the persisted model, training on first use if needed."""
    return model_lib.ensure_model()


@st.cache_data(ttl=3600, show_spinner="Loading market data...")
def get_market_overview() -> dict:
    """Load live market values (cached; falls back to the committed snapshot)."""
    return market_data.get_market_data()


@st.cache_data(show_spinner=False)
def get_hood_meta() -> pd.DataFrame:
    """Neighbourhood metadata joined to market names, with display labels."""
    meta = market_data.load_neighborhood_meta()
    markets = market_data.load_markets()
    merged = meta.merge(markets[["market_id", "market", "state"]], on="market_id", how="left")
    merged["location"] = merged.apply(
        lambda row: f"{row['neighborhood']} ({market_short_name(row['market'])})", axis=1
    )
    return merged


@st.cache_data(show_spinner=False)
def get_test_predictions(_model, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Recreate the held-out split and return ``(y_test, y_pred)`` dollars."""
    X = df[config.FEATURE_COLUMNS]
    y = df[config.TARGET_COLUMN].to_numpy(dtype=float)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED
    )
    return y_test, _model.predict(X_test)


@st.cache_data(show_spinner=False)
def get_age_banded_data(df: pd.DataFrame) -> pd.DataFrame:
    """Attach a human-readable property-age band used for colour encoding."""
    data = df.copy()
    data["age_band"] = pd.cut(
        data["property_age"],
        bins=config.AGE_BINS,
        labels=config.AGE_LABELS,
        right=True,
        include_lowest=True,
    )
    return data


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def render_market_selector(summary: pd.DataFrame) -> pd.Series:
    """Render the global market selector and return the selected market row."""
    st.sidebar.markdown("### 🌎 Market")
    options = summary.sort_values("size_rank")
    label_to_row = {
        f"{row.market}": row for row in options.itertuples(index=False)
    }
    choice = st.sidebar.selectbox(
        "Select a market",
        options=list(label_to_row.keys()),
        index=0,
        help="Values are live monthly Zillow Research data (ZHVI).",
    )
    return label_to_row[choice]


def render_filters(df: pd.DataFrame, market: str, hoods: pd.DataFrame) -> pd.DataFrame:
    """Render listing filters scoped to the selected market."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔎 Filters")

    market_df = df[df["market"] == market]
    locations = list(hoods.sort_values("latest_value", ascending=False)["location"])
    locations = [loc for loc in locations if loc in set(market_df["neighborhood"])]

    selected_locations = st.sidebar.multiselect(
        "Locations", options=locations, default=locations
    )

    price_min, price_max = int(market_df["price"].min()), int(market_df["price"].max())
    price_range = st.sidebar.slider(
        "Price range ($)",
        min_value=price_min,
        max_value=price_max,
        value=(price_min, price_max),
        step=10_000,
        format="$%d",
    )

    bedroom_range = st.sidebar.slider(
        "Bedrooms",
        min_value=int(market_df["bedrooms"].min()),
        max_value=int(market_df["bedrooms"].max()),
        value=(int(market_df["bedrooms"].min()), int(market_df["bedrooms"].max())),
    )

    mask = (
        market_df["neighborhood"].isin(selected_locations)
        & market_df["price"].between(*price_range)
        & market_df["bedrooms"].between(*bedroom_range)
    )
    filtered = market_df.loc[mask]
    if filtered.empty:
        st.sidebar.warning("No listings match the current filters.")
    return filtered


def render_prediction_tool(
    model,
    metrics: dict,
    market_row,
    hoods: pd.DataFrame,
    market_df: pd.DataFrame,
) -> None:
    """Render the live valuation widget for the selected market."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Live Valuation Tool")
    st.sidebar.caption(f"Estimating within **{market_row.market}**.")

    locations = list(hoods.sort_values("latest_value", ascending=False)["location"])
    locations = [loc for loc in locations if loc in set(market_df["neighborhood"])]
    if not locations:
        st.sidebar.info("No locations available for this market.")
        return

    numeric_bounds = config.FEATURE_BOUNDS

    with st.sidebar.form("valuation_form", clear_on_submit=False):
        neighborhood = st.selectbox("Location", options=locations, index=0)
        sqft = st.slider(
            "Living area (sq ft)",
            min_value=int(numeric_bounds["sqft"][0]),
            max_value=int(numeric_bounds["sqft"][1]),
            value=2_000,
            step=50,
        )
        col_a, col_b = st.columns(2)
        with col_a:
            bedrooms = st.number_input("Bedrooms", min_value=1, max_value=6, value=3, step=1)
        with col_b:
            bathrooms = st.number_input("Bathrooms", min_value=1.0, max_value=5.0, value=2.0, step=0.5)
        col_c, col_d = st.columns(2)
        with col_c:
            year_built = st.number_input(
                "Year built", min_value=1900, max_value=config.REFERENCE_YEAR, value=2005, step=1
            )
        with col_d:
            garage_spaces = st.number_input("Garage spaces", min_value=0, max_value=3, value=2, step=1)
        has_pool = st.toggle("Swimming pool", value=False)

        submitted = st.form_submit_button("Estimate Value", width="stretch", type="primary")

    if submitted:
        features = {
            "market": market_row.market,
            "neighborhood": neighborhood,
            "bedrooms": int(bedrooms),
            "bathrooms": float(bathrooms),
            "sqft": int(sqft),
            "lot_size": float(max(sqft * 3.5, numeric_bounds["lot_size"][0])),
            "year_built": int(year_built),
            "has_pool": bool(has_pool),
            "garage_spaces": int(garage_spaces),
        }
        st.session_state["prediction"] = model_lib.predict_with_range(model, features, metrics)
        st.session_state["prediction_features"] = features

    estimate = st.session_state.get("prediction")
    if not estimate:
        st.sidebar.info("Submit the form to generate an estimate.")
        return

    features = st.session_state["prediction_features"]
    if features.get("market") != market_row.market:
        st.sidebar.warning("Estimate is for a different market. Submit again to refresh.")
        return

    hood_row = hoods[hoods["location"] == features["neighborhood"]]
    reference_value = float(hood_row["latest_value"].iloc[0]) if not hood_row.empty else None

    st.sidebar.markdown("<br/>", unsafe_allow_html=True)
    st.sidebar.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">Model Estimate</div>
            <div class="result-value">{dollars(estimate['point'])}</div>
            <div class="result-range">Range {dollars(estimate['low'])} – {dollars(estimate['high'])}</div>
            <div class="result-note">±{estimate['half_width_pct']:.1f}% empirical range · model MAPE {estimate['mape']:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if reference_value:
        difference = estimate["point"] - reference_value
        pct = difference / reference_value * 100.0
        if pct > 1:
            pill = f'<span class="pill pill-up">▲ {pct:+.1f}% vs. neighborhood median</span>'
        elif pct < -1:
            pill = f'<span class="pill pill-down">▼ {pct:+.1f}% vs. neighborhood median</span>'
        else:
            pill = '<span class="pill pill-neutral">≈ in line with neighborhood</span>'
        st.sidebar.markdown("<br/>", unsafe_allow_html=True)
        st.sidebar.markdown(pill, unsafe_allow_html=True)
        st.sidebar.caption(
            f"{features['neighborhood']} real median: {dollars(reference_value)} (Zillow ZHVI)"
        )


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
def render_markets_tab(
    market_row,
    summary: pd.DataFrame,
    history: pd.DataFrame,
    hoods: pd.DataFrame,
    source: str,
) -> None:
    """Live market overview, trend and cross-market comparison."""
    badge = "🟢 live" if source == "live" else "🟡 snapshot"
    st.markdown(f"#### {market_row.market}")
    st.caption(
        f"Zillow Research ZHVI · latest month {pd.to_datetime(market_row.latest_month):%b %Y} · source: {badge}"
    )

    col_1, col_2, col_3, col_4 = st.columns(4)
    with col_1:
        kpi_card("Median Home Value", dollars(market_row.latest_value), "Latest published month")
    with col_2:
        kpi_card("Year over Year", f"{market_row.yoy_pct:+.1f}%", f"1 yr ago {money(market_row.value_1y_ago)}")
    with col_3:
        kpi_card("5-Year Change", f"{market_row.change_5y_pct:+.1f}%", "Since five years ago")
    with col_4:
        kpi_card("Month over Month", f"{market_row.mom_pct:+.2f}%", "Latest monthly move")

    st.markdown("<br/>", unsafe_allow_html=True)

    # --- Trend ------------------------------------------------------------ #
    st.markdown("#### 10-Year Value Trend")
    market_history = history[history["market_id"] == market_row.market_id].sort_values("month")
    market_history = market_history[
        market_history["month"] >= market_history["month"].max() - pd.DateOffset(years=10)
    ]
    trend = go.Figure()
    trend.add_trace(
        go.Scatter(
            x=market_history["month"],
            y=market_history["value"],
            mode="lines",
            line=dict(color=config.COLORS["primary"], width=3),
            fill="tozeroy",
            fillcolor="rgba(79,70,229,0.08)",
            hovertemplate="%{x|%b %Y}<br>$%{y:,.0f}<extra></extra>",
            name=market_row.market,
        )
    )
    trend.update_yaxes(tickprefix="$", tickformat=",")
    trend = style_figure(trend, height=360)
    st.plotly_chart(trend, width="stretch")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Latest Value — All Markets")
        frame = summary.sort_values("latest_value")
        colors = [
            config.COLORS["primary"]
            if mid == market_row.market_id
            else "#C7D2FE"
            for mid in frame["market_id"]
        ]
        bars = go.Figure(
            go.Bar(
                x=frame["latest_value"],
                y=frame["market"],
                orientation="h",
                marker_color=colors,
                text=[money(v) for v in frame["latest_value"]],
                textposition="outside",
                cliponaxis=False,
                hovertemplate="%{y}<br>$%{x:,.0f}<extra></extra>",
            )
        )
        bars.update_xaxes(tickprefix="$", tickformat=",")
        bars = style_figure(bars, height=460)
        st.plotly_chart(bars, width="stretch")

    with col_right:
        st.markdown("#### Year-over-Year Growth — All Markets")
        frame = summary.sort_values("yoy_pct")
        colors = [
            config.COLORS["success"] if v >= 0 else config.COLORS["danger"]
            for v in frame["yoy_pct"]
        ]
        growth = go.Figure(
            go.Bar(
                x=frame["yoy_pct"],
                y=frame["market"],
                orientation="h",
                marker_color=colors,
                text=[f"{v:+.1f}%" for v in frame["yoy_pct"]],
                textposition="outside",
                cliponaxis=False,
                hovertemplate="%{y}<br>%{x:.2f}%<extra></extra>",
            )
        )
        growth.add_vline(x=0, line_color=config.COLORS["muted"], line_width=1)
        growth.update_xaxes(ticksuffix="%")
        growth = style_figure(growth, height=460)
        st.plotly_chart(growth, width="stretch")

    st.markdown(f"#### Neighborhood Home Values — {market_row.market}")
    table = hoods.sort_values("latest_value", ascending=False)[
        ["neighborhood", "city", "county", "latest_value", "base_price_per_sqft"]
    ].rename(
        columns={
            "neighborhood": "Neighborhood",
            "city": "City",
            "county": "County",
            "latest_value": "Median Home Value",
            "base_price_per_sqft": "Price / Sq Ft (est.)",
        }
    )
    styled = table.copy()
    styled["Median Home Value"] = styled["Median Home Value"].map(dollars)
    styled["Price / Sq Ft (est.)"] = styled["Price / Sq Ft (est.)"].map(lambda v: f"${v:,.0f}")
    st.dataframe(styled, width="stretch", hide_index=True, height=380)


def render_analytics_tab(
    df: pd.DataFrame, importance: pd.DataFrame, hoods: pd.DataFrame, market: str
) -> None:
    """Price-per-sqft distribution and location structure for the market."""
    banded = get_age_banded_data(df)
    order = [
        loc
        for loc in hoods.sort_values("latest_value", ascending=False)["location"]
        if loc in set(banded["neighborhood"])
    ]

    st.markdown("#### Price per Sq Ft by Location")
    st.caption(f"Each point is a listing in {market}, colour-coded by property age band.")

    scatter = px.scatter(
        banded,
        x="neighborhood",
        y="price_per_sqft",
        color="age_band",
        category_orders={"neighborhood": order, "age_band": config.AGE_LABELS},
        color_discrete_sequence=config.CHART_SEQUENCE,
        opacity=0.65,
        labels={"neighborhood": "", "price_per_sqft": "Price per sq ft ($)", "age_band": "Age band"},
        hover_data={"bedrooms": True, "bathrooms": True, "sqft": ":,", "price": ":,"},
    )
    scatter.update_traces(marker=dict(size=8, line=dict(width=0.5, color="white")))
    scatter.update_yaxes(tickprefix="$", tickformat=",")
    scatter = style_figure(scatter, height=460)
    st.plotly_chart(scatter, width="stretch")

    col_left, col_right = st.columns([1.1, 1])

    with col_left:
        st.markdown("#### Median Listing Price by Location")
        location_stats = get_location_stats(df).sort_values("median_price")
        bar = px.bar(
            location_stats,
            x="median_price",
            y="neighborhood",
            orientation="h",
            text="median_price",
            labels={"median_price": "Median price ($)", "neighborhood": ""},
            color="median_price",
            color_continuous_scale=["#C7D2FE", config.COLORS["primary"]],
        )
        bar.update_traces(texttemplate="%{text:$,.0f}", textposition="outside", cliponaxis=False)
        bar.update_layout(coloraxis_showscale=False)
        bar.update_xaxes(tickprefix="$", tickformat=",")
        bar = style_figure(bar, height=420)
        st.plotly_chart(bar, width="stretch")

    with col_right:
        st.markdown("#### What Drives Home Prices")
        st.caption("Aggregated XGBoost gain importance across all markets.")
        imp = importance.sort_values("importance")
        imp_bar = px.bar(
            imp,
            x="importance",
            y="feature",
            orientation="h",
            labels={"importance": "Relative importance", "feature": ""},
            color="importance",
            color_continuous_scale=["#E0E7FF", config.COLORS["primary"]],
        )
        imp_bar.update_layout(coloraxis_showscale=False)
        imp_bar.update_traces(texttemplate="%{x:.3f}", textposition="outside", cliponaxis=False)
        imp_bar = style_figure(imp_bar, height=420)
        st.plotly_chart(imp_bar, width="stretch")


def render_model_tab(model, metrics: dict, df: pd.DataFrame) -> None:
    """Model quality diagnostics: metrics, parity plot and residual spread."""
    model_name = metrics.get("model_name", "Regressor")

    col_1, col_2, col_3, col_4 = st.columns(4)
    with col_1:
        kpi_card("MAE", dollars(metrics.get("mae", 0)), "Avg. miss on test set")
    with col_2:
        kpi_card("RMSE", dollars(metrics.get("rmse", 0)), "Penalises large misses")
    with col_3:
        kpi_card("MAPE", f"{metrics.get('mape', 0):.2f}%", "Mean absolute % error")
    with col_4:
        kpi_card("R²", f"{metrics.get('r2', 0):.4f}", "Variance explained")

    st.markdown("<br/>", unsafe_allow_html=True)
    st.caption(
        f"Estimator: **{model_name}** · target transform: "
        f"{metrics.get('target_transform', 'log1p/expm1')} · "
        f"trained {metrics.get('trained_at', 'n/a')} · "
        f"{metrics.get('n_train', 0):,} train / {metrics.get('n_test', 0):,} test rows · "
        f"{metrics.get('n_records', 0):,} listings."
    )

    y_test, y_pred = get_test_predictions(model, df)

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Predicted vs. Actual Price")
        parity = go.Figure()
        parity.add_trace(
            go.Scatter(
                x=y_test,
                y=y_pred,
                mode="markers",
                marker=dict(size=6, color=config.COLORS["primary"], opacity=0.4, line=dict(width=0)),
                name="Test listings",
                hovertemplate="Actual %{x:$,.0f}<br>Predicted %{y:$,.0f}<extra></extra>",
            )
        )
        limits = [float(min(y_test.min(), y_pred.min())), float(max(y_test.max(), y_pred.max()))]
        parity.add_trace(
            go.Scatter(
                x=limits,
                y=limits,
                mode="lines",
                line=dict(color=config.COLORS["danger"], dash="dash", width=2),
                name="Perfect prediction",
            )
        )
        parity.update_xaxes(tickprefix="$", tickformat=",")
        parity.update_yaxes(tickprefix="$", tickformat=",")
        parity = style_figure(parity, height=420)
        st.plotly_chart(parity, width="stretch")

    with col_right:
        st.markdown("#### Percentage Error Distribution")
        pct_error = (y_pred - y_test) / y_test * 100.0
        hist = go.Figure(
            go.Histogram(
                x=pct_error, nbinsx=45,
                marker=dict(color=config.COLORS["accent"], line=dict(width=0)),
                name="Error",
                hovertemplate="Error %{x:.1f}%<br>Count %{y}<extra></extra>",
            )
        )
        hist.add_vline(x=0, line_dash="dash", line_color=config.COLORS["danger"], annotation_text="Zero error")
        hist.update_xaxes(title="Percentage error (%)")
        hist.update_yaxes(title="Listings")
        hist = style_figure(hist, height=420)
        st.plotly_chart(hist, width="stretch")

    with st.expander("Evaluation summary"):
        summary = pd.DataFrame(
            [
                ("Model", model_name),
                ("R² score", f"{metrics.get('r2', 0):.4f}"),
                ("MAE", dollars(metrics.get("mae", 0))),
                ("RMSE", dollars(metrics.get("rmse", 0))),
                ("MAPE", f"{metrics.get('mape', 0):.2f}%"),
                ("Median APE", f"{metrics.get('median_ape', 0):.2f}%"),
                ("Baseline MAE (predict median)", dollars(metrics.get("baseline_mae", 0))),
                ("Improvement vs. baseline", f"{metrics.get('improvement_vs_baseline_pct', 0):.1f}% lower MAE"),
                ("80% empirical range", f"{metrics.get('residual_p10', 0):+.1f}% to {metrics.get('residual_p90', 0):+.1f}%"),
                ("Listings", f"{metrics.get('n_records', 0):,}"),
            ],
            columns=["Metric", "Value"],
        )
        st.dataframe(summary, width="stretch", hide_index=True)


def render_data_tab(df: pd.DataFrame, hoods: pd.DataFrame, market: str) -> None:
    """Exploratory table view with download support."""
    st.markdown(f"#### Neighborhood Market Summary — {market}")
    table = hoods.sort_values("latest_value", ascending=False)[
        ["neighborhood", "city", "county", "latest_value", "base_price_per_sqft"]
    ].rename(
        columns={
            "neighborhood": "Neighborhood",
            "city": "City",
            "county": "County",
            "latest_value": "Median Home Value",
            "base_price_per_sqft": "Price / Sq Ft (est.)",
        }
    )
    styled = table.copy()
    styled["Median Home Value"] = styled["Median Home Value"].map(dollars)
    styled["Price / Sq Ft (est.)"] = styled["Price / Sq Ft (est.)"].map(lambda v: f"${v:,.0f}")
    st.dataframe(styled, width="stretch", hide_index=True)

    st.markdown("#### Listing Explorer")
    st.caption(f"{len(df):,} synthetic listings match the current filters in {market}.")
    display_columns = config.RAW_COLUMNS + ["price_per_sqft", "property_age"]
    st.dataframe(df[display_columns], width="stretch", height=380)

    st.download_button(
        "⬇️ Download filtered listings (CSV)",
        data=df[display_columns].to_csv(index=False).encode("utf-8"),
        file_name="filtered_listings.csv",
        mime="text/csv",
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    inject_css()

    data = get_data()
    model, metrics, importance = get_model()
    overview = get_market_overview()
    summary = overview["summary"]
    history = overview["history"]
    source = overview["source"]
    hood_meta = get_hood_meta()

    st.markdown(
        """
        <div class="hero">
            <h1>🏙️ Real Estate Price Estimator &amp; Market Insights</h1>
            <p>Live US market data · gradient-boosted valuation · interactive analytics</p>
            <span class="live-badge">Market values: Zillow Research (monthly ZHVI)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Sidebar ---------------------------------------------------------- #
    market_row = render_market_selector(summary)
    market = market_row.market
    market_df = data[data["market"] == market]
    hoods = hood_meta[hood_meta["market_id"] == market_row.market_id]

    filtered = render_filters(data, market, hoods)
    render_prediction_tool(model, metrics, market_row, hoods, market_df)
    display = filtered if not filtered.empty else market_df

    # --- KPI cards -------------------------------------------------------- #
    kpi_1, kpi_2, kpi_3, kpi_4 = st.columns(4)
    with kpi_1:
        kpi_card("Listings", f"{len(display):,}", f"Synthetic, {market_short_name(market)}")
    with kpi_2:
        kpi_card(
            "Median Market Price",
            dollars(market_row.latest_value),
            f"Real ZHVI · {market_row.yoy_pct:+.1f}% YoY",
        )
    with kpi_3:
        kpi_card(
            "Model Accuracy (100 − MAPE)",
            f"{100 - metrics.get('mape', 0):.1f}%",
            f"Mean absolute % error {metrics.get('mape', 0):.2f}%",
        )
    with kpi_4:
        kpi_card("Model R² Score", f"{metrics.get('r2', 0):.3f}", "Held-out variance explained")

    st.markdown("<br/>", unsafe_allow_html=True)

    tab_markets, tab_analytics, tab_model, tab_data = st.tabs(
        ["🌎 Markets", "📊 Market Analytics", "🤖 Model Insights", "🗂️ Data Explorer"]
    )
    with tab_markets:
        render_markets_tab(market_row, summary, history, hoods, source)
    with tab_analytics:
        if display.empty:
            st.warning("No listings match the current filters.")
        else:
            render_analytics_tab(display, importance, hoods, market)
    with tab_model:
        render_model_tab(model, metrics, data)
    with tab_data:
        render_data_tab(display, hoods, market)

    st.markdown(
        f"""
        <div class="footer">
            {config.ZILLOW_ATTRIBUTION} · Listing-level estimates from a model trained on synthetic
            listings calibrated to real neighbourhood medians. Portfolio demo — not financial advice.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

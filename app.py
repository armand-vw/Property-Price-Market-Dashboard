"""
app.py
======
Streamlit dashboard for the Real Estate Price Estimator & Market Insights.

The dashboard is scoped to a **country** selected in the sidebar. The United
States shows a national view (aggregated from the 15 real Zillow metros) plus a
location-based valuation tool; the other markets show national BIS house-price
indices. A plain-English education layer (tooltips, a "what this means"
summary and a Hot/Warm/Cool rating) helps everyday users interpret the numbers.

Run with::

    streamlit run app.py
"""

from __future__ import annotations

import html
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split

import config
import insights
import international_data
import market_data
import model as model_lib
from data_loader import get_market_stats, load_or_create_data

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
            --ink: #0F172A; --muted: #64748B; --border: #E2E8F0;
            --primary: #4F46E5; --surface: #FFFFFF;
        }
        html, body, [class*="css"] { font-family: Inter, -apple-system, "Segoe UI", Roboto, Arial, sans-serif; }
        .block-container { padding-top: 2.0rem; padding-bottom: 3rem; max-width: 1500px; }

        .hero {
            background: linear-gradient(120deg, #4F46E5 0%, #6366F1 45%, #0EA5E9 100%);
            border-radius: 18px; padding: 26px 32px; color: #FFFFFF;
            box-shadow: 0 10px 30px rgba(79, 70, 229, 0.22); margin-bottom: 20px;
        }
        .hero h1 { margin: 0; font-size: 1.7rem; font-weight: 700; letter-spacing: -0.02em; }
        .hero p { margin: 6px 0 0 0; opacity: 0.92; font-size: 0.95rem; }

        .kpi-card {
            background: var(--surface); border: 1px solid var(--border); border-radius: 14px;
            padding: 16px 18px; box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06); height: 100%;
        }
        .kpi-label { font-size: 0.72rem; letter-spacing: 0.07em; text-transform: uppercase; color: var(--muted); font-weight: 600; }
        .kpi-label .info { cursor: help; color: var(--muted); font-size: 0.85em; }
        .kpi-value { font-size: 1.55rem; font-weight: 700; color: var(--ink); margin-top: 6px; }
        .kpi-sub { font-size: 0.78rem; color: var(--muted); margin-top: 3px; }

        .summary-card {
            background: linear-gradient(135deg, #EEF2FF 0%, #E0F2FE 100%);
            border: 1px solid #C7D2FE; border-radius: 16px; padding: 18px 22px; margin-bottom: 8px;
        }
        .summary-label { font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; color: #3730A3; font-weight: 700; }
        .summary-card p { margin: 8px 0 0 0; font-size: 0.98rem; color: #1E1B4B; line-height: 1.55; }
        .temp-badge { display: inline-block; margin-top: 10px; padding: 3px 12px; border-radius: 999px; font-size: 0.78rem; font-weight: 700; }
        .temp-hot { background: #FEE2E2; color: #991B1B; }
        .temp-warm { background: #FEF3C7; color: #92400E; }
        .temp-cool { background: #DBEAFE; color: #1E40AF; }

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


def kpi_card(label: str, value: str, sub: str = "", help_text: str | None = None) -> None:
    """Render a single KPI card, with an optional ⓘ tooltip."""
    icon = f' <span class="info" title="{html.escape(help_text)}">ⓘ</span>' if help_text else ""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}{icon}</div>
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


def temperature_badge(temperature: dict | None) -> str:
    """HTML badge for the Hot/Warm/Cool market temperature."""
    if not temperature:
        return ""
    label = temperature.get("label", "Warm")
    css = {"Hot": "temp-hot", "Warm": "temp-warm", "Cool": "temp-cool"}.get(label, "temp-warm")
    return f'<span class="temp-badge {css}">{label} market</span>'


def render_summary_card(narrative: str, temperature: dict | None) -> None:
    """Render the plain-English 'what this means' card."""
    st.markdown(
        f"""
        <div class="summary-card">
            <div class="summary-label">What this means</div>
            <p>{html.escape(narrative)}</p>
            {temperature_badge(temperature)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_glossary() -> None:
    """A plain-English glossary of every metric."""
    with st.expander("📚 Market 101 — what these numbers mean"):
        for term, definition in insights.glossary().items():
            st.markdown(f"**{term.replace('_', ' ').title()}** — {definition}")


def render_valuation_explainer(model) -> None:
    """Explain the most recent estimate with per-feature SHAP contributions."""
    features = st.session_state.get("prediction_features")
    if not features:
        return
    explanation = model_lib.explain_prediction(model, features)
    if explanation is None or explanation.empty:
        return

    with st.expander("🔍 Why this estimate? — what pushed the price up or down", expanded=True):
        st.caption(
            "Approximate effect of each factor on this estimate (XGBoost SHAP "
            "contributions, shown as a percentage of the model's baseline). "
            "Green pulls the price up, red pulls it down."
        )
        frame = explanation.sort_values("impact_pct")
        colors = [
            config.COLORS["success"] if v >= 0 else config.COLORS["danger"]
            for v in frame["impact_pct"]
        ]
        figure = go.Figure(
            go.Bar(
                x=frame["impact_pct"], y=frame["feature"], orientation="h",
                marker_color=colors, text=[f"{v:+.0f}%" for v in frame["impact_pct"]],
                textposition="outside", cliponaxis=False,
                hovertemplate="%{y}<br>%{x:+.1f}%<extra></extra>",
            )
        )
        figure.add_vline(x=0, line_color=config.COLORS["muted"], line_width=1)
        figure.update_xaxes(ticksuffix="%")
        st.plotly_chart(style_figure(figure, 320), width="stretch")


# --------------------------------------------------------------------------- #
# Query-parameter state (shareable URLs)
# --------------------------------------------------------------------------- #
_ESTIMATOR_PARAMS: dict[str, str] = {
    "location": "location_select",
    "sqft": "sqft_input",
    "beds": "beds_input",
    "baths": "baths_input",
    "year": "year_input",
    "garage": "garage_input",
    "lot": "lot_input",
    "pool": "pool_input",
}
_NUMERIC_DEFAULTS: dict[str, float] = {
    "sqft_input": 2_000,
    "beds_input": 3,
    "baths_input": 2.0,
    "year_input": 2005,
    "garage_input": 2,
    "lot_input": 7_000,
}
_PARAM_FOR_KEY: dict[str, str] = {key: param for param, key in _ESTIMATOR_PARAMS.items()}


def seed_estimator_from_query(hood_meta: pd.DataFrame) -> None:
    """Seed the location and property inputs from query params (once)."""
    options = list(hood_meta.sort_values("latest_value", ascending=False)["location"])
    if not options:
        return
    if st.session_state.get("location_select") not in options:
        wanted = st.query_params.get("location")
        st.session_state["location_select"] = wanted if wanted in options else options[0]

    for key, default in _NUMERIC_DEFAULTS.items():
        if key in st.session_state:
            continue
        raw = st.query_params.get(_PARAM_FOR_KEY[key])
        try:
            st.session_state[key] = type(default)(raw) if raw not in (None, "") else default
        except (TypeError, ValueError):
            st.session_state[key] = default

    if "pool_input" not in st.session_state:
        raw = st.query_params.get("pool")
        st.session_state["pool_input"] = str(raw).lower() in {"1", "true", "yes"} if raw else False


def sync_query_params() -> None:
    """Mirror the current selections into the URL query string."""
    for param, key in _ESTIMATOR_PARAMS.items():
        if key not in st.session_state:
            continue
        value = st.session_state[key]
        st.query_params[param] = "1" if param == "pool" and value else (
            "0" if param == "pool" else str(value)
        )


def share_url() -> str:
    """Build the full shareable URL for the current state."""
    base = (getattr(st.context, "url", "") or "").split("?")[0]
    query = urlencode(st.query_params.to_dict())
    if not base:
        return f"?{query}" if query else ""
    return f"{base}?{query}" if query else base


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
def get_market_overview(force_refresh: bool = False) -> dict:
    """Load live US market values (cached; falls back to the committed snapshot)."""
    return market_data.get_market_data(force_refresh=force_refresh)


@st.cache_data(ttl=3600, show_spinner="Loading international data...")
def get_international(force_refresh: bool = False) -> dict:
    """Load international market data (BIS live-or-snapshot + UK regions)."""
    return international_data.get_country_data(force_refresh=force_refresh)


@st.cache_data(show_spinner=False)
def get_hood_meta() -> pd.DataFrame:
    """US neighbourhood metadata joined to market names, with display labels."""
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
def render_country_selector() -> str:
    """Render the country switcher and return the selected country code."""
    st.sidebar.markdown("### 🗺️ Country")
    codes = list(config.COUNTRY_ORDER)
    if "country_select" not in st.session_state:
        wanted = st.query_params.get("country")
        st.session_state["country_select"] = wanted if wanted in codes else "US"

    code = st.sidebar.selectbox(
        "Select a country",
        options=codes,
        format_func=lambda c: config.COUNTRIES[c]["name"],
        key="country_select",
        help="United States: national market plus a location-based valuation tool.",
    )
    st.query_params["country"] = code
    return code


def render_international_sidebar(country_code: str) -> None:
    """Explain the international (non-US) experience in the sidebar."""
    name = international_data.country_name(country_code)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📡 International Market")
    st.sidebar.caption(f"National market data for **{name}**.")
    st.sidebar.info(
        "Live valuation is available for the **United States** only (real "
        "listing-level data). For this country we show real national market data "
        "from the Bank for International Settlements (BIS)."
    )
    with st.sidebar.expander("📚 Market 101"):
        for term, definition in insights.glossary().items():
            st.caption(f"**{term.replace('_', ' ').title()}** — {definition}")


def render_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Render simple national listing filters."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔎 Filters")

    price_min, price_max = int(df["price"].min()), int(df["price"].max())
    price_range = st.sidebar.slider(
        "Price range ($)", min_value=price_min, max_value=price_max,
        value=(price_min, price_max), step=10_000, format="$%d",
    )
    bedroom_range = st.sidebar.slider(
        "Bedrooms",
        min_value=int(df["bedrooms"].min()),
        max_value=int(df["bedrooms"].max()),
        value=(int(df["bedrooms"].min()), int(df["bedrooms"].max())),
    )
    mask = df["price"].between(*price_range) & df["bedrooms"].between(*bedroom_range)
    filtered = df.loc[mask]
    if filtered.empty:
        st.sidebar.warning("No listings match the current filters.")
    return filtered


def render_prediction_tool(
    model, metrics: dict, hood_meta: pd.DataFrame, force_sync: bool = True
) -> None:
    """Render the valuation tool with a single all-US location dropdown."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Live Valuation Tool")
    st.sidebar.caption("Estimate any US property — pick a location.")

    hood_meta = hood_meta.sort_values("latest_value", ascending=False)
    option_to_market = dict(zip(hood_meta["location"], hood_meta["market"], strict=False))
    options = list(option_to_market)
    if not options:
        st.sidebar.info("No locations available.")
        return

    numeric_bounds = config.FEATURE_BOUNDS

    with st.sidebar.form("valuation_form", clear_on_submit=False):
        location = st.selectbox("Location", options=options, key="location_select")
        sqft = st.slider(
            "Living area (sq ft)", min_value=int(numeric_bounds["sqft"][0]),
            max_value=int(numeric_bounds["sqft"][1]), step=50, key="sqft_input",
        )
        col_a, col_b = st.columns(2)
        with col_a:
            bedrooms = st.number_input("Bedrooms", min_value=1, max_value=6, step=1, key="beds_input")
        with col_b:
            bathrooms = st.number_input("Bathrooms", min_value=1.0, max_value=5.0, step=0.5, key="baths_input")
        col_c, col_d = st.columns(2)
        with col_c:
            year_built = st.number_input(
                "Year built", min_value=1900, max_value=config.REFERENCE_YEAR, step=1, key="year_input"
            )
        with col_d:
            garage_spaces = st.number_input("Garage spaces", min_value=0, max_value=3, step=1, key="garage_input")
        lot_size = st.slider(
            "Lot size (sq ft)", min_value=int(numeric_bounds["lot_size"][0]),
            max_value=int(numeric_bounds["lot_size"][1]), step=250, key="lot_input",
        )
        has_pool = st.toggle("Swimming pool", key="pool_input")
        submitted = st.form_submit_button("Estimate Value", width="stretch", type="primary")

    if submitted:
        features = {
            "market": option_to_market[location],
            "neighborhood": location,
            "bedrooms": int(bedrooms),
            "bathrooms": float(bathrooms),
            "sqft": int(sqft),
            "lot_size": float(lot_size),
            "year_built": int(year_built),
            "has_pool": bool(has_pool),
            "garage_spaces": int(garage_spaces),
        }
        st.session_state["prediction"] = model_lib.predict_with_range(model, features, metrics)
        st.session_state["prediction_features"] = features

    if force_sync:
        sync_query_params()
    link = share_url()
    if link:
        with st.sidebar.expander("🔗 Shareable link"):
            st.code(link, language=None)
            st.caption("Copy this URL to restore this location and estimate.")

    estimate = st.session_state.get("prediction")
    if not estimate:
        st.sidebar.info("Submit the form to generate an estimate.")
        return

    features = st.session_state["prediction_features"]
    hood_row = hood_meta[hood_meta["location"] == features["neighborhood"]]
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
# US national tab
# --------------------------------------------------------------------------- #
def render_us_national_tab(
    summary: pd.DataFrame,
    history: pd.DataFrame,
    health: pd.DataFrame,
    national: dict,
) -> None:
    """National US aggregates plus all-metro comparison charts."""
    st.markdown("#### United States — national snapshot")
    st.caption("Aggregated (median) across the 15 largest US metros · Zillow Research.")

    row_1 = st.columns(4)
    with row_1[0]:
        kpi_card("Median Home Value", dollars(national["latest_value"] or 0),
                 "Median across metros", config.GLOSSARY["zhvi"])
    with row_1[1]:
        kpi_card("Median Rent", f"${national['latest_rent']:,.0f}/mo" if national["latest_rent"] else "n/a",
                 "Median across metros", config.GLOSSARY["zori"])
    with row_1[2]:
        kpi_card("Gross Rental Yield", f"{national['gross_yield_pct']:.1f}%" if national["gross_yield_pct"] else "n/a",
                 "Annual rent ÷ value", config.GLOSSARY["yield"])
    with row_1[3]:
        kpi_card("Median Sale Price", dollars(national["median_sale_price"] or 0),
                 "Closed sales", config.GLOSSARY["median_sale_price"])

    row_2 = st.columns(4)
    with row_2[0]:
        kpi_card("Year over Year", f"{national['yoy_pct']:+.1f}%" if national["yoy_pct"] is not None else "n/a",
                 "Across metros", config.GLOSSARY["yoy"])
    with row_2[1]:
        kpi_card("5-Year Change", f"{national['change_5y_pct']:+.1f}%" if national["change_5y_pct"] is not None else "n/a",
                 "Since five years ago", config.GLOSSARY["change_5y"])
    with row_2[2]:
        kpi_card("Days to Pending", f"{national['days_to_pending']:.0f}" if national["days_to_pending"] else "n/a",
                 "Median, latest month", config.GLOSSARY["days_to_pending"])
    with row_2[3]:
        kpi_card("For-Sale Inventory", f"{national['inventory']:,.0f}" if national["inventory"] else "n/a",
                 "Active listings", config.GLOSSARY["inventory"])

    st.markdown("<br/>", unsafe_allow_html=True)

    national_history = market_data.build_national_history(history)
    st.markdown("#### National Home Value Trend")
    trend = go.Figure()
    trend.add_trace(
        go.Scatter(
            x=national_history["month"], y=national_history["value"], mode="lines",
            line=dict(color=config.COLORS["primary"], width=3),
            fill="tozeroy", fillcolor="rgba(79,70,229,0.08)",
            hovertemplate="%{x|%b %Y}<br>$%{y:,.0f}<extra></extra>",
        )
    )
    trend.update_yaxes(tickprefix="$", tickformat=",")
    st.plotly_chart(style_figure(trend, 360), width="stretch")

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("#### Median Home Value by Metro")
        frame = summary.sort_values("latest_value")
        bars = px.bar(
            frame, x="latest_value", y="market", orientation="h", text="latest_value",
            labels={"latest_value": "Median price ($)", "market": ""},
            color="latest_value", color_continuous_scale=["#C7D2FE", config.COLORS["primary"]],
        )
        bars.update_traces(texttemplate="%{text:$,.0f}", textposition="outside", cliponaxis=False)
        bars.update_layout(coloraxis_showscale=False)
        bars.update_xaxes(tickprefix="$", tickformat=",")
        st.plotly_chart(style_figure(bars, 440), width="stretch")

    with col_right:
        st.markdown("#### Gross Rental Yield by Metro")
        frame = summary.dropna(subset=["gross_yield_pct"]).sort_values("gross_yield_pct")
        bars = px.bar(
            frame, x="gross_yield_pct", y="market", orientation="h", text="gross_yield_pct",
            labels={"gross_yield_pct": "Gross yield (%)", "market": ""},
            color="gross_yield_pct", color_continuous_scale=["#BAE6FD", config.COLORS["accent"]],
        )
        bars.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
        bars.update_layout(coloraxis_showscale=False)
        bars.update_xaxes(ticksuffix="%")
        st.plotly_chart(style_figure(bars, 440), width="stretch")

    st.markdown("#### Market Health by Metro")
    national_health = market_data.build_national_health(health)
    health_col_1, health_col_2 = st.columns(2)
    with health_col_1:
        st.markdown("##### Days to Pending")
        pivot = summary.dropna(subset=["days_to_pending"]).sort_values("days_to_pending")
        bars = px.bar(
            pivot, x="days_to_pending", y="market", orientation="h",
            labels={"days_to_pending": "Days", "market": ""},
            color="days_to_pending", color_continuous_scale=["#BBF7D0", config.COLORS["success"]],
        )
        bars.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_figure(bars, 380), width="stretch")
    with health_col_2:
        st.markdown("##### For-Sale Inventory")
        pivot = summary.dropna(subset=["inventory"]).sort_values("inventory")
        bars = px.bar(
            pivot, x="inventory", y="market", orientation="h",
            labels={"inventory": "Listings", "market": ""},
            color="inventory", color_continuous_scale=["#E0E7FF", config.COLORS["primary"]],
        )
        bars.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_figure(bars, 380), width="stretch")

    st.markdown("#### National Market Health Trend")
    if not national_health.empty:
        health_chart = go.Figure()
        for metric, color, axis in [
            ("inventory", config.COLORS["accent"], "y2"),
            ("days_to_pending", config.COLORS["primary"], "y"),
        ]:
            series = national_health[national_health["metric"] == metric]
            health_chart.add_trace(
                go.Scatter(
                    x=series["month"], y=series["value"], mode="lines", name=metric.replace("_", " ").title(),
                    line=dict(color=color, width=2.5), yaxis=axis,
                )
            )
        health_chart.update_layout(
            yaxis=dict(title="Days to pending"),
            yaxis2=dict(title="Inventory", overlaying="y", side="right", showgrid=False),
        )
        st.plotly_chart(style_figure(health_chart, 360), width="stretch")


def render_analytics_tab(df: pd.DataFrame, importance: pd.DataFrame) -> None:
    """Price-per-sqft distribution by metro and feature importance."""
    banded = get_age_banded_data(df)
    order = list(df.groupby("market")["price"].median().sort_values().index)

    st.markdown("#### Price per Sq Ft by Metro")
    st.caption("Each point is a listing, colour-coded by property age band.")
    scatter = px.scatter(
        banded, x="market", y="price_per_sqft", color="age_band",
        category_orders={"market": order, "age_band": config.AGE_LABELS},
        color_discrete_sequence=config.CHART_SEQUENCE, opacity=0.6,
        labels={"market": "", "price_per_sqft": "Price per sq ft ($)", "age_band": "Age band"},
        hover_data={"bedrooms": True, "bathrooms": True, "sqft": ":,", "price": ":,"},
    )
    scatter.update_traces(marker=dict(size=7, line=dict(width=0.5, color="white")))
    scatter.update_yaxes(tickprefix="$", tickformat=",")
    st.plotly_chart(style_figure(scatter, 460), width="stretch")

    col_left, col_right = st.columns([1.1, 1])
    with col_left:
        st.markdown("#### Median Listing Price by Metro")
        stats = get_market_stats(df).sort_values("median_price")
        bar = px.bar(
            stats, x="median_price", y="market", orientation="h", text="median_price",
            labels={"median_price": "Median price ($)", "market": ""},
            color="median_price", color_continuous_scale=["#C7D2FE", config.COLORS["primary"]],
        )
        bar.update_traces(texttemplate="%{text:$,.0f}", textposition="outside", cliponaxis=False)
        bar.update_layout(coloraxis_showscale=False)
        bar.update_xaxes(tickprefix="$", tickformat=",")
        st.plotly_chart(style_figure(bar, 420), width="stretch")

    with col_right:
        st.markdown("#### What Drives Home Prices")
        st.caption("Aggregated XGBoost gain importance.")
        imp = importance.sort_values("importance")
        imp_bar = px.bar(
            imp, x="importance", y="feature", orientation="h",
            labels={"importance": "Relative importance", "feature": ""},
            color="importance", color_continuous_scale=["#E0E7FF", config.COLORS["primary"]],
        )
        imp_bar.update_layout(coloraxis_showscale=False)
        imp_bar.update_traces(texttemplate="%{x:.3f}", textposition="outside", cliponaxis=False)
        st.plotly_chart(style_figure(imp_bar, 420), width="stretch")


def render_model_tab(model, metrics: dict, df: pd.DataFrame) -> None:
    """Model quality diagnostics: metrics, parity plot and residual spread."""
    model_name = metrics.get("model_name", "Regressor")

    col_1, col_2, col_3, col_4 = st.columns(4)
    with col_1:
        kpi_card("MAE", dollars(metrics.get("mae", 0)), "Avg. miss on test set")
    with col_2:
        kpi_card("RMSE", dollars(metrics.get("rmse", 0)), "Penalises large misses")
    with col_3:
        kpi_card("MAPE", f"{metrics.get('mape', 0):.2f}%", "Mean absolute % error", config.GLOSSARY["mape"])
    with col_4:
        kpi_card("R²", f"{metrics.get('r2', 0):.4f}", "Variance explained", config.GLOSSARY["r2"])

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
                x=y_test, y=y_pred, mode="markers",
                marker=dict(size=6, color=config.COLORS["primary"], opacity=0.4, line=dict(width=0)),
                name="Test listings",
                hovertemplate="Actual %{x:$,.0f}<br>Predicted %{y:$,.0f}<extra></extra>",
            )
        )
        limits = [float(min(y_test.min(), y_pred.min())), float(max(y_test.max(), y_pred.max()))]
        parity.add_trace(
            go.Scatter(x=limits, y=limits, mode="lines",
                       line=dict(color=config.COLORS["danger"], dash="dash", width=2),
                       name="Perfect prediction")
        )
        parity.update_xaxes(tickprefix="$", tickformat=",")
        parity.update_yaxes(tickprefix="$", tickformat=",")
        st.plotly_chart(style_figure(parity, 420), width="stretch")

    with col_right:
        st.markdown("#### Percentage Error Distribution")
        pct_error = (y_pred - y_test) / y_test * 100.0
        hist = go.Figure(
            go.Histogram(
                x=pct_error, nbinsx=45,
                marker=dict(color=config.COLORS["accent"], line=dict(width=0)),
                name="Error", hovertemplate="Error %{x:.1f}%<br>Count %{y}<extra></extra>",
            )
        )
        hist.add_vline(x=0, line_dash="dash", line_color=config.COLORS["danger"], annotation_text="Zero error")
        hist.update_xaxes(title="Percentage error (%)")
        hist.update_yaxes(title="Listings")
        st.plotly_chart(style_figure(hist, 420), width="stretch")

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


def render_data_tab(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Metro summary table and filtered listings with download support."""
    st.markdown("#### Metro Market Summary")
    table = summary.sort_values("latest_value", ascending=False)[
        ["market", "state", "latest_value", "yoy_pct", "gross_yield_pct", "days_to_pending", "inventory"]
    ].rename(
        columns={
            "market": "Metro", "state": "State", "latest_value": "Median Home Value",
            "yoy_pct": "YoY (%)", "gross_yield_pct": "Gross Yield (%)",
            "days_to_pending": "Days to Pending", "inventory": "Inventory",
        }
    )
    styled = table.copy()
    styled["Median Home Value"] = styled["Median Home Value"].map(dollars)
    styled["YoY (%)"] = styled["YoY (%)"].map(lambda v: f"{v:+.1f}%" if pd.notna(v) else "n/a")
    styled["Gross Yield (%)"] = styled["Gross Yield (%)"].map(lambda v: f"{v:.1f}%" if pd.notna(v) else "n/a")
    styled["Days to Pending"] = styled["Days to Pending"].map(lambda v: f"{v:.0f}" if pd.notna(v) else "n/a")
    styled["Inventory"] = styled["Inventory"].map(lambda v: f"{v:,.0f}" if pd.notna(v) else "n/a")
    st.dataframe(styled, width="stretch", hide_index=True)

    st.markdown("#### Listing Explorer")
    st.caption(f"{len(df):,} synthetic listings match the current filters.")
    display_columns = config.RAW_COLUMNS + ["price_per_sqft", "property_age"]
    st.dataframe(df[display_columns], width="stretch", height=380)

    st.download_button(
        "⬇️ Download filtered listings (CSV)",
        data=df[display_columns].to_csv(index=False).encode("utf-8"),
        file_name="filtered_listings.csv",
        mime="text/csv",
    )


# --------------------------------------------------------------------------- #
# International view
# --------------------------------------------------------------------------- #
def render_international_tab(intl: dict, country_code: str) -> None:
    """International market view: national index, trends and comparison."""
    summary = intl["summary"]
    bis = intl["bis"]
    source = intl["source"]
    row = summary[summary["country_code"] == country_code].iloc[0]
    name = international_data.country_name(country_code)

    temperature = insights.market_temperature(row["yoy_pct"], None, None)
    narrative = insights.build_market_narrative(
        name, yoy=row["yoy_pct"], change_5y=row["change_5y_pct"], temperature=temperature
    )
    render_summary_card(narrative, temperature)

    badge = "🟢 live" if source == "live" else "🟡 snapshot"
    st.markdown(f"#### {name} — national market")
    st.caption(
        f"BIS nominal house price index (2010 = 100) · quarterly · "
        f"latest {row['period']} · source: {badge}"
    )

    col_1, col_2, col_3, col_4 = st.columns(4)
    with col_1:
        kpi_card("House Price Index", f"{row['index']:.1f}", f"Latest {row['period']}", config.GLOSSARY["index"])
    with col_2:
        kpi_card("Year over Year", f"{row['yoy_pct']:+.1f}%", "Nominal, BIS", config.GLOSSARY["yoy"])
    with col_3:
        kpi_card("5-Year Change", f"{row['change_5y_pct']:+.1f}%", "Nominal index", config.GLOSSARY["change_5y"])
    with col_4:
        kpi_card("Market Temperature", temperature["label"], "From YoY change", config.GLOSSARY["temperature"])

    st.markdown("<br/>", unsafe_allow_html=True)

    tab_overview, tab_compare, *rest = st.tabs(
        ["📈 Overview", "🌍 All Countries"] + (["🇬🇧 UK Regions"] if country_code == "GB" else [])
    )
    with tab_overview:
        _render_country_overview(bis, country_code)
    with tab_compare:
        _render_comparison(intl["comparison"], summary)
    if country_code == "GB" and rest:
        with rest[0]:
            _render_uk_regions(intl["uk_regions"], intl["uk_summary"])


def _render_country_overview(bis: pd.DataFrame, country_code: str) -> None:
    """Index and YoY trend for a single country."""
    series = bis[bis["country_code"] == country_code].sort_values("period_date").tail(60).copy()

    st.markdown("#### House Price Index (last 15 years)")
    trend = go.Figure()
    trend.add_trace(
        go.Scatter(
            x=series["period_date"], y=series["index"], mode="lines",
            line=dict(color=config.COLORS["primary"], width=3),
            fill="tozeroy", fillcolor="rgba(79,70,229,0.08)",
            hovertemplate="%{x|%Y}<br>Index %{y:.1f}<extra></extra>",
        )
    )
    st.plotly_chart(style_figure(trend, 360), width="stretch")

    st.markdown("#### Year-over-Year Change")
    colors = [config.COLORS["success"] if v >= 0 else config.COLORS["danger"] for v in series["yoy_pct"]]
    yoy = go.Figure(go.Bar(x=series["period_date"], y=series["yoy_pct"], marker_color=colors))
    yoy.add_hline(y=0, line_color=config.COLORS["muted"], line_width=1)
    yoy.update_yaxes(ticksuffix="%")
    st.plotly_chart(style_figure(yoy, 320), width="stretch")


def _render_comparison(comparison: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Cross-country indexed growth and latest growth comparison."""
    st.markdown("#### Home Price Growth — Indexed (window start = 100)")
    st.caption("A curated set of countries indexed to 100 at the start of a common window.")
    growth = px.line(
        comparison, x="period_date", y="indexed", color="country",
        color_discrete_sequence=config.CHART_SEQUENCE,
        labels={"period_date": "", "indexed": "Index (start = 100)", "country": ""},
    )
    growth.update_traces(line=dict(width=2.4))
    st.plotly_chart(style_figure(growth, 460), width="stretch")

    st.markdown("#### Latest Year-over-Year Growth")
    frame = summary.sort_values("yoy_pct").tail(15)
    colors = [config.COLORS["success"] if v >= 0 else config.COLORS["danger"] for v in frame["yoy_pct"]]
    bars = go.Figure(
        go.Bar(
            x=frame["yoy_pct"], y=frame["country"], orientation="h",
            marker_color=colors, text=[f"{v:+.1f}%" for v in frame["yoy_pct"]],
            textposition="outside", cliponaxis=False,
            hovertemplate="%{y}<br>%{x:.2f}%<extra></extra>",
        )
    )
    bars.add_vline(x=0, line_color=config.COLORS["muted"], line_width=1)
    bars.update_xaxes(ticksuffix="%")
    st.plotly_chart(style_figure(bars, 460), width="stretch")


def _render_uk_regions(uk_regions: pd.DataFrame, uk_summary: pd.DataFrame) -> None:
    """UK nation prices, HPI and trends (HM Land Registry, GBP)."""
    st.markdown("#### Average House Price by UK Nation")
    st.caption("HM Land Registry UK House Price Index · monthly · GBP.")
    bars = px.bar(
        uk_summary.sort_values("avg_price_gbp"), x="avg_price_gbp", y="region",
        orientation="h", text="avg_price_gbp",
        labels={"avg_price_gbp": "Average price (£)", "region": ""},
        color="avg_price_gbp", color_continuous_scale=["#BAE6FD", config.COLORS["accent"]],
    )
    bars.update_traces(texttemplate="£%{text:,.0f}", textposition="outside", cliponaxis=False)
    bars.update_layout(coloraxis_showscale=False)
    bars.update_xaxes(tickprefix="£", tickformat=",")
    st.plotly_chart(style_figure(bars, 320), width="stretch")

    st.markdown("#### Average Price Trend")
    trend = px.line(
        uk_regions, x="month_date", y="avg_price_gbp", color="region",
        color_discrete_sequence=config.CHART_SEQUENCE,
        labels={"month_date": "", "avg_price_gbp": "Average price (£)", "region": ""},
    )
    trend.update_traces(line=dict(width=2.4))
    trend.update_yaxes(tickprefix="£", tickformat=",")
    st.plotly_chart(style_figure(trend, 420), width="stretch")

    table = uk_summary.rename(
        columns={
            "region": "Nation", "month": "Month", "avg_price_gbp": "Average Price (£)",
            "hpi": "HPI", "yoy_pct": "YoY (%)",
        }
    )
    table["Average Price (£)"] = table["Average Price (£)"].map(lambda v: f"£{v:,.0f}")
    table["YoY (%)"] = table["YoY (%)"].map(lambda v: f"{v:+.1f}%")
    st.dataframe(table, width="stretch", hide_index=True)


# --------------------------------------------------------------------------- #
# Dashboards
# --------------------------------------------------------------------------- #
def _render_us_dashboard(data, model, metrics, importance, hood_meta, force_refresh: bool) -> None:
    """The United States experience (national view + valuation)."""
    overview = get_market_overview(force_refresh)
    summary = overview["summary"]
    history = overview["history"]
    health = overview["health"]
    source = overview["source"]

    national = market_data.build_national_summary(summary)
    temperature = insights.market_temperature(
        national["yoy_pct"], national["days_to_pending"], None
    )
    narrative = insights.build_market_narrative(
        "the United States", yoy=national["yoy_pct"], change_5y=national["change_5y_pct"],
        days_to_pending=national["days_to_pending"], temperature=temperature,
    )

    st.markdown(
        """
        <div class="hero">
            <h1>🏙️ Real Estate Price Estimator &amp; Market Insights</h1>
            <p>National US market view · gradient-boosted valuation · plain-English insights</p>
            <span class="live-badge">Market data: Zillow Research (monthly ZHVI / ZORI)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_summary_card(narrative, temperature)
    render_prediction_tool(model, metrics, hood_meta)

    display = render_filters(data)

    kpi_1, kpi_2, kpi_3, kpi_4 = st.columns(4)
    with kpi_1:
        kpi_card("Median Home Value", dollars(national["latest_value"] or 0),
                 f"National median · {source}", config.GLOSSARY["zhvi"])
    with kpi_2:
        kpi_card("Year over Year", f"{national['yoy_pct']:+.1f}%" if national["yoy_pct"] is not None else "n/a",
                 "Across 15 metros", config.GLOSSARY["yoy"])
    with kpi_3:
        kpi_card("Model Accuracy (100 − MAPE)", f"{100 - metrics.get('mape', 0):.1f}%",
                 f"Mean absolute % error {metrics.get('mape', 0):.2f}%", config.GLOSSARY["mape"])
    with kpi_4:
        kpi_card("Model R² Score", f"{metrics.get('r2', 0):.3f}",
                 "Held-out variance explained", config.GLOSSARY["r2"])

    st.markdown("<br/>", unsafe_allow_html=True)

    render_valuation_explainer(model)

    tab_national, tab_analytics, tab_model, tab_data = st.tabs(
        ["🇺🇸 National", "📊 Market Analytics", "🤖 Model Insights", "🗂️ Data Explorer"]
    )
    with tab_national:
        render_us_national_tab(summary, history, health, national)
        render_glossary()
    with tab_analytics:
        if display.empty:
            st.warning("No listings match the current filters.")
        else:
            render_analytics_tab(display, importance)
    with tab_model:
        render_model_tab(model, metrics, data)
    with tab_data:
        render_data_tab(display, summary)

    st.markdown(
        f"""
        <div class="footer">
            {config.ZILLOW_ATTRIBUTION} · Listing-level estimates from a model trained on synthetic
            listings calibrated to real neighbourhood medians. Portfolio demo — not financial advice.
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_international_dashboard(country_code: str, force_refresh: bool) -> None:
    """The international (non-US) market-insights experience."""
    intl = get_international(force_refresh)
    name = international_data.country_name(country_code)

    st.markdown(
        f"""
        <div class="hero">
            <h1>🏙️ {name} — Market Insights</h1>
            <p>National house price index · quarterly · plain-English insights</p>
            <span class="live-badge">{config.BIS_ATTRIBUTION}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_international_sidebar(country_code)
    render_international_tab(intl, country_code)

    st.markdown(
        f"""
        <div class="footer">
            {config.BIS_ATTRIBUTION} · National, index-based data (not listing prices).
            Valuation is available for the United States.
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_css()

    data = get_data()
    model, metrics, importance = get_model()
    hood_meta = get_hood_meta()

    st.sidebar.markdown("### 📡 Market Data")
    force_refresh = st.sidebar.button(
        "🔄 Refresh market data",
        width="stretch",
        help="Re-fetch the latest market data now, bypassing the 24-hour cache.",
    )
    if force_refresh:
        get_market_overview.clear()
        get_international.clear()

    seed_estimator_from_query(hood_meta)

    country_code = render_country_selector()
    if country_code == "US":
        _render_us_dashboard(data, model, metrics, importance, hood_meta, force_refresh)
    else:
        _render_international_dashboard(country_code, force_refresh)


if __name__ == "__main__":
    main()

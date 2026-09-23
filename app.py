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

from urllib.parse import urlencode

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split

import config
import international_data
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
# Query-parameter state (shareable URLs)
# --------------------------------------------------------------------------- #
#: Query-param name -> session-state key for the estimator inputs.
_SHARE_PARAMS: dict[str, str] = {
    "market": "market_select",
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

#: Session-state key -> query-param name (inverse of :data:`_SHARE_PARAMS`).
_PARAM_FOR_KEY: dict[str, str] = {key: param for param, key in _SHARE_PARAMS.items()}


def seed_market_from_query(summary: pd.DataFrame) -> None:
    """Seed the market selector from ``?market=`` on first load."""
    if "market_select" in st.session_state:
        return
    names = list(summary.sort_values("size_rank")["market"])
    wanted = st.query_params.get("market")
    st.session_state["market_select"] = wanted if wanted in names else names[0]


def seed_inputs_from_query(hoods: pd.DataFrame, market_df: pd.DataFrame) -> None:
    """Seed location and property inputs from query params (once)."""
    options = [
        loc
        for loc in hoods.sort_values("latest_value", ascending=False)["location"]
        if loc in set(market_df["neighborhood"])
    ]
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
    for param, key in _SHARE_PARAMS.items():
        if key not in st.session_state:
            continue
        value = st.session_state[key]
        if param == "pool":
            st.query_params[param] = "1" if value else "0"
        else:
            st.query_params[param] = str(value)


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
    """Load live market values (cached; falls back to the committed snapshot)."""
    return market_data.get_market_data(force_refresh=force_refresh)


@st.cache_data(ttl=3600, show_spinner="Loading international data...")
def get_international(force_refresh: bool = False) -> dict:
    """Load international market data (BIS live-or-snapshot + UK regions)."""
    return international_data.get_country_data(force_refresh=force_refresh)


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
def render_market_selector(summary: pd.DataFrame, source: str, fetched_at) -> pd.Series:
    """Render the global market selector and return the selected market row."""
    st.sidebar.markdown("### 🌎 Market")
    options = summary.sort_values("size_rank")
    label_to_row = {row.market: row for row in options.itertuples(index=False)}
    choice = st.sidebar.selectbox(
        "Select a market",
        options=list(label_to_row),
        key="market_select",
        help="Values are live monthly Zillow Research data (ZHVI home values, ZORI rents).",
    )
    badge = "🟢 live" if source == "live" else "🟡 snapshot"
    stamp = fetched_at.strftime("%d %b %H:%M") if hasattr(fetched_at, "strftime") else "—"
    st.sidebar.caption(f"Source: {badge} · updated {stamp} UTC")
    return label_to_row[choice]


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
        help="United States: full metro/neighborhood analytics. Others: national BIS data.",
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
        "Live valuation and neighborhood analytics are available for the "
        "**United States** only (real listing-level data). For this country we "
        "show real national market data from the Bank for International "
        "Settlements (BIS) — switch back to the US to estimate a property."
    )


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
        neighborhood = st.selectbox("Location", options=locations, key="location_select")
        sqft = st.slider(
            "Living area (sq ft)",
            min_value=int(numeric_bounds["sqft"][0]),
            max_value=int(numeric_bounds["sqft"][1]),
            step=50,
            key="sqft_input",
        )
        col_a, col_b = st.columns(2)
        with col_a:
            bedrooms = st.number_input(
                "Bedrooms", min_value=1, max_value=6, step=1, key="beds_input"
            )
        with col_b:
            bathrooms = st.number_input(
                "Bathrooms", min_value=1.0, max_value=5.0, step=0.5, key="baths_input"
            )
        col_c, col_d = st.columns(2)
        with col_c:
            year_built = st.number_input(
                "Year built", min_value=1900, max_value=config.REFERENCE_YEAR,
                step=1, key="year_input",
            )
        with col_d:
            garage_spaces = st.number_input(
                "Garage spaces", min_value=0, max_value=3, step=1, key="garage_input"
            )
        lot_size = st.slider(
            "Lot size (sq ft)",
            min_value=int(numeric_bounds["lot_size"][0]),
            max_value=int(numeric_bounds["lot_size"][1]),
            step=250,
            key="lot_input",
        )
        has_pool = st.toggle("Swimming pool", key="pool_input")

        submitted = st.form_submit_button("Estimate Value", width="stretch", type="primary")

    if submitted:
        features = {
            "market": market_row.market,
            "neighborhood": neighborhood,
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

    sync_query_params()
    link = share_url()
    if link:
        with st.sidebar.expander("🔗 Shareable link"):
            st.code(link, language=None)
            st.caption("Copy this URL to restore this market, location and estimate.")

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
def render_international_tab(intl: dict, country_code: str) -> None:
    """International market view: national index, trends and comparison."""
    summary = intl["summary"]
    bis = intl["bis"]
    source = intl["source"]
    row = summary[summary["country_code"] == country_code].iloc[0]
    name = international_data.country_name(country_code)

    badge = "🟢 live" if source == "live" else "🟡 snapshot"
    st.markdown(f"#### {name} — national market")
    st.caption(
        f"BIS nominal house price index (2010 = 100) · quarterly · "
        f"latest {row['period']} · source: {badge}"
    )

    col_1, col_2, col_3, col_4 = st.columns(4)
    with col_1:
        kpi_card("House Price Index", f"{row['index']:.1f}", f"Latest {row['period']}")
    with col_2:
        kpi_card("Year over Year", f"{row['yoy_pct']:+.1f}%", "Nominal, BIS")
    with col_3:
        kpi_card("5-Year Change", f"{row['change_5y_pct']:+.1f}%", "Nominal index")
    with col_4:
        kpi_card("Coverage", "National", "Index-based (not prices)")

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
            hovertemplate="%{x|%Y-Q%q}<br>Index %{y:.1f}<extra></extra>",
        )
    )
    trend = style_figure(trend, height=360)
    st.plotly_chart(trend, width="stretch")

    st.markdown("#### Year-over-Year Change")
    colors = [config.COLORS["success"] if v >= 0 else config.COLORS["danger"] for v in series["yoy_pct"]]
    yoy = go.Figure(go.Bar(x=series["period_date"], y=series["yoy_pct"], marker_color=colors))
    yoy.add_hline(y=0, line_color=config.COLORS["muted"], line_width=1)
    yoy.update_yaxes(ticksuffix="%")
    yoy = style_figure(yoy, height=320)
    st.plotly_chart(yoy, width="stretch")


def _render_comparison(comparison: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Cross-country indexed growth and latest growth comparison."""
    st.markdown("#### Home Price Growth — Indexed (window start = 100)")
    st.caption("All six countries indexed to 100 at the start of a common window.")
    growth = px.line(
        comparison, x="period_date", y="indexed", color="country",
        color_discrete_sequence=config.CHART_SEQUENCE,
        labels={"period_date": "", "indexed": "Index (start = 100)", "country": ""},
    )
    growth.update_traces(line=dict(width=2.4))
    growth = style_figure(growth, height=460)
    st.plotly_chart(growth, width="stretch")

    st.markdown("#### Latest Year-over-Year Growth")
    frame = summary.sort_values("yoy_pct")
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
    bars = style_figure(bars, height=360)
    st.plotly_chart(bars, width="stretch")


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
    bars = style_figure(bars, height=320)
    st.plotly_chart(bars, width="stretch")

    st.markdown("#### Average Price Trend")
    trend = px.line(
        uk_regions, x="month_date", y="avg_price_gbp", color="region",
        color_discrete_sequence=config.CHART_SEQUENCE, labels={"month_date": "", "avg_price_gbp": "Average price (£)", "region": ""},
    )
    trend.update_traces(line=dict(width=2.4))
    trend.update_yaxes(tickprefix="£", tickformat=",")
    trend = style_figure(trend, height=420)
    st.plotly_chart(trend, width="stretch")

    table = uk_summary.rename(
        columns={
            "region": "Nation", "month": "Month", "avg_price_gbp": "Average Price (£)",
            "hpi": "HPI", "yoy_pct": "YoY (%)",
        }
    )
    table["Average Price (£)"] = table["Average Price (£)"].map(lambda v: f"£{v:,.0f}")
    table["YoY (%)"] = table["YoY (%)"].map(lambda v: f"{v:+.1f}%")
    st.dataframe(table, width="stretch", hide_index=True)


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

    rent_1, rent_2, rent_3, rent_4 = st.columns(4)
    with rent_1:
        if pd.notna(market_row.latest_rent):
            kpi_card("Median Rent", f"${market_row.latest_rent:,.0f}/mo", f"{market_row.rent_yoy_pct:+.1f}% YoY (ZORI)")
        else:
            kpi_card("Median Rent", "n/a", "Rent data unavailable")
    with rent_2:
        if pd.notna(market_row.gross_yield_pct):
            kpi_card("Gross Rental Yield", f"{market_row.gross_yield_pct:.1f}%", "Annual rent ÷ home value")
        else:
            kpi_card("Gross Rental Yield", "n/a", "Rent data unavailable")

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

    st.markdown("#### Gross Rental Yield — All Markets")
    st.caption("Annual median rent (ZORI) as a percentage of median home value (ZHVI).")
    yield_frame = summary.dropna(subset=["gross_yield_pct"]).sort_values("gross_yield_pct")
    if not yield_frame.empty:
        yield_colors = [
            config.COLORS["accent"]
            if mid == market_row.market_id
            else "#BAE6FD"
            for mid in yield_frame["market_id"]
        ]
        yield_chart = go.Figure(
            go.Bar(
                x=yield_frame["gross_yield_pct"],
                y=yield_frame["market"],
                orientation="h",
                marker_color=yield_colors,
                text=[f"{v:.1f}%" for v in yield_frame["gross_yield_pct"]],
                textposition="outside",
                cliponaxis=False,
                hovertemplate="%{y}<br>%{x:.2f}%<extra></extra>",
            )
        )
        yield_chart.update_xaxes(ticksuffix="%")
        yield_chart = style_figure(yield_chart, height=460)
        st.plotly_chart(yield_chart, width="stretch")

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
def _render_us_dashboard(data, model, metrics, importance, hood_meta, force_refresh: bool) -> None:
    """The full United States experience (markets, analytics, valuation)."""
    overview = get_market_overview(force_refresh)
    summary = overview["summary"]
    history = overview["history"]
    source = overview["source"]
    fetched_at = overview["fetched_at"]

    st.markdown(
        """
        <div class="hero">
            <h1>🏙️ Real Estate Price Estimator &amp; Market Insights</h1>
            <p>Live US market data · gradient-boosted valuation · interactive analytics</p>
            <span class="live-badge">Market values &amp; rents: Zillow Research (monthly ZHVI / ZORI)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Sidebar: market + scoped filters + estimator --------------------- #
    seed_market_from_query(summary)
    market_row = render_market_selector(summary, source, fetched_at)
    market = market_row.market
    market_df = data[data["market"] == market]
    hoods = hood_meta[hood_meta["market_id"] == market_row.market_id]
    seed_inputs_from_query(hoods, market_df)

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


def _render_international_dashboard(country_code: str, force_refresh: bool) -> None:
    """The international (non-US) market-insights experience."""
    intl = get_international(force_refresh)
    name = international_data.country_name(country_code)

    st.markdown(
        f"""
        <div class="hero">
            <h1>🏙️ {name} — Market Insights</h1>
            <p>National house price index · quarterly · interactive comparison</p>
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
            Valuation and neighborhood analytics are available for the United States.
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_css()

    data = get_data()
    model, metrics, importance = get_model()
    hood_meta = get_hood_meta()

    # --- Sidebar: market data refresh ------------------------------------- #
    st.sidebar.markdown("### 📡 Market Data")
    force_refresh = st.sidebar.button(
        "🔄 Refresh market data",
        width="stretch",
        help="Re-fetch the latest market data now, bypassing the 24-hour cache.",
    )
    if force_refresh:
        get_market_overview.clear()
        get_international.clear()

    country_code = render_country_selector()
    if country_code == "US":
        _render_us_dashboard(data, model, metrics, importance, hood_meta, force_refresh)
    else:
        _render_international_dashboard(country_code, force_refresh)


if __name__ == "__main__":
    main()

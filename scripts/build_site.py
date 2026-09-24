"""
scripts/build_site.py
=====================
Generate the static GitHub Pages landing page in ``docs/``.

The page embeds real, interactive Plotly figures built from live/committed
Zillow market data and the trained pipeline, so it stays in sync with the
project. The full interactive app runs locally or via Docker (see the page).

Usage
-----
    python scripts/build_site.py
"""

from __future__ import annotations

import html
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from property_insights import (  # noqa: E402
    config,  # noqa: E402
    insights,  # noqa: E402
    international_data,  # noqa: E402
    market_data,  # noqa: E402
)
from property_insights.data_loader import load_or_create_data  # noqa: E402
from property_insights.model import ensure_model  # noqa: E402

REPO_URL = "https://github.com/armand-vw/Property-Price-Market-Dashboard"
PAGES_URL = "https://armand-vw.github.io/Property-Price-Market-Dashboard/"

DOCS_DIR = config.BASE_DIR / "docs"
INDEX_PATH = DOCS_DIR / "index.html"

FONT_FAMILY = "Inter, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"


def plotly_js_cdn() -> str:
    """Return a CDN URL for the plotly.js build bundled with this package."""
    try:
        from plotly.offline.offline import get_plotlyjs_version

        version = get_plotlyjs_version()
    except Exception:  # pragma: no cover - defensive
        version = "2.35.2"
    return f"https://cdn.plot.ly/plotly-{version}.min.js"


def money(value: float) -> str:
    """Compact currency formatting (``$1.23M`` / ``$664.0K``)."""
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:,.2f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:,.1f}K"
    return f"${value:,.0f}"


def dollars(value: float) -> str:
    return f"${value:,.0f}"


def style_fig(fig: go.Figure, height: int = 420) -> go.Figure:
    """Apply the shared light-corporate theme."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        font=dict(family=FONT_FAMILY, size=13, color=config.COLORS["ink"]),
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        colorway=config.CHART_SEQUENCE,
    )
    fig.update_xaxes(showgrid=False, linecolor=config.COLORS["border"])
    fig.update_yaxes(gridcolor=config.COLORS["border"], zeroline=False)
    return fig


def fig_to_html(fig: go.Figure) -> str:
    """Serialize a figure to an embeddable HTML fragment."""
    return fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        config={"displayModeBar": False, "responsive": True, "displaylogo": False},
    )


# --------------------------------------------------------------------------- #
# Figure builders
# --------------------------------------------------------------------------- #
def build_market_values(summary: pd.DataFrame) -> str:
    """Horizontal bar of the latest median home value across all markets."""
    frame = summary.sort_values("latest_value")
    fig = px.bar(
        frame,
        x="latest_value",
        y="market",
        orientation="h",
        text="latest_value",
        labels={"latest_value": "Median home value ($)", "market": ""},
        color="latest_value",
        color_continuous_scale=["#C7D2FE", config.COLORS["primary"]],
    )
    fig.update_traces(texttemplate="%{text:$,.0f}", textposition="outside", cliponaxis=False)
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(tickprefix="$", tickformat=",")
    return fig_to_html(style_fig(fig, 520))


def build_market_growth(history: pd.DataFrame, summary: pd.DataFrame, top_n: int = 6) -> str:
    """10-year indexed home-value growth for the largest markets."""
    top_markets = summary.sort_values("size_rank").head(top_n)
    fig = go.Figure()
    for color, row in zip(config.CHART_SEQUENCE, top_markets.itertuples(), strict=False):
        series = history[history["market_id"] == row.market_id].sort_values("month")
        series = series[series["month"] >= series["month"].max() - pd.DateOffset(years=10)]
        if series.empty:
            continue
        indexed = series["value"] / series["value"].iloc[0] * 100.0
        fig.add_trace(
            go.Scatter(
                x=series["month"],
                y=indexed,
                mode="lines",
                name=row.market.split(",")[0],
                line=dict(width=2.5, color=color),
                hovertemplate="%{x|%b %Y}<br>Index %{y:.1f}<extra></extra>",
            )
        )
    fig.update_yaxes(title="Index (=100 ten years ago)")
    return fig_to_html(style_fig(fig, 460))


def build_rental_yield(summary: pd.DataFrame) -> str:
    """Horizontal bar of gross rental yield across all markets."""
    frame = summary.dropna(subset=["gross_yield_pct"]).sort_values("gross_yield_pct")
    fig = px.bar(
        frame,
        x="gross_yield_pct",
        y="market",
        orientation="h",
        text="gross_yield_pct",
        labels={"gross_yield_pct": "Gross rental yield (%)", "market": ""},
        color="gross_yield_pct",
        color_continuous_scale=["#BAE6FD", config.COLORS["accent"]],
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(ticksuffix="%")
    return fig_to_html(style_fig(fig, 520))


def build_country_growth(comparison: pd.DataFrame) -> str:
    """Cross-country home-price growth, indexed to a common base."""
    fig = px.line(
        comparison,
        x="period_date",
        y="indexed",
        color="country",
        color_discrete_sequence=config.CHART_SEQUENCE,
        labels={"period_date": "", "indexed": "Index (window start = 100)", "country": ""},
    )
    fig.update_traces(line=dict(width=2.4))
    return fig_to_html(style_fig(fig, 500))


def _num(value) -> float | None:
    """Return a JSON-safe float (``None`` for NaN/inf)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _series(rows: pd.DataFrame, date_col: str, value_col: str) -> list[list]:
    """Convert two columns into ``[[date, value], ...]`` for the browser."""
    return [
        [date.strftime("%Y-%m-%d"), _num(value)]
        for date, value in zip(rows[date_col], rows[value_col], strict=False)
        if _num(value) is not None
    ]


def _build_sidebar_payload(
    summary: pd.DataFrame,
    history: pd.DataFrame,
    bis: pd.DataFrame,
    country_summary: pd.DataFrame,
    health: pd.DataFrame,
) -> dict[str, str]:
    """Serialise the data, education copy and options for the page sidebar."""
    countries: list[dict] = []
    for code in config.COUNTRY_ORDER:
        group = bis[bis["country_code"] == code].sort_values("period_date")
        row = country_summary[country_summary["country_code"] == code]
        if group.empty or row.empty:
            continue
        group = group.tail(60)
        latest = row.iloc[0]
        name = config.COUNTRIES[code]["name"]
        temperature = insights.market_temperature(latest["yoy_pct"], None, None)
        narrative = insights.build_market_narrative(
            name, yoy=latest["yoy_pct"], change_5y=latest["change_5y_pct"], temperature=temperature
        )
        countries.append(
            {
                "code": code,
                "name": name,
                "period": str(latest["period"]),
                "index": _num(latest["index"]),
                "yoy": _num(latest["yoy_pct"]),
                "change": _num(latest["change_5y_pct"]),
                "source": config.COUNTRIES[code]["source"],
                "temperature": temperature,
                "narrative": narrative,
                "index_history": _series(group, "period_date", "index"),
                "yoy_history": _series(group, "period_date", "yoy_pct"),
            }
        )

    national = market_data.build_national_summary(summary)
    national_history = market_data.build_national_history(history)
    national_history["yoy"] = national_history["value"] / national_history["value"].shift(12) * 100.0 - 100.0
    national_health = market_data.build_national_health(health)
    us_row = country_summary[country_summary["country_code"] == "US"]
    us_group = bis[bis["country_code"] == "US"].sort_values("period_date").tail(60)
    temperature = insights.market_temperature(
        national["yoy_pct"], national["days_to_pending"], None
    )
    narrative = insights.build_market_narrative(
        "the United States", yoy=national["yoy_pct"], change_5y=national["change_5y_pct"],
        days_to_pending=national["days_to_pending"], temperature=temperature,
    )
    us_national = {
        "name": "United States",
        "period": str(us_row.iloc[0]["period"]) if not us_row.empty else "",
        "index": _num(us_row.iloc[0]["index"]) if not us_row.empty else None,
        "index_history": _series(us_group, "period_date", "index"),
        "median_value": _num(national["latest_value"]),
        "rent": _num(national["latest_rent"]),
        "yield": _num(national["gross_yield_pct"]),
        "median_sale_price": _num(national["median_sale_price"]),
        "yoy": _num(national["yoy_pct"]),
        "change_5y": _num(national["change_5y_pct"]),
        "days_to_pending": _num(national["days_to_pending"]),
        "inventory": _num(national["inventory"]),
        "temperature": temperature,
        "narrative": narrative,
        "value_history": _series(national_history, "month", "value"),
        "value_yoy_history": _series(national_history, "month", "yoy"),
        "health_history": {
            "inventory": _series(national_health[national_health["metric"] == "inventory"], "month", "value"),
            "days_to_pending": _series(national_health[national_health["metric"] == "days_to_pending"], "month", "value"),
        },
    }

    country_options = "".join(
        f'<option value="{item["code"]}">{html.escape(item["name"])}</option>'
        for item in countries
    )
    glossary = "".join(
        f'<div class="glossary-item"><strong>{html.escape(term.replace("_", " ").title())}</strong>'
        f'<span>{html.escape(definition)}</span></div>'
        for term, definition in insights.glossary().items()
    )
    return {
        "json": json.dumps({"countries": countries, "us": us_national}),
        "country_options": country_options,
        "glossary": glossary,
    }


def build_feature_importance(importance: pd.DataFrame) -> str:
    """Horizontal bar chart of aggregated feature importances."""
    frame = importance.sort_values("importance")
    fig = px.bar(
        frame,
        x="importance",
        y="feature",
        orientation="h",
        labels={"importance": "Relative importance", "feature": ""},
        color="importance",
        color_continuous_scale=["#E0E7FF", config.COLORS["primary"]],
    )
    fig.update_layout(coloraxis_showscale=False)
    fig.update_traces(texttemplate="%{x:.3f}", textposition="outside", cliponaxis=False)
    return fig_to_html(style_fig(fig, 460))


# --------------------------------------------------------------------------- #
# Page template
# --------------------------------------------------------------------------- #
TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Real Estate Price Estimator &amp; Market Insights Dashboard</title>
<meta name="description" content="Interactive real-estate price estimator and market insights dashboard with live Zillow market data, Streamlit, XGBoost, scikit-learn and Plotly." />
<meta property="og:title" content="Real Estate Price Estimator &amp; Market Insights" />
<meta property="og:description" content="Live US market data, six-plus international markets, gradient-boosted valuation." />
<meta property="og:type" content="website" />
<meta property="og:image" content="__PAGES_URL__hero.png" />
<meta name="twitter:card" content="summary_large_image" />
<meta name="twitter:image" content="__PAGES_URL__hero.png" />
<link rel="stylesheet" href="style.css?v=__ASSET_VERSION__" />
<script src="__PLOTLY_CDN__"></script>
</head>
<body>

<header class="nav">
  <div class="nav-inner">
    <span class="brand">🏙️ Property&nbsp;Insights</span>
    <nav>
      <a href="#dashboard">Explorer</a>
      <a href="#preview">Preview</a>
      <a href="#model">Model</a>
      <a href="#stack">Stack</a>
      <a class="btn btn-ghost" href="__REPO_URL__" target="_blank" rel="noopener">GitHub ↗</a>
    </nav>
  </div>
</header>

<section class="hero">
  <div class="hero-inner">
    <span class="eyebrow">Machine Learning · Real Estate Analytics</span>
    <h1>Real Estate Price Estimator &amp;<br />Market Insights Dashboard</h1>
    <p class="lede">
      An end-to-end, production-style application combining <strong>live US market data</strong>
      from Zillow Research and <strong>international data</strong> from the Bank for
      International Settlements. Explore six countries, real neighbourhood values and trends,
      and estimate US property prices.
    </p>
    <div class="cta-row">
      <a class="btn btn-primary" href="#run" target="_self">▶ Run the app</a>
      <a class="btn btn-outline" href="__REPO_URL__" target="_blank" rel="noopener">View source</a>
    </div>
    <p class="cta-note">Select a market or country from the dropdowns below · run the full app locally or with Docker.</p>
  </div>
</section>

<section class="kpis">
  <div class="kpi"><span class="kpi-label">Markets Covered</span><span class="kpi-value">__MARKETS__</span><span class="kpi-sub">Largest US metros</span></div>
  <div class="kpi"><span class="kpi-label">Median Metro Value</span><span class="kpi-value">__MEDIAN_PRICE__</span><span class="kpi-sub">Live Zillow ZHVI</span></div>
  <div class="kpi"><span class="kpi-label">Model Accuracy</span><span class="kpi-value">__ACCURACY__</span><span class="kpi-sub">100 − MAPE (__MAPE__)</span></div>
  <div class="kpi"><span class="kpi-label">R² Score</span><span class="kpi-value">__R2__</span><span class="kpi-sub">Held-out variance explained</span></div>
</section>

<section id="dashboard" class="app-shell">
    <aside class="app-sidebar">
      <div class="sidebar-brand">🏙️ Property Insights</div>
      <div class="sidebar-hint">Choose a country to update the dashboard.</div>
      <label for="countrySelect">Country</label>
      <select id="countrySelect" aria-label="Country">__COUNTRY_OPTIONS__</select>
      <p class="explorer-note">United States: national Zillow data plus a location-based valuation app. Other countries: BIS national house-price index. Run the full app for live valuation.</p>
      <div class="sidebar-links">
        <a class="btn-sidebar" href="#stack">▶ Run the full app</a>
        <a class="btn-sidebar" href="__REPO_URL__" target="_blank" rel="noopener">GitHub ↗</a>
      </div>
    </aside>
  <main class="app-main">
    <div class="app-head">
      <h2 id="dashTitle">United States</h2>
      <span id="dashSub" class="muted"></span>
    </div>
    <div class="summary-card">
      <div class="summary-label">What this means</div>
      <p id="summaryText">Loading…</p>
      <span id="tempBadge" class="temp-badge temp-warm">—</span>
    </div>
    <div class="kpis-row">
      <div class="mini-kpi"><span id="k1l">Metric</span><strong id="k1v">—</strong><span id="k1s"></span></div>
      <div class="mini-kpi"><span id="k2l">Metric</span><strong id="k2v">—</strong><span id="k2s"></span></div>
      <div class="mini-kpi"><span id="k3l">Metric</span><strong id="k3v">—</strong><span id="k3s"></span></div>
      <div class="mini-kpi"><span id="k4l">Metric</span><strong id="k4v">—</strong><span id="k4s"></span></div>
    </div>
    <div id="healthWrap">
      <div class="kpis-row">
        <div class="mini-kpi"><span>Days to Pending</span><strong id="h1v">—</strong><span>Median, latest month</span></div>
        <div class="mini-kpi"><span>For-Sale Inventory</span><strong id="h2v">—</strong><span>Active listings</span></div>
        <div class="mini-kpi"><span>Median Sale Price</span><strong id="h3v">—</strong><span>Closed sales</span></div>
      </div>
      <div class="chart-card"><h3>Market Health (US)</h3><p class="muted">For-sale inventory and median days to pending for the selected metro.</p><div id="explorerHealth"></div></div>
    </div>
    <div class="chart-card"><div id="explorerChart"></div></div>
    <div class="chart-card"><h3>Year-over-Year Change</h3><p class="muted">Annual change for the selected market / country.</p><div id="explorerYoy"></div></div>
    <div class="chart-card">
      <h3>📚 Market 101 — what these numbers mean</h3>
      <div class="glossary-grid">__GLOSSARY__</div>
    </div>
  </main>
</section>

<section class="features">
  <h2>What it does</h2>
  <div class="grid grid-3">
    <div class="card"><div class="icon">🌎</div><h3>Live market data</h3><p>Real median home values, rents and trends for the 15 largest US metros, fetched from Zillow Research and refreshed automatically.</p></div>
    <div class="card"><div class="icon">📍</div><h3>Real neighborhoods</h3><p>Explore actual neighbourhood home values within each market, with a committed snapshot for offline reliability.</p></div>
    <div class="card"><div class="icon">🎯</div><h3>Live valuation</h3><p>Configure a property and get an instant estimate with an empirical valuation range and neighbourhood comparison.</p></div>
    <div class="card"><div class="icon">🧠</div><h3>Explainable model</h3><p>XGBoost with aggregated feature importances, predicted-vs-actual parity and residual diagnostics.</p></div>
    <div class="card"><div class="icon">🧱</div><h3>Leak-free pipeline</h3><p>Imputation, scaling and one-hot encoding fitted inside an sklearn pipeline — never on the test set.</p></div>
    <div class="card"><div class="icon">⚙️</div><h3>Production ready</h3><p>Joblib artifact, live-to-snapshot fallback, pytest suite and CI on every push.</p></div>
  </div>
</section>

<section id="preview" class="section">
  <h2>Interactive preview</h2>
  <p class="section-lede">Rendered from real Zillow market data and the trained model at build time.</p>
  <div class="chart-card"><h3>Median Home Value by Market</h3><p class="muted">Latest published month across the 15 largest US metros.</p><div class="chart">__CHART_VALUES__</div></div>
  <div class="grid grid-2">
    <div class="chart-card"><h3>10-Year Value Growth</h3><p class="muted">Home values indexed to 100 ten years ago — the biggest metros compared.</p><div class="chart">__CHART_GROWTH__</div></div>
    <div class="chart-card"><h3>Gross Rental Yield by Market</h3><p class="muted">Annual median rent (ZORI) as a percentage of median home value (ZHVI).</p><div class="chart">__CHART_YIELD__</div></div>
  </div>
  <div class="chart-card"><h3>International Home Price Growth</h3><p class="muted">BIS nominal house price index for six countries, indexed to 100 at the window start.</p><div class="chart">__CHART_COUNTRIES__</div></div>
  <div class="chart-card"><h3>What Drives Home Prices</h3><p class="muted">Aggregated XGBoost gain importance (US model).</p><div class="chart">__CHART_IMPORTANCE__</div></div>
</section>

<section id="model" class="section">
  <h2>Model performance</h2>
  <p class="section-lede">80/20 hold-out evaluation, trained on the <code>log1p</code> target and scored in dollars.</p>
  <div class="table-wrap">
    <table>
      <tbody>
        <tr><th>Estimator</th><td>__MODEL_NAME__</td></tr>
        <tr><th>Training listings</th><td>__TRAIN_TEST__</td></tr>
        <tr><th>Markets / locations</th><td>__MARKETS__ / __LOCATIONS__</td></tr>
        <tr><th>MAE</th><td><strong>__MAE__</strong></td></tr>
        <tr><th>RMSE</th><td><strong>__RMSE__</strong></td></tr>
        <tr><th>MAPE</th><td><strong>__MAPE__</strong></td></tr>
        <tr><th>R² score</th><td><strong>__R2__</strong></td></tr>
        <tr><th>Improvement vs. median baseline</th><td>__BASELINE__</td></tr>
        <tr><th>Top price drivers</th><td>__TOP_DRIVERS__</td></tr>
      </tbody>
    </table>
  </div>
</section>

<section id="stack" class="section">
  <h2>Tech stack &amp; data</h2>
  <div class="pills">
    <span class="pill">Python 3.12</span>
    <span class="pill">Streamlit</span>
    <span class="pill">XGBoost</span>
    <span class="pill">scikit-learn</span>
    <span class="pill">Plotly</span>
    <span class="pill">pandas</span>
    <span class="pill">Zillow Research</span>
    <span class="pill">BIS</span>
    <span class="pill">HM Land Registry</span>
    <span class="pill">pytest</span>
    <span class="pill">Docker</span>
    <span class="pill">GitHub Actions</span>
  </div>
  <p class="section-lede" style="margin-top:16px">
    Market values: Zillow Research ZHVI (monthly, latest published month). Listing-level
    features are synthesised and calibrated to real neighbourhood medians.
  </p>
  <div class="code-card" id="run">
    <div class="code-title">Run locally</div>
    <pre><code>git clone __REPO_URL__.git
cd Property-Price-Market-Dashboard
python3 -m venv .venv &amp;&amp; source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py</code></pre>
  </div>
  <div class="code-card">
    <div class="code-title">Run with Docker (self-contained, no external hosting)</div>
    <pre><code>docker build -t property-insights .
docker run --rm -p 8501:8501 property-insights
# fully offline (committed snapshot, no outbound calls):
docker run --rm -p 8501:8501 -e RPE_OFFLINE=1 property-insights</code></pre>
  </div>
</section>

<footer class="footer">
  <p>Built with Streamlit, Plotly, XGBoost &amp; scikit-learn · Market data: Zillow Research. Estimates are illustrative and not financial advice.</p>
  <p><a href="__REPO_URL__" target="_blank" rel="noopener">Source on GitHub</a> · Generated __GENERATED_AT__ · build __BUILD_STAMP__</p>
</footer>

<script>
(function () {
  if (typeof Plotly === "undefined") { return; }
  var DATA = __SIDEBAR_DATA__;
  var countrySel = document.getElementById("countrySelect");
  var healthWrap = document.getElementById("healthWrap");
  var titleEl = document.getElementById("dashTitle");
  var subEl = document.getElementById("dashSub");
  var summaryEl = document.getElementById("summaryText");
  var tempEl = document.getElementById("tempBadge");
  var trendEl = document.getElementById("explorerChart");
  var yoyEl = document.getElementById("explorerYoy");
  var healthEl = document.getElementById("explorerHealth");

  function setText(id, value) { var el = document.getElementById(id); if (el) { el.textContent = value; } }
  function setKpi(i, label, value, sub) {
    setText("k" + i + "l", label);
    setText("k" + i + "v", value);
    setText("k" + i + "s", sub || "");
  }
  function isNum(v) { return v !== null && v !== undefined && !isNaN(v); }
  function money(v) {
    if (!isNum(v)) { return "n/a"; }
    if (v >= 1e6) { return "$" + (v / 1e6).toFixed(2) + "M"; }
    if (v >= 1e3) { return "$" + Math.round(v / 1e3) + "K"; }
    return "$" + Math.round(v);
  }
  function pct(v) { return isNum(v) ? (v >= 0 ? "+" : "") + v.toFixed(1) + "%" : "n/a"; }
  function setTemp(t) {
    if (!tempEl || !t || !t.label) { return; }
    tempEl.textContent = t.label + " market";
    tempEl.className = "temp-badge " + (t.label === "Hot" ? "temp-hot" : (t.label === "Cool" ? "temp-cool" : "temp-warm"));
  }
  function layout(title, isMoney) {
    return {
      title: { text: title, font: { size: 16 } },
      template: "plotly_white", height: 360,
      margin: { l: 60, r: 20, t: 50, b: 40 },
      yaxis: { tickprefix: isMoney ? "$" : "", tickformat: ",", gridcolor: "#E2E8F0" },
      xaxis: { showgrid: false },
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)"
    };
  }
  function drawTrend(series, title, isMoney) {
    var x = series.map(function (p) { return p[0]; });
    var y = series.map(function (p) { return p[1]; });
    Plotly.react(trendEl, [{
      x: x, y: y, mode: "lines",
      line: { color: "#4F46E5", width: 3 },
      fill: "tozeroy", fillcolor: "rgba(79,70,229,0.08)",
      hovertemplate: (isMoney ? "%{x|%b %Y}<br>$%{y:,.0f}" : "%{x|%Y}<br>%{y:.1f}") + "<extra></extra>"
    }], layout(title, isMoney), { displayModeBar: false, responsive: true });
  }
  function drawYoy(series, title) {
    var x = series.map(function (p) { return p[0]; });
    var y = series.map(function (p) { return p[1]; });
    Plotly.react(yoyEl, [{
      x: x, y: y, type: "bar",
      marker: { color: y.map(function (v) { return v >= 0 ? "#059669" : "#DC2626"; }) },
      hovertemplate: "%{x|%b %Y}<br>%{y:.1f}%<extra></extra>"
    }], layout(title, false), { displayModeBar: false, responsive: true });
  }
  function drawHealth(health, name) {
    var inv = (health && health.inventory) || [];
    var doz = (health && health.days_to_pending) || [];
    Plotly.react(healthEl, [
      {
        x: inv.map(function (p) { return p[0]; }), y: inv.map(function (p) { return p[1]; }),
        mode: "lines", name: "For-sale inventory",
        line: { color: "#0EA5E9", width: 2.5 }, yaxis: "y",
        hovertemplate: "%{x|%b %Y}<br>%{y:,.0f}<extra>Inventory</extra>"
      },
      {
        x: doz.map(function (p) { return p[0]; }), y: doz.map(function (p) { return p[1]; }),
        mode: "lines", name: "Days to pending",
        line: { color: "#4F46E5", width: 2.5 }, yaxis: "y2",
        hovertemplate: "%{x|%b %Y}<br>%{y:.0f} days<extra>Pending</extra>"
      }
    ], {
      title: { text: name + " — market health", font: { size: 15 } },
      template: "plotly_white", height: 340,
      margin: { l: 60, r: 60, t: 50, b: 40 },
      legend: { orientation: "h", y: 1.12, x: 0 },
      yaxis: { title: "Inventory", tickformat: ",", gridcolor: "#E2E8F0" },
      yaxis2: { title: "Days to pending", overlaying: "y", side: "right", showgrid: false },
      xaxis: { showgrid: false },
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)"
    }, { displayModeBar: false, responsive: true });
  }
  function renderUs() {
    var us = DATA.us;
    if (healthWrap) { healthWrap.style.display = "block"; }
    titleEl.textContent = "United States";
    subEl.textContent = "National view · median of 15 metros · Zillow Research";
    setText("summaryText", us.narrative);
    setTemp(us.temperature);
    setKpi(1, "Median Home Value", money(us.median_value), "National median");
    setKpi(2, "Year over Year", pct(us.yoy), "Across 15 metros");
    setKpi(3, "Gross Rental Yield", isNum(us.yield) ? us.yield.toFixed(1) + "%" : "n/a", "Annual rent ÷ value");
    setKpi(4, "Median Rent", isNum(us.rent) ? money(us.rent) + "/mo" : "n/a", "ZORI");
    setText("h1v", isNum(us.days_to_pending) ? Math.round(us.days_to_pending).toString() : "n/a");
    setText("h2v", isNum(us.inventory) ? Math.round(us.inventory).toLocaleString() : "n/a");
    setText("h3v", money(us.median_sale_price));
    drawTrend(us.value_history, "United States — median home value", true);
    drawYoy(us.value_yoy_history, "United States — year-over-year change");
    drawHealth(us.health_history, "United States");
  }
  function renderCountry() {
    var code = countrySel.value;
    if (code === "US") { renderUs(); return; }
    if (healthWrap) { healthWrap.style.display = "none"; }
    var c = DATA.countries.filter(function (x) { return x.code === code; })[0];
    if (!c) { return; }
    titleEl.textContent = c.name;
    subEl.textContent = "BIS nominal index (2010 = 100) · quarterly · latest " + c.period;
    setText("summaryText", c.narrative);
    setTemp(c.temperature);
    setKpi(1, "House Price Index", isNum(c.index) ? c.index.toFixed(1) : "n/a", "Latest " + c.period);
    setKpi(2, "Year over Year", pct(c.yoy), "BIS nominal");
    setKpi(3, "5-Year Change", pct(c.change), "Nominal index");
    setKpi(4, "Market Temperature", c.temperature ? c.temperature.label : "n/a", "From YoY change");
    drawTrend(c.index_history, c.name + " — house price index (2010 = 100)", false);
    drawYoy(c.yoy_history, c.name + " — year-over-year change");
  }
  function readHash() {
    var out = {};
    var raw = (location.hash || "").replace(/^#/, "");
    raw.split("&").forEach(function (pair) {
      var kv = pair.split("=");
      if (kv[0]) { out[decodeURIComponent(kv[0])] = decodeURIComponent(kv[1] || ""); }
    });
    return out;
  }
  function writeHash() {
    var hash = "#country=" + encodeURIComponent(countrySel.value);
    try { history.replaceState(null, "", hash); } catch (err) { location.hash = hash; }
  }
  function applyHash() {
    var params = readHash();
    if (params.country && DATA.countries.some(function (c) { return c.code === params.country; })) {
      countrySel.value = params.country;
    }
  }

  applyHash();
  countrySel.addEventListener("change", function () { renderCountry(); writeHash(); });
  renderCountry();
  writeHash();
})();
</script>

</body>
</html>
"""


def render_page(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    history: pd.DataFrame,
    comparison: pd.DataFrame,
    bis: pd.DataFrame,
    country_summary: pd.DataFrame,
    health: pd.DataFrame,
    metrics: dict,
    importance: pd.DataFrame,
) -> str:
    """Populate the template with data, metrics and chart fragments."""
    top_drivers = ", ".join(importance.head(4)["feature"].tolist())
    median_metro = summary["latest_value"].median()
    sidebar = _build_sidebar_payload(summary, history, bis, country_summary, health)
    build_stamp = datetime.now(UTC)
    replacements = {
        "__PLOTLY_CDN__": plotly_js_cdn(),
        "__REPO_URL__": REPO_URL,
        "__PAGES_URL__": PAGES_URL,
        "__ASSET_VERSION__": build_stamp.strftime("%Y%m%d%H%M"),
        "__MARKETS__": f"{df['market'].nunique()}",
        "__LOCATIONS__": f"{df['neighborhood'].nunique()}",
        "__MEDIAN_PRICE__": money(median_metro),
        "__ACCURACY__": f"{100 - metrics.get('mape', 0):.1f}%",
        "__MAPE__": f"{metrics.get('mape', 0):.2f}%",
        "__R2__": f"{metrics.get('r2', 0):.3f}",
        "__MODEL_NAME__": html.escape(str(metrics.get("model_name", "Regressor"))),
        "__TRAIN_TEST__": f"{metrics.get('n_records', 0):,}",
        "__MAE__": dollars(metrics.get("mae", 0)),
        "__RMSE__": dollars(metrics.get("rmse", 0)),
        "__BASELINE__": f"{metrics.get('improvement_vs_baseline_pct', 0):.1f}% lower MAE",
        "__TOP_DRIVERS__": html.escape(top_drivers),
        "__SIDEBAR_DATA__": sidebar["json"],
        "__COUNTRY_OPTIONS__": sidebar["country_options"],
        "__GLOSSARY__": sidebar["glossary"],
        "__CHART_VALUES__": build_market_values(summary),
        "__CHART_GROWTH__": build_market_growth(history, summary),
        "__CHART_YIELD__": build_rental_yield(summary),
        "__CHART_COUNTRIES__": build_country_growth(comparison),
        "__CHART_IMPORTANCE__": build_feature_importance(importance),
        "__GENERATED_AT__": datetime.now(UTC).strftime("%b %Y"),
        "__BUILD_STAMP__": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
    }

    page = TEMPLATE
    for token, value in replacements.items():
        page = page.replace(token, value)
    return page


def main() -> None:
    """Build ``docs/index.html`` from the market data and model."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    data = load_or_create_data()
    _, metrics, importance = ensure_model(data)
    overview = market_data.get_market_data()
    summary = overview["summary"]
    history = overview["history"]
    health = overview["health"]
    intl = international_data.get_country_data()
    comparison = intl["comparison"]
    bis = intl["bis"]

    page = render_page(
        data, summary, history, comparison, bis, intl["summary"], health, metrics, importance
    )
    INDEX_PATH.write_text(page, encoding="utf-8")

    print("=" * 62)
    print("GitHub Pages site generated")
    print("=" * 62)
    print(f"Output        : {INDEX_PATH}")
    print(f"Markets       : {summary.shape[0]}")
    print(f"Countries     : {len(intl['summary'])} (source: {intl['source']})")
    print(f"Listings      : {len(data):,}")
    print(f"R2 / MAPE     : {metrics.get('r2', 0):.3f} / {metrics.get('mape', 0):.2f}%")


if __name__ == "__main__":
    main()

"""
scripts/build_site.py
=====================
Generate the static GitHub Pages landing page in ``docs/``.

The page embeds real, interactive Plotly figures built from live/committed
Zillow market data and the trained pipeline, so it stays in sync with the
project. The live prediction tool runs on Streamlit Community Cloud and is
linked from the page.

Usage
-----
    python scripts/build_site.py

    # point the "Launch live app" buttons at your deployment:
    LIVE_APP_URL=https://your-app.streamlit.app python scripts/build_site.py
"""

from __future__ import annotations

import html
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
import market_data  # noqa: E402
from data_loader import load_or_create_data  # noqa: E402
from model import ensure_model  # noqa: E402

REPO_URL = "https://github.com/armand-vw/Property-Price-Market-Dashboard"
PAGES_URL = "https://armand-vw.github.io/Property-Price-Market-Dashboard/"
LIVE_APP_URL = os.environ.get("LIVE_APP_URL", "").strip()

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
    for color, row in zip(config.CHART_SEQUENCE, top_markets.itertuples()):
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
<meta property="og:description" content="Live US market data, gradient-boosted valuation and a live price estimation tool." />
<meta property="og:type" content="website" />
<link rel="stylesheet" href="style.css" />
<script src="__PLOTLY_CDN__"></script>
</head>
<body>

<header class="nav">
  <div class="nav-inner">
    <span class="brand">🏙️ Property&nbsp;Insights</span>
    <nav>
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
      from Zillow Research with a gradient-boosted valuation model. Select a market, explore
      real neighbourhood values and trends, and estimate property prices.
    </p>
    <div class="cta-row">
      <a class="btn btn-primary" href="__LIVE_APP_URL__" target="_blank" rel="noopener">🚀 Launch live app</a>
      <a class="btn btn-outline" href="__REPO_URL__" target="_blank" rel="noopener">View source</a>
    </div>
    <p class="cta-note">Live predictor on Streamlit Community Cloud · charts below are interactive.</p>
  </div>
</section>

<section class="kpis">
  <div class="kpi"><span class="kpi-label">Markets Covered</span><span class="kpi-value">__MARKETS__</span><span class="kpi-sub">Largest US metros</span></div>
  <div class="kpi"><span class="kpi-label">Median Metro Value</span><span class="kpi-value">__MEDIAN_PRICE__</span><span class="kpi-sub">Live Zillow ZHVI</span></div>
  <div class="kpi"><span class="kpi-label">Model Accuracy</span><span class="kpi-value">__ACCURACY__</span><span class="kpi-sub">100 − MAPE (__MAPE__)</span></div>
  <div class="kpi"><span class="kpi-label">R² Score</span><span class="kpi-value">__R2__</span><span class="kpi-sub">Held-out variance explained</span></div>
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
  <div class="chart-card"><h3>What Drives Home Prices</h3><p class="muted">Aggregated XGBoost gain importance.</p><div class="chart">__CHART_IMPORTANCE__</div></div>
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
    <span class="pill">pytest</span>
    <span class="pill">GitHub Actions</span>
  </div>
  <p class="section-lede" style="margin-top:16px">
    Market values: Zillow Research ZHVI (monthly, latest published month). Listing-level
    features are synthesised and calibrated to real neighbourhood medians.
  </p>
  <div class="code-card">
    <div class="code-title">Run locally</div>
    <pre><code>git clone __REPO_URL__.git
cd Property-Price-Market-Dashboard
python3 -m venv .venv &amp;&amp; source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py</code></pre>
  </div>
</section>

<footer class="footer">
  <p>Built with Streamlit, Plotly, XGBoost &amp; scikit-learn · Market data: Zillow Research. Estimates are illustrative and not financial advice.</p>
  <p><a href="__REPO_URL__" target="_blank" rel="noopener">Source on GitHub</a> · Generated __GENERATED_AT__</p>
</footer>

</body>
</html>
"""


def render_page(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    history: pd.DataFrame,
    metrics: dict,
    importance: pd.DataFrame,
) -> str:
    """Populate the template with data, metrics and chart fragments."""
    top_drivers = ", ".join(importance.head(4)["feature"].tolist())
    median_metro = summary["latest_value"].median()
    replacements = {
        "__PLOTLY_CDN__": plotly_js_cdn(),
        "__LIVE_APP_URL__": LIVE_APP_URL or REPO_URL,
        "__REPO_URL__": REPO_URL,
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
        "__CHART_VALUES__": build_market_values(summary),
        "__CHART_GROWTH__": build_market_growth(history, summary),
        "__CHART_YIELD__": build_rental_yield(summary),
        "__CHART_IMPORTANCE__": build_feature_importance(importance),
        "__GENERATED_AT__": datetime.now(timezone.utc).strftime("%b %Y"),
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

    page = render_page(data, summary, history, metrics, importance)
    INDEX_PATH.write_text(page, encoding="utf-8")

    print("=" * 62)
    print("GitHub Pages site generated")
    print("=" * 62)
    print(f"Output        : {INDEX_PATH}")
    print(f"Markets       : {summary.shape[0]}")
    print(f"Listings      : {len(data):,}")
    print(f"R2 / MAPE     : {metrics.get('r2', 0):.3f} / {metrics.get('mape', 0):.2f}%")
    print(f"Live app URL  : {LIVE_APP_URL or '(not set - buttons link to GitHub)'}")


if __name__ == "__main__":
    main()

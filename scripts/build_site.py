"""
scripts/build_site.py
=====================
Generate the static GitHub Pages landing page in ``docs/``.

The page is *not* a screenshot - it embeds real, interactive Plotly figures
rendered from the actual cleaned dataset and trained pipeline, so it stays in
sync with the project. The live prediction tool lives on Streamlit Community
Cloud and is linked from the page.

Usage
-----
    python scripts/build_site.py

To point the "Launch live app" buttons at your deployed app::

    LIVE_APP_URL=https://your-app.streamlit.app python scripts/build_site.py
"""

from __future__ import annotations

import html
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make the project root importable when run as ``python scripts/build_site.py``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import config
from data_loader import get_neighborhood_stats, load_or_create_data
from model import ensure_model

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
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
        config={
            "displayModeBar": False,
            "responsive": True,
            "displaylogo": False,
        },
    )


# --------------------------------------------------------------------------- #
# Figure builders
# --------------------------------------------------------------------------- #
def build_price_per_sqft(df: pd.DataFrame, stats: pd.DataFrame) -> str:
    """Price per square foot by neighbourhood, coloured by age band."""
    banded = df.copy()
    banded["age_band"] = pd.cut(
        banded["property_age"],
        bins=config.AGE_BINS,
        labels=config.AGE_LABELS,
        include_lowest=True,
    )
    fig = px.scatter(
        banded,
        x="neighborhood",
        y="price_per_sqft",
        color="age_band",
        category_orders={
            "neighborhood": list(stats.index),
            "age_band": config.AGE_LABELS,
        },
        color_discrete_sequence=config.CHART_SEQUENCE,
        opacity=0.62,
        labels={
            "neighborhood": "",
            "price_per_sqft": "Price per sq ft ($)",
            "age_band": "Age band",
        },
        hover_data={
            "bedrooms": True,
            "bathrooms": True,
            "sqft": ":,",
            "price": ":,",
        },
    )
    fig.update_traces(marker=dict(size=8, line=dict(width=0.4, color="white")))
    fig.update_yaxes(tickprefix="$", tickformat=",")
    return fig_to_html(style_fig(fig, 440))


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
    fig.update_traces(
        texttemplate="%{x:.3f}", textposition="outside", cliponaxis=False
    )
    return fig_to_html(style_fig(fig, 440))


def build_median_price(stats: pd.DataFrame) -> str:
    """Horizontal bar chart of median price by neighbourhood."""
    frame = stats.reset_index().sort_values("median_price")
    fig = px.bar(
        frame,
        x="median_price",
        y="neighborhood",
        orientation="h",
        text="median_price",
        labels={"median_price": "Median price ($)", "neighborhood": ""},
        color="median_price",
        color_continuous_scale=["#C7D2FE", config.COLORS["primary"]],
    )
    fig.update_traces(
        texttemplate="%{text:$,.0f}", textposition="outside", cliponaxis=False
    )
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(tickprefix="$", tickformat=",")
    return fig_to_html(style_fig(fig, 440))


# --------------------------------------------------------------------------- #
# Page template
# --------------------------------------------------------------------------- #
TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Real Estate Price Estimator &amp; Market Insights Dashboard</title>
<meta name="description" content="Interactive real-estate price estimator and market insights dashboard built with Streamlit, XGBoost, scikit-learn and Plotly." />
<meta property="og:title" content="Real Estate Price Estimator &amp; Market Insights" />
<meta property="og:description" content="Gradient-boosted valuation model, interactive market analytics and a live price estimation tool." />
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
      An end-to-end, production-style application that estimates property values and
      surfaces market insights. Gradient-boosted valuation, leak-free scikit-learn
      pipelines, and interactive analytics — from data synthesis to a live dashboard.
    </p>
    <div class="cta-row">
      <a class="btn btn-primary" href="__LIVE_APP_URL__" target="_blank" rel="noopener">🚀 Launch live app</a>
      <a class="btn btn-outline" href="__REPO_URL__" target="_blank" rel="noopener">View source</a>
    </div>
    <p class="cta-note">Live predictor hosted on Streamlit Community Cloud · charts below are interactive.</p>
  </div>
</section>

<section class="kpis">
  <div class="kpi"><span class="kpi-label">Total Listings</span><span class="kpi-value">__TOTAL_LISTINGS__</span><span class="kpi-sub">Synthetic property records</span></div>
  <div class="kpi"><span class="kpi-label">Median Market Price</span><span class="kpi-value">__MEDIAN_PRICE__</span><span class="kpi-sub">Across all listings</span></div>
  <div class="kpi"><span class="kpi-label">Model Accuracy</span><span class="kpi-value">__ACCURACY__</span><span class="kpi-sub">100 − MAPE (__MAPE__)</span></div>
  <div class="kpi"><span class="kpi-label">R² Score</span><span class="kpi-value">__R2__</span><span class="kpi-sub">Held-out variance explained</span></div>
</section>

<section class="features">
  <h2>What it does</h2>
  <div class="grid grid-3">
    <div class="card"><div class="icon">📈</div><h3>Live valuation</h3><p>Configure a property — neighbourhood, size, beds, baths, age, pool, garage — and get an instant estimate with an empirical valuation range.</p></div>
    <div class="card"><div class="icon">🧭</div><h3>Market analytics</h3><p>Explore price-per-sqft distributions, neighbourhood medians and age-band segmentation with interactive Plotly charts.</p></div>
    <div class="card"><div class="icon">🧠</div><h3>Explainable model</h3><p>XGBoost with aggregated feature importances, predicted-vs-actual parity and residual diagnostics surfaced in the dashboard.</p></div>
    <div class="card"><div class="icon">🧱</div><h3>Leak-free pipeline</h3><p>Imputation, scaling and one-hot encoding fitted inside an sklearn pipeline — never on the test set.</p></div>
    <div class="card"><div class="icon">📉</div><h3>Honest uncertainty</h3><p>Valuation ranges come from held-out residual quantiles, not an over-confident symmetric band.</p></div>
    <div class="card"><div class="icon">⚙️</div><h3>Production ready</h3><p>Joblib-persisted artifact, graceful Random-Forest fallback, pytest suite and CI on every push.</p></div>
  </div>
</section>

<section id="preview" class="section">
  <h2>Interactive preview</h2>
  <p class="section-lede">These charts are rendered from the actual dataset and trained model at build time.</p>
  <div class="chart-card"><h3>Price per Sq Ft by Neighborhood</h3><p class="muted">Each point is a listing, colour-coded by property age band.</p><div class="chart">__CHART_PSF__</div></div>
  <div class="grid grid-2">
    <div class="chart-card"><h3>Median Price by Neighborhood</h3><p class="muted">Market tiers across the eight neighborhoods.</p><div class="chart">__CHART_MEDIAN__</div></div>
    <div class="chart-card"><h3>What Drives Home Prices</h3><p class="muted">Aggregated XGBoost gain importance.</p><div class="chart">__CHART_IMPORTANCE__</div></div>
  </div>
</section>

<section id="model" class="section">
  <h2>Model performance</h2>
  <p class="section-lede">80/20 hold-out evaluation, trained on the <code>log1p</code> target and scored in dollars.</p>
  <div class="table-wrap">
    <table>
      <tbody>
        <tr><th>Estimator</th><td>__MODEL_NAME__</td></tr>
        <tr><th>Train / Test rows</th><td>__TRAIN_TEST__</td></tr>
        <tr><th>MAE</th><td><strong>__MAE__</strong></td></tr>
        <tr><th>RMSE</th><td><strong>__RMSE__</strong></td></tr>
        <tr><th>MAPE</th><td><strong>__MAPE__</strong></td></tr>
        <tr><th>Median APE</th><td>__MEDIAN_APE__</td></tr>
        <tr><th>R² score</th><td><strong>__R2__</strong></td></tr>
        <tr><th>Improvement vs. median baseline</th><td>__BASELINE__</td></tr>
        <tr><th>Top price drivers</th><td>__TOP_DRIVERS__</td></tr>
      </tbody>
    </table>
  </div>
</section>

<section id="stack" class="section">
  <h2>Tech stack</h2>
  <div class="pills">
    <span class="pill">Python 3.12</span>
    <span class="pill">Streamlit</span>
    <span class="pill">XGBoost</span>
    <span class="pill">scikit-learn</span>
    <span class="pill">Plotly</span>
    <span class="pill">pandas</span>
    <span class="pill">NumPy</span>
    <span class="pill">joblib</span>
    <span class="pill">pytest</span>
    <span class="pill">GitHub Actions</span>
  </div>
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
  <p>Built with Streamlit, Plotly, XGBoost &amp; scikit-learn. Synthetic data — estimates are illustrative and not financial advice.</p>
  <p><a href="__REPO_URL__" target="_blank" rel="noopener">Source on GitHub</a> · Generated __GENERATED_AT__</p>
</footer>

</body>
</html>
"""


def render_page(
    df: pd.DataFrame,
    stats: pd.DataFrame,
    metrics: dict,
    importance: pd.DataFrame,
) -> str:
    """Populate the template with data, metrics and chart fragments."""
    top_drivers = ", ".join(importance.head(4)["feature"].tolist())
    replacements = {
        "__PLOTLY_CDN__": plotly_js_cdn(),
        "__LIVE_APP_URL__": LIVE_APP_URL or REPO_URL,
        "__REPO_URL__": REPO_URL,
        "__TOTAL_LISTINGS__": f"{len(df):,}",
        "__MEDIAN_PRICE__": money(df["price"].median()),
        "__ACCURACY__": f"{100 - metrics.get('mape', 0):.1f}%",
        "__MAPE__": f"{metrics.get('mape', 0):.2f}%",
        "__R2__": f"{metrics.get('r2', 0):.3f}",
        "__MODEL_NAME__": html.escape(str(metrics.get("model_name", "Regressor"))),
        "__TRAIN_TEST__": f"{metrics.get('n_train', 0):,} / {metrics.get('n_test', 0):,}",
        "__MAE__": dollars(metrics.get("mae", 0)),
        "__RMSE__": dollars(metrics.get("rmse", 0)),
        "__MEDIAN_APE__": f"{metrics.get('median_ape', 0):.2f}%",
        "__BASELINE__": f"{metrics.get('improvement_vs_baseline_pct', 0):.1f}% lower MAE",
        "__TOP_DRIVERS__": html.escape(top_drivers),
        "__CHART_PSF__": build_price_per_sqft(df, stats),
        "__CHART_MEDIAN__": build_median_price(stats),
        "__CHART_IMPORTANCE__": build_feature_importance(importance),
        "__GENERATED_AT__": datetime.now(timezone.utc).strftime("%b %Y"),
    }

    page = TEMPLATE
    for token, value in replacements.items():
        page = page.replace(token, value)
    return page


def main() -> None:
    """Build ``docs/index.html`` from the live data and model."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    data = load_or_create_data()
    model, metrics, importance = ensure_model(data)
    stats = get_neighborhood_stats(data)

    page = render_page(data, stats, metrics, importance)
    INDEX_PATH.write_text(page, encoding="utf-8")

    print("=" * 62)
    print("GitHub Pages site generated")
    print("=" * 62)
    print(f"Output        : {INDEX_PATH}")
    print(f"Listings      : {len(data):,}")
    print(f"R2 / MAPE     : {metrics.get('r2', 0):.3f} / {metrics.get('mape', 0):.2f}%")
    print(f"Live app URL  : {LIVE_APP_URL or '(not set - buttons link to GitHub)'}")


if __name__ == "__main__":
    main()

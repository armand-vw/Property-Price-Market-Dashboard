# Self-contained image for the Real Estate Price Estimator & Market Insights
# Dashboard. The market snapshot and fitted model are committed, so the running
# container needs no network access: it serves live-or-snapshot data either way.
#
#   docker build -t property-insights .
#   docker run --rm -p 8501:8501 property-insights
#   # fully offline (use the committed snapshot, no outbound calls):
#   docker run --rm -p 8501:8501 -e RPE_OFFLINE=1 property-insights

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Install dependencies first so Docker layer caching works.
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Application code and committed artifacts.
COPY app.py config.py data_loader.py market_data.py international_data.py insights.py model.py ./
COPY .streamlit/ ./.streamlit/
COPY market_data/ ./market_data/
COPY models/ ./models/

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]

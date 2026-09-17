# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm AS base
COPY --from=ghcr.io/astral-sh/uv:0.11.24 /uv /usr/local/bin/uv
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app:/app/src
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app \
    && chown app:app /app
COPY pyproject.toml uv.lock .python-version ./

FROM base AS trainer
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --group train --no-install-project
COPY src ./src
COPY config ./config
COPY scripts ./scripts
RUN mkdir -p artifacts output data utils && chown app:app artifacts output data utils
USER app
CMD ["python", "src/pipeline.py"]

FROM base AS api
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --group api --no-install-project
COPY src/__init__.py src/inference_pipeline.py src/scenario_features.py src/FeatureEngineering.py ./src/
COPY src/helpers ./src/helpers
COPY src/display_streamlit/api.py src/display_streamlit/inference_service.py ./src/display_streamlit/
USER app
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"
CMD ["uvicorn", "src.display_streamlit.api:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS ui
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --group ui --no-install-project
COPY src/display_streamlit/streamlit_app.py ./src/display_streamlit/
COPY src/display_streamlit/pages ./src/display_streamlit/pages
COPY src/display_streamlit/data_streamlit ./src/display_streamlit/data_streamlit
COPY src/display_streamlit/models_streamlit/demo_scenarios.csv ./src/display_streamlit/models_streamlit/
WORKDIR /app/src/display_streamlit
USER app
EXPOSE 8501
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4)"
CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false"]

# Serving image for the LOS FastAPI app (Phase 4 — deploy).
#
# Ships an immutable model baked in at serving_model/, produced first by:
#     uv run python -m los_pred.export_model
# so the container needs NO MLflow tracking DB at runtime.
#
# Build:  docker build -t los-pred-api .
# Run:    docker run --rm -p 8000:8000 los-pred-api
# Then:   http://127.0.0.1:8000/docs

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# LightGBM's runtime needs the OpenMP shared library (libgomp).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies + the package. src is needed because pyproject builds the
# los_pred package; README.md is referenced by pyproject's `readme` field.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

# Bake in the exported model bundle and tell the app to load from it.
COPY serving_model ./serving_model
ENV LOS_MODEL_DIR=/app/serving_model

EXPOSE 8000
CMD ["uv", "run", "--no-sync", "--no-dev", \
     "uvicorn", "los_pred.app:app", "--host", "0.0.0.0", "--port", "8000"]

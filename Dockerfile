# ---------------------------------------------------------------------------
# Stage 1: base — single-install runtime image
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS base

LABEL org.opencontainers.image.title="ayase" \
    org.opencontainers.image.description="Video dataset validator for ML training" \
    org.opencontainers.image.licenses="MIT" \
    org.opencontainers.image.source="https://github.com/seruva19/ayase"

# System libraries required by opencv-python, imageio-ffmpeg, and pillow
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Non-root user
RUN groupadd --gid 1000 ayase && \
    useradd --uid 1000 --gid ayase --create-home ayase

WORKDIR /app

# --- Layer-cached dependency install ---
# Copy only the build metadata first so that dependency layers are cached
# as long as pyproject.toml does not change.
COPY pyproject.toml README.md LICENSE ./
COPY src/ayase/__init__.py src/ayase/__init__.py

# Install the full single-install runtime distribution
RUN pip install --no-cache-dir . && \
    rm -rf /tmp/*

# --- Copy the full source ---
COPY src/ src/

# Re-install so the package entry-point points at the real source
RUN pip install --no-cache-dir --no-deps . && \
    rm -rf /tmp/*

# Drop to non-root
USER ayase

ENTRYPOINT ["ayase"]

# ---------------------------------------------------------------------------
# Stage 2: compatibility alias for the single-install runtime image
# ---------------------------------------------------------------------------
FROM base AS ml

ENTRYPOINT ["ayase"]

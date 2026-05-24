# ═══════════════════════════════════════════════════════════════════════════════
#  Retail Sales Forecasting — Flask API Docker Image
#  Base: python:3.11-slim  |  Created by: sanT
# ═══════════════════════════════════════════════════════════════════════════════

FROM python:3.11-slim

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        libgomp1 \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── API source ─────────────────────────────────────────────────────────────────
COPY forecast_api.py .

# ── Shared output directory ────────────────────────────────────────────────────
RUN mkdir -p /shared/uploads /shared/outputs

# ── Expose port ────────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Start with Gunicorn (production-grade WSGI server) ─────────────────────────
CMD ["gunicorn", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "1", \
     "--timeout", "600", \
     "--log-level", "info", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "forecast_api:app"]

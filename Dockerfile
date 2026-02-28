# ─────────────────────────────────────────────────────────────
#  VayuSwarm — Multi-stage Docker Build
# ─────────────────────────────────────────────────────────────
FROM python:3.12-slim AS base

# System deps for OpenCV, ZMQ, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libzmq5 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install Python deps first (cached layer) ──
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]" 2>/dev/null || \
    pip install --no-cache-dir .

# ── Copy source ──
COPY . .

# Ensure data directory exists
RUN mkdir -p /app/data

# ── Default env ──
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# ─────────────────────────────────────────────────────────────
#  Target: Ground station + Dashboard
# ─────────────────────────────────────────────────────────────
FROM base AS ground
EXPOSE 5555 5556 8080
CMD ["python", "scripts/launch_ground.py", \
     "--config", "config/simulation.yaml", \
     "--dashboard"]

# ─────────────────────────────────────────────────────────────
#  Target: Drone agent
# ─────────────────────────────────────────────────────────────
FROM base AS drone
CMD ["python", "scripts/launch_drone.py", \
     "--id", "drone-0", \
     "--config", "config/simulation.yaml"]

# ─────────────────────────────────────────────────────────────
#  Target: Test runner
# ─────────────────────────────────────────────────────────────
FROM base AS test
CMD ["bash", "run_tests.sh"]

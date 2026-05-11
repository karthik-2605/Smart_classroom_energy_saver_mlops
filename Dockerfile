# ============================================================
# Dockerfile
# Smart Classroom Energy Saver — RL Training Container
# ============================================================
# Build:  docker build -t smart-classroom-rl .
# Run:    docker run smart-classroom-rl
# Run v2: docker run smart-classroom-rl python train.py --config config/qlearning_v2.yaml
# ============================================================

# --- Base image: lightweight Python 3.11 ---
FROM python:3.11-slim

# Metadata labels
LABEL maintainer="your-email@example.com"
LABEL project="smart-classroom-energy-saver"
LABEL version="1.0"

# --- Set working directory inside container ---
WORKDIR /app

# --- Install system dependencies ---
# (needed for matplotlib headless rendering)
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# --- Copy requirements first (Docker layer caching) ---
# This means if only code changes, pip install is NOT re-run
COPY requirements.txt .

# --- Install Python dependencies ---
RUN pip install --no-cache-dir -r requirements.txt

# --- Copy all project files into container ---
COPY . .

# --- Create output directories (in case they don't exist) ---
RUN mkdir -p results models

# --- Set environment variable for headless matplotlib ---
ENV MPLBACKEND=Agg

# --- Default command: train with baseline config ---
CMD ["python", "train.py", "--config", "config/qlearning_v1.yaml"]

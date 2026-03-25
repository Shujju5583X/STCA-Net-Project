# ---- Build Stage ----
FROM python:3.10-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---- Runtime Stage ----
FROM python:3.10-slim

WORKDIR /app

# Install only runtime system dependencies (OpenCV needs these)
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    libopenblas-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Set environment variables for CPU operation
ENV MEDIAPIPE_DISABLE_GPU=1
ENV KMP_DUPLICATE_LIB_OK=True
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV OPENBLAS_NUM_THREADS=4

# Copy application code
COPY . /app

# Expose the port that the application will run on
EXPOSE 10000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:10000/health')" || exit 1

# Run the application
CMD ["python", "server.py"]

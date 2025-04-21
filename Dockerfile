# Use CUDA base image for GPU support
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY tests/ tests/
COPY deployment/ deployment/

# Set up logging directory
RUN mkdir -p /var/log/digigod

# Expose ports
EXPOSE 8000

# Set entrypoint
ENTRYPOINT ["python3", "-m", "src.core.digigod_nexus"]

# Default command
CMD ["--mode=cloud", "--qpus=128", "--ethics=asilomar_v5"] 
# Builder stage
FROM python:3.10-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set up virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install quantum dependencies
RUN pip install --no-cache-dir \
    qiskit[all]==0.44.0 \
    pennylane==0.31.0 \
    cirq==1.2.0 \
    mitiq==0.28.0

# Runtime stage
FROM python:3.10-slim

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY src/ /app/src/
COPY tests/ /app/tests/
COPY setup.py pyproject.toml README.md ./

# Install the package
RUN pip install -e .

# Set environment variables
ENV PYTHONPATH=/app
ENV QUANTUM_BACKEND=ibm_kyiv
ENV LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "from entangled_multimodal_system.monitoring.quantum_telemetry import QuantumMonitor; QuantumMonitor().track_operation(None, 'health_check')" || exit 1

# Default command
CMD ["python", "-m", "entangled_multimodal_system"] 
@echo off

:: Create and activate virtual environment
python -m venv quantum_env
call quantum_env\Scripts\activate

:: Upgrade pip and setuptools
python -m pip install --upgrade pip setuptools wheel

:: Install core dependencies
pip install -r requirements.txt

:: Install quantum computing extensions
pip install qiskit[all]
pip install qiskit-aer-gpu  :: For GPU acceleration if available

:: Install machine learning extensions
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  :: For CUDA 11.8
pip install tensorflow[and-cuda]  :: For GPU support

:: Install visualization extensions
pip install plotly[all]
pip install pyvista[all]

:: Install audio processing extensions
pip install pyaudio
pip install librosa

:: Install development tools
pip install black isort flake8 mypy
pip install pytest pytest-cov pytest-benchmark

:: Install documentation tools
pip install sphinx sphinx-rtd-theme

:: Install monitoring tools
pip install prometheus-client grafana-api

:: Install database tools
pip install sqlalchemy alembic redis

:: Install web tools
pip install fastapi uvicorn websockets

:: Install security tools
pip install python-jose passlib bcrypt cryptography

:: Verify installations
python -c "import qiskit; print('Qiskit version:', qiskit.__version__)"
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"

:: Create necessary directories
mkdir logs
mkdir data
mkdir models
mkdir tests
mkdir docs

:: Set up environment variables
echo set QUANTUM_ENV=development >> quantum_env\Scripts\activate.bat
echo set PYTHONPATH=%PYTHONPATH%;%CD% >> quantum_env\Scripts\activate.bat

:: Initialize git repository if not already done
if not exist ".git" (
    git init
    git add .
    git commit -m "Initial commit"
)

echo Setup complete! Activate the environment with: call quantum_env\Scripts\activate.bat 
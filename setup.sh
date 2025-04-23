#!/bin/bash

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Install RDKit (if not available through pip)
if ! pip show rdkit &> /dev/null; then
    echo "Installing RDKit..."
    conda install -c conda-forge rdkit
fi

# Install Qiskit (if not available through pip)
if ! pip show qiskit &> /dev/null; then
    echo "Installing Qiskit..."
    pip install qiskit
    pip install qiskit-machine-learning
fi

# Create necessary directories
mkdir -p data/raw
mkdir -p data/processed
mkdir -p models
mkdir -p results
mkdir -p logs

# Set up pre-commit hooks
pre-commit install

echo "Setup completed successfully!" 
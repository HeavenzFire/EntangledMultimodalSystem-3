import pytest
import os
import sys
import numpy as np
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment before each test."""
    yield

@pytest.fixture
def mock_time_series():
    """Generate mock time series data"""
    t = np.linspace(0, 100, 1000)
    T = 0.5 + 0.3 * np.sin(0.1 * t)
    L = 0.6 + 0.2 * np.cos(0.15 * t)
    S = 0.4 + 0.4 * np.sin(0.05 * t + np.pi/4)
    U = 0.7 + 0.1 * np.cos(0.2 * t + np.pi/3)
    return t, T, L, S, U

@pytest.fixture
def mock_metrics():
    """Generate mock metrics for testing"""
    return {
        'unity_convergence': 0.95,
        'synch_resonance': 0.85,
        'T_L_ratio': 1.618,
        'topological_consistency': 0.75,
        'temporal_symmetry': 0.92
    }

def pytest_addoption(parser):
    """Add custom command line options for test configuration"""
    parser.addoption(
        "--quantum",
        action="store_true",
        help="Run quantum-specific tests"
    )
    parser.addoption(
        "--sacred-geometry-level",
        type=int,
        default=11,
        help="Set sacred geometry dimension level"
    )
    parser.addoption(
        "--report-format",
        choices=['merkaba_html', 'standard', 'detailed'],
        default='standard',
        help="Set test report format"
    )

def pytest_configure(config):
    """Configure pytest with custom settings"""
    if config.getoption("--quantum"):
        config.addinivalue_line(
            "markers", "quantum: mark test as requiring quantum backend"
        )
    
    if config.getoption("--sacred-geometry-level"):
        config.addinivalue_line(
            "markers", "sacred_geometry: mark test as requiring sacred geometry validation"
        ) 
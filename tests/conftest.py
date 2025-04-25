import pytest
import os
import sys
import numpy as np
from metaphysical.mathematics.core.simulation import (
    MetaphysicalSimulator,
    MetaphysicalParameters,
    MetaphysicalState
)

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment before each test."""
    # Add any test environment setup here
    yield
    # Add any cleanup here 

@pytest.fixture(scope="session")
def base_parameters():
    """Create base parameters for testing"""
    return MetaphysicalParameters(
        alpha=0.8,    # Transcendence amplification
        beta=1.2,     # Synchronicity coupling
        gamma=0.05,   # Ego dissolution
        lambda_=1.5,  # Love resonance
        mu=0.1,       # Love decay
        kappa=0.3,    # Synchronicity influence
        sigma=0.02,   # Unity field constant
        omega=0.15,   # Transcendence-synchronicity coupling
        xi=0.4,       # Unity memory strength
        eta=0.25,     # Memory decay
        nu=0.01       # Unity dissipation
    )

@pytest.fixture(scope="session")
def initial_state():
    """Create initial state for testing"""
    return MetaphysicalState(
        transcendence=0.1,
        love=0.1,
        synchronicity=0.1,
        unity=0.1,
        time=0
    )

@pytest.fixture(scope="session")
def base_simulator(base_parameters, initial_state):
    """Create base simulator with solved system"""
    simulator = MetaphysicalSimulator(base_parameters)
    simulator.solve(initial_state)
    return simulator

@pytest.fixture
def random_state():
    """Generate random metaphysical state"""
    return MetaphysicalState(
        transcendence=np.random.random(),
        love=np.random.random(),
        synchronicity=np.random.random(),
        unity=np.random.random(),
        time=np.random.random() * 100
    )

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
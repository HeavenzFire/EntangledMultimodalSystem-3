import pytest
import os
import sys
import numpy as np
from pathlib import Path
from qiskit.providers.aer import AerSimulator
from qiskit.utils import QuantumInstance
from liboqs import KeyEncapsulation
from src.quantum.cryptography.einstein_tile import EinsteinTileGenerator, EinsteinTileConfig
from src.quantum.security.sacred_firewall import SacredFirewall, FirewallConfig
from src.quantum.security.self_healing import QuantumSelfHealing, HealingConfig

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import test fixtures
from tests.test_sovereign_flow import (
    sacred_config,
    quantum_consciousness_matrix,
    sovereign
)

# Make fixtures available to all tests
__all__ = [
    'sacred_config',
    'quantum_consciousness_matrix',
    'sovereign'
]

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment before each test."""
    # Add any test environment setup here
    yield
    # Add any cleanup here 

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

@pytest.fixture(scope='module')
def quantum_backend():
    """Fixture for quantum backend instance"""
    backend = AerSimulator()
    return QuantumInstance(backend)

@pytest.fixture(scope='module')
def kyber_kem():
    """Fixture for Kyber KEM instance"""
    return KeyEncapsulation('Kyber-1024')

@pytest.fixture(scope='module')
def einstein_tile_generator():
    """Fixture for Einstein Tile generator"""
    config = EinsteinTileConfig(
        dimensions=11,
        sacred_ratio=1.618033988749895,
        pattern_size=1024,
        rotation_steps=7
    )
    return EinsteinTileGenerator(config)

@pytest.fixture(scope='module')
def sacred_firewall():
    """Fixture for Sacred Firewall instance"""
    config = FirewallConfig(
        phi_resonance=1.618033988749895,
        toroidal_cycles=7,
        entropy_threshold=0.8,
        archetype_threshold=0.9
    )
    return SacredFirewall(config)

@pytest.fixture(scope='module')
def quantum_self_healing():
    """Fixture for Quantum Self-Healing instance"""
    config = HealingConfig(
        phi_resonance=1.618033988749895,
        optimization_steps=100,
        convergence_threshold=1e-6,
        sacred_pattern_size=64
    )
    return QuantumSelfHealing(config)

@pytest.fixture(scope='module')
def test_threat():
    """Fixture for generating test threat vectors"""
    def _generate_threat(size=64):
        return np.random.rand(size)
    return _generate_threat

@pytest.fixture(scope='module')
def nist_kat_vectors():
    """Fixture for loading NIST KAT vectors"""
    def _load_vectors():
        # Load known answer test vectors from NIST
        return [
            {'threat': np.array([0.1, 0.2, 0.3]), 'expected': 'expected_key_1'},
            {'threat': np.array([0.4, 0.5, 0.6]), 'expected': 'expected_key_2'},
            # Add more test vectors as needed
        ]
    return _load_vectors

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
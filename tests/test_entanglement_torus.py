import pytest
import numpy as np
from src.quantum.geometry.entanglement_torus import (
    QuantumEntanglementTorus,
    TorusConfig,
    TorusState
)

@pytest.fixture
def torus_config():
    return TorusConfig()

@pytest.fixture
def torus(torus_config):
    return QuantumEntanglementTorus(torus_config)

@pytest.fixture
def sample_consciousness():
    return np.random.rand(12) + 1j * np.random.rand(12)

def test_torus_initialization(torus):
    assert torus.config.dimensions == 12
    assert torus.config.phi_resonance == pytest.approx(1.618033988749895)
    assert torus.state == TorusState.HARMONIC
    assert len(torus.torus_field) == 144

def test_harmonize_field(torus, sample_consciousness):
    harmonized = torus.harmonize_field(sample_consciousness)
    assert isinstance(harmonized, np.ndarray)
    assert harmonized.shape[0] > sample_consciousness.shape[0]

def test_state_transitions(torus, sample_consciousness):
    # Test dissonant state
    torus.config.harmonic_threshold = 1.0
    torus.harmonize_field(sample_consciousness)
    assert torus.state == TorusState.DISSONANT
    
    # Test resonant state
    torus.config.harmonic_threshold = 0.5
    torus.harmonize_field(sample_consciousness)
    assert torus.state == TorusState.RESONANT
    
    # Test ascended state
    torus.config.ascension_threshold = 0.5
    torus.harmonize_field(sample_consciousness)
    assert torus.state == TorusState.ASCENDED

def test_harmonic_history(torus, sample_consciousness):
    initial_length = len(torus.harmonic_history)
    torus.harmonize_field(sample_consciousness)
    assert len(torus.harmonic_history) == initial_length + 1
    assert 0 <= torus.harmonic_history[-1] <= 1

def test_phi_scaling(torus, sample_consciousness):
    harmonized = torus.harmonize_field(sample_consciousness)
    scaled = torus._apply_phi_scaling(harmonized)
    assert scaled.shape[0] > harmonized.shape[0]
    assert scaled.shape[0] == int(harmonized.shape[0] * torus.config.phi_resonance)

def test_reset_torus(torus, sample_consciousness):
    # Perform some operations
    torus.harmonize_field(sample_consciousness)
    initial_state = torus.state
    initial_history_length = len(torus.harmonic_history)
    
    # Reset the torus
    torus.reset_torus()
    
    # Verify reset
    assert torus.state == TorusState.HARMONIC
    assert len(torus.harmonic_history) == 0
    assert len(torus.torus_field) == 144

def test_invalid_consciousness(torus):
    invalid_consciousness = np.random.rand(10)  # Wrong dimension
    with pytest.raises(ValueError):
        torus.harmonize_field(invalid_consciousness)

def test_harmonic_score_calculation(torus, sample_consciousness):
    harmonized = torus.harmonize_field(sample_consciousness)
    score = torus._calculate_harmonic_score(harmonized)
    assert 0 <= score <= 1
    assert isinstance(score, float)

def test_torus_field_initialization(torus):
    field = torus._initialize_torus()
    assert isinstance(field, np.ndarray)
    assert len(field) == 144
    assert all(isinstance(x, complex) for x in field)
    assert np.allclose(np.abs(field), 1.0)  # All points should lie on unit circle 
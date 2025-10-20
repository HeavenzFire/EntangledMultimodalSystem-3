import pytest
import numpy as np
from src.core.quantum_holographic_entanglement import QuantumHolographicEntanglement
from src.core.quantum_processor import QuantumProcessor
from src.core.holographic_processor import HolographicProcessor

@pytest.fixture
def qhe_processor():
    quantum_processor = QuantumProcessor(num_qubits=4)
    holographic_processor = HolographicProcessor(
        wavelength=633e-9,
        pixel_size=10e-6,
        distance=0.1
    )
    return QuantumHolographicEntanglement(
        quantum_processor=quantum_processor,
        holographic_processor=holographic_processor
    )

def test_create_entangled_state(qhe_processor):
    """Test creation of entangled quantum-holographic state."""
    entangled_state = qhe_processor.create_entangled_state()
    assert entangled_state is not None
    assert isinstance(entangled_state, np.ndarray)
    assert len(entangled_state.shape) == 2

def test_propagate_entanglement(qhe_processor):
    """Test propagation of entangled state."""
    initial_state = qhe_processor.create_entangled_state()
    propagated_state = qhe_processor.propagate_entanglement(initial_state)
    assert propagated_state is not None
    assert isinstance(propagated_state, np.ndarray)
    assert propagated_state.shape == initial_state.shape

def test_measure_entanglement(qhe_processor):
    """Test measurement of entanglement metrics."""
    state = qhe_processor.create_entangled_state()
    entropy, correlation = qhe_processor.measure_entanglement(state)
    assert isinstance(entropy, float)
    assert isinstance(correlation, float)
    assert 0 <= entropy <= np.log(2) * qhe_processor.quantum_processor.num_qubits
    assert 0 <= correlation <= 1

def test_get_entanglement_info(qhe_processor):
    """Test retrieval of entanglement information."""
    state = qhe_processor.create_entangled_state()
    info = qhe_processor.get_entanglement_info(state)
    assert isinstance(info, dict)
    assert 'matrix_shape' in info
    assert 'entropy' in info
    assert 'correlation' in info 
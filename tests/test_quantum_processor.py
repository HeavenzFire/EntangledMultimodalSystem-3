import pytest
import numpy as np
from src.core.quantum_processor import QuantumProcessor
from src.utils.errors import ModelError

@pytest.fixture
def quantum_processor():
    return QuantumProcessor(num_qubits=4)

def test_initialization(quantum_processor):
    """Test quantum processor initialization."""
    assert quantum_processor.num_qubits == 4
    assert quantum_processor.state is not None
    assert isinstance(quantum_processor.state, np.ndarray)
    assert quantum_processor.state.shape == (2**4,)

def test_create_quantum_state(quantum_processor):
    """Test creation of quantum states."""
    state = quantum_processor.create_quantum_state()
    assert state is not None
    assert isinstance(state, np.ndarray)
    assert state.shape == (2**4,)
    assert np.abs(np.sum(np.abs(state)**2) - 1.0) < 1e-10

def test_apply_quantum_gate(quantum_processor):
    """Test application of quantum gates."""
    initial_state = quantum_processor.state.copy()
    
    # Test Hadamard gate
    quantum_processor.apply_gate("H", 0)
    assert not np.array_equal(quantum_processor.state, initial_state)
    
    # Test CNOT gate
    quantum_processor.apply_gate("CNOT", [0, 1])
    assert not np.array_equal(quantum_processor.state, initial_state)

def test_measure_state(quantum_processor):
    """Test quantum state measurement."""
    # Apply Hadamard to create superposition
    quantum_processor.apply_gate("H", 0)
    result = quantum_processor.measure_state()
    assert isinstance(result, dict)
    assert "measurement_outcome" in result
    assert "probability_distribution" in result
    assert isinstance(result["probability_distribution"], np.ndarray)

def test_entangle_qubits(quantum_processor):
    """Test qubit entanglement."""
    quantum_processor.entangle_qubits([0, 1])
    state = quantum_processor.state
    # Verify entanglement by checking reduced density matrix
    density_matrix = np.outer(state, state.conj())
    reduced_matrix = np.trace(density_matrix.reshape(2, 2, 2, 2), axis1=1, axis2=3)
    eigenvalues = np.linalg.eigvals(reduced_matrix)
    # Check if the state is maximally entangled
    assert np.allclose(eigenvalues, [0.5, 0.5], atol=1e-10)

def test_quantum_error_correction(quantum_processor):
    """Test quantum error correction."""
    # Introduce error
    quantum_processor.apply_gate("X", 0)  # Bit flip error
    # Apply error correction
    corrected_state = quantum_processor.apply_error_correction()
    assert corrected_state is not None
    assert isinstance(corrected_state, np.ndarray)

def test_get_quantum_state_info(quantum_processor):
    """Test retrieval of quantum state information."""
    info = quantum_processor.get_quantum_state_info()
    assert isinstance(info, dict)
    assert "state_vector" in info
    assert "num_qubits" in info
    assert "entanglement_measure" in info

def test_error_handling(quantum_processor):
    """Test error handling in quantum operations."""
    # Test invalid number of qubits
    with pytest.raises(ModelError):
        QuantumProcessor(num_qubits=0)
    
    # Test invalid gate
    with pytest.raises(ModelError):
        quantum_processor.apply_gate("INVALID", 0)
    
    # Test invalid qubit index
    with pytest.raises(ModelError):
        quantum_processor.apply_gate("H", quantum_processor.num_qubits + 1)

def test_reset_processor(quantum_processor):
    """Test reset of quantum processor."""
    # Apply some operations
    quantum_processor.apply_gate("H", 0)
    quantum_processor.apply_gate("CNOT", [0, 1])
    
    # Reset
    quantum_processor.reset_processor()
    
    # Verify reset state
    assert np.array_equal(quantum_processor.state, np.array([1] + [0]*(2**4-1))) 
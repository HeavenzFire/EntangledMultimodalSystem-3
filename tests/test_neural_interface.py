import pytest
import numpy as np
import tensorflow as tf
from src.core.neural_interface import NeuralInterface
from src.core.quantum_processor import QuantumProcessor
from src.core.holographic_processor import HolographicProcessor

@pytest.fixture
def neural_interface():
    quantum_processor = QuantumProcessor(num_qubits=4)
    holographic_processor = HolographicProcessor(
        wavelength=633e-9,
        pixel_size=10e-6,
        distance=0.1
    )
    return NeuralInterface(
        quantum_processor=quantum_processor,
        holographic_processor=holographic_processor
    )

def test_build_neural_network(neural_interface):
    """Test building of neural network architecture."""
    model = neural_interface.build_neural_network()
    assert isinstance(model, tf.keras.Model)
    assert len(model.layers) > 0

def test_process_quantum_state(neural_interface):
    """Test processing of quantum states."""
    quantum_state = neural_interface.quantum_processor.create_quantum_state()
    processed_state = neural_interface.process_quantum_state(quantum_state)
    assert processed_state is not None
    assert isinstance(processed_state, np.ndarray)

def test_process_holographic_data(neural_interface):
    """Test processing of holographic data."""
    hologram = neural_interface.holographic_processor.create_hologram()
    processed_data = neural_interface.process_holographic_data(hologram)
    assert processed_data is not None
    assert isinstance(processed_data, np.ndarray)

def test_neural_processing(neural_interface):
    """Test neural network processing."""
    quantum_state = neural_interface.quantum_processor.create_quantum_state()
    hologram = neural_interface.holographic_processor.create_hologram()
    predictions = neural_interface.neural_processing(quantum_state, hologram)
    assert predictions is not None
    assert isinstance(predictions, np.ndarray)

def test_system_integration(neural_interface):
    """Test integration of quantum and holographic systems."""
    integrated_state = neural_interface.system_integration()
    assert integrated_state is not None
    assert isinstance(integrated_state, dict)
    assert all(key in integrated_state for key in ['quantum_state', 'holographic_state', 'neural_output'])

def test_get_interface_status(neural_interface):
    """Test retrieval of interface status."""
    status = neural_interface.get_interface_status()
    assert isinstance(status, dict)
    assert all(key in status for key in ['model_state', 'processing_status', 'integration_metrics'])

def test_reset_interface(neural_interface):
    """Test reset of neural interface."""
    initial_state = neural_interface.system_integration()
    neural_interface.reset_interface()
    new_state = neural_interface.system_integration()
    assert new_state is not None
    assert isinstance(new_state, dict)
    assert not np.array_equal(
        new_state['neural_output'],
        initial_state['neural_output']
    ) 
import pytest
import numpy as np
from src.core.synchronization_manager import SynchronizationManager
from src.core.quantum_processor import QuantumProcessor
from src.core.holographic_processor import HolographicProcessor
from src.core.neural_interface import NeuralInterface
from src.core.quantum_holographic_entanglement import QuantumHolographicEntanglement

@pytest.fixture
def sync_manager():
    quantum_processor = QuantumProcessor(num_qubits=4)
    holographic_processor = HolographicProcessor(
        wavelength=633e-9,
        pixel_size=10e-6,
        distance=0.1
    )
    neural_interface = NeuralInterface(
        quantum_processor=quantum_processor,
        holographic_processor=holographic_processor
    )
    qhe_processor = QuantumHolographicEntanglement(
        quantum_processor=quantum_processor,
        holographic_processor=holographic_processor
    )
    return SynchronizationManager(
        quantum_processor=quantum_processor,
        holographic_processor=holographic_processor,
        neural_interface=neural_interface,
        qhe_processor=qhe_processor
    )

def test_synchronize_systems(sync_manager):
    """Test synchronization of all system components."""
    sync_state = sync_manager.synchronize_systems()
    assert sync_state is not None
    assert isinstance(sync_state, dict)
    assert all(key in sync_state for key in ['quantum', 'holographic', 'neural', 'entanglement'])

def test_propagate_synchronized_state(sync_manager):
    """Test propagation of synchronized state."""
    initial_state = sync_manager.synchronize_systems()
    propagated_state = sync_manager.propagate_synchronized_state(initial_state)
    assert propagated_state is not None
    assert isinstance(propagated_state, dict)
    assert propagated_state.keys() == initial_state.keys()

def test_measure_synchronization(sync_manager):
    """Test measurement of synchronization metrics."""
    state = sync_manager.synchronize_systems()
    sync_score = sync_manager.measure_synchronization(state)
    assert isinstance(sync_score, float)
    assert 0 <= sync_score <= 1

def test_get_system_status(sync_manager):
    """Test retrieval of system status."""
    status = sync_manager.get_system_status()
    assert isinstance(status, dict)
    assert all(key in status for key in ['sync_state', 'measurements', 'components_status'])

def test_reset_synchronization(sync_manager):
    """Test reset of synchronization state."""
    initial_state = sync_manager.synchronize_systems()
    sync_manager.reset_synchronization()
    new_state = sync_manager.synchronize_systems()
    assert new_state is not None
    assert isinstance(new_state, dict)
    # States should be different after reset
    assert not np.array_equal(
        new_state['quantum'],
        initial_state['quantum']
    ) 
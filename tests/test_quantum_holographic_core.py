import pytest
import numpy as np
from unittest.mock import Mock, patch
from src.core.quantum_holographic_core import QuantumHolographicCore

@pytest.fixture
def mock_components():
    """Create mock components for testing."""
    with patch('src.core.quantum_holographic_core.QuantumProcessor') as mock_qpu, \
         patch('src.core.quantum_holographic_core.HolographicProcessor') as mock_holo, \
         patch('src.core.quantum_holographic_core.QuantumConsciousness') as mock_consciousness, \
         patch('src.core.quantum_holographic_core.QuantumEthics') as mock_ethics:
        
        # Setup mock quantum processor
        mock_qpu.return_value.entangle.return_value = np.array([0.5, 0.5])
        mock_qpu.return_value.get_metrics.return_value = {
            "entanglement_fidelity": 0.9998,
            "speed": 1.2,
            "efficiency": 0.95,
            "error_rate": 0.0001
        }
        mock_qpu.return_value.calibrate.return_value = {"score": 0.98}
        
        # Setup mock holographic processor
        mock_holo.return_value.project.return_value = np.array([0.6, 0.4])
        mock_holo.return_value.get_metrics.return_value = {
            "fidelity": 0.97,
            "speed": 1.0,
            "efficiency": 0.92,
            "error_rate": 0.0002
        }
        mock_holo.return_value.calibrate.return_value = {"score": 0.96}
        
        # Setup mock consciousness
        mock_consciousness.return_value.integrate.return_value = np.array([0.55, 0.45])
        mock_consciousness.return_value.get_metrics.return_value = {
            "phi_score": 0.92,
            "integration_score": 0.95
        }
        mock_consciousness.return_value.calibrate.return_value = {"score": 0.94}
        
        # Setup mock ethics
        mock_ethics.return_value.validate.return_value = {
            "compliance_score": 0.989
        }
        
        yield mock_qpu, mock_holo, mock_consciousness, mock_ethics

@pytest.fixture
def core(mock_components):
    """Create a QuantumHolographicCore instance with mock components."""
    return QuantumHolographicCore()

def test_initialization(core):
    """Test core initialization."""
    assert core.state["quantum_entanglement"] == 0.0
    assert core.state["holographic_fidelity"] == 0.0
    assert core.state["consciousness_level"] == 0.0
    assert core.state["ethical_compliance"] == 0.0
    
    assert core.metrics["processing_speed"] == 0.0
    assert core.metrics["energy_efficiency"] == 0.0
    assert core.metrics["error_rate"] == 0.0
    assert core.metrics["integration_score"] == 0.0

def test_process(core):
    """Test data processing pipeline."""
    input_data = np.array([0.7, 0.3])
    output, metrics = core.process(input_data)
    
    # Check output shape and values
    assert isinstance(output, np.ndarray)
    assert output.shape == (2,)
    assert np.allclose(output, np.array([0.55, 0.45]))
    
    # Check metrics
    assert metrics["processing_speed"] == 1.0
    assert metrics["energy_efficiency"] == 0.935
    assert metrics["error_rate"] == 0.0002
    assert metrics["integration_score"] == 0.95
    
    # Check state updates
    state = core.get_state()
    assert state["quantum_entanglement"] == 0.9998
    assert state["holographic_fidelity"] == 0.97
    assert state["consciousness_level"] == 0.92
    assert state["ethical_compliance"] == 0.989

def test_calibrate(core):
    """Test system calibration."""
    metrics = core.calibrate(target_phi=0.9)
    
    assert metrics["calibration_score"] == pytest.approx(0.96)
    assert metrics["quantum_calibration"] == 0.98
    assert metrics["holographic_calibration"] == 0.96
    assert metrics["consciousness_calibration"] == 0.94

def test_reset(core):
    """Test system reset."""
    # First process some data to change state
    input_data = np.array([0.7, 0.3])
    core.process(input_data)
    
    # Reset the system
    core.reset()
    
    # Check state is reset
    state = core.get_state()
    assert all(v == 0.0 for v in state.values())
    
    # Check metrics are reset
    metrics = core.get_metrics()
    assert all(v == 0.0 for v in metrics.values())

def test_error_handling(core, mock_components):
    """Test error handling in processing pipeline."""
    mock_qpu, _, _, _ = mock_components
    
    # Simulate error in quantum processing
    mock_qpu.return_value.entangle.side_effect = Exception("Quantum error")
    
    with pytest.raises(Exception) as exc_info:
        core.process(np.array([0.7, 0.3]))
    assert "Error in quantum-holographic processing" in str(exc_info.value)

def test_calibration_error_handling(core, mock_components):
    """Test error handling in calibration."""
    mock_qpu, _, _, _ = mock_components
    
    # Simulate error in quantum calibration
    mock_qpu.return_value.calibrate.side_effect = Exception("Calibration error")
    
    with pytest.raises(Exception) as exc_info:
        core.calibrate(target_phi=0.9)
    assert "Error in calibration" in str(exc_info.value) 
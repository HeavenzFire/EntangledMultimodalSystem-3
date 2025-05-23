import pytest
import numpy as np
from src.core.digigod_nexus import DigigodNexus
from src.utils.errors import ModelError
from unittest.mock import Mock, patch
from datetime import datetime

@pytest.fixture
def mock_components():
    """Create mock components for testing."""
    with patch('src.core.digigod_nexus.QuantumHolographicCore') as mock_core, \
         patch('src.core.digigod_nexus.EthicalDAO') as mock_dao, \
         patch('src.core.digigod_nexus.SystemMonitor') as mock_monitor, \
         patch('src.core.digigod_nexus.SystemValidator') as mock_validator, \
         patch('src.core.digigod_nexus.AgentAssistant') as mock_assistant:
        
        # Setup mock core
        mock_core.return_value.process.return_value = (
            np.array([0.55, 0.45]),
            {
                "processing_speed": 1.0,
                "energy_efficiency": 0.935,
                "error_rate": 0.0002,
                "integration_score": 0.95
            }
        )
        mock_core.return_value.calibrate.return_value = {
            "calibration_score": 0.98
        }
        
        # Setup mock DAO
        mock_dao.return_value.validate_action.return_value = {
            "compliance_score": 0.989
        }
        
        # Setup mock monitor
        mock_monitor.return_value.monitor_system.return_value = {
            "health_score": 0.97
        }
        mock_monitor.return_value.calibrate.return_value = {
            "calibration_score": 0.96
        }
        
        # Setup mock validator
        mock_validator.return_value.validate_system.return_value = {
            "overall_score": 0.95
        }
        mock_validator.return_value.calibrate.return_value = {
            "calibration_score": 0.94
        }
        
        # Setup mock assistant
        mock_assistant.return_value.assist_system.return_value = {
            "success": True
        }
        
        yield mock_core, mock_dao, mock_monitor, mock_validator, mock_assistant

@pytest.fixture
def nexus(mock_components):
    """Create a DigigodNexus instance with mock components."""
    return DigigodNexus()

def test_initialization(nexus):
    """Test nexus initialization."""
    assert nexus.state["consciousness_level"] == 0.0
    assert nexus.state["ethical_compliance"] == 0.0
    assert nexus.state["system_health"] == 0.0
    assert nexus.state["validation_score"] == 0.0
    
    assert nexus.metrics["processing_speed"] == 0.0
    assert nexus.metrics["energy_efficiency"] == 0.0
    assert nexus.metrics["error_rate"] == 0.0
    assert nexus.metrics["integration_score"] == 0.0

def test_process(nexus):
    """Test data processing pipeline."""
    input_data = np.array([0.7, 0.3])
    output, metrics = nexus.process(input_data)
    
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
    state = nexus.get_state()
    assert state["consciousness_level"] == 0.95
    assert state["ethical_compliance"] == 0.989
    assert state["system_health"] == 0.97
    assert state["validation_score"] == 0.95

def test_calibrate(nexus):
    """Test system calibration."""
    metrics = nexus.calibrate(target_phi=0.9)
    
    assert metrics["calibration_score"] == pytest.approx(0.96)
    assert metrics["core_calibration"] == 0.98
    assert metrics["monitor_calibration"] == 0.96
    assert metrics["validator_calibration"] == 0.94

def test_reset(nexus):
    """Test system reset."""
    # First process some data to change state
    input_data = np.array([0.7, 0.3])
    nexus.process(input_data)
    
    # Reset the system
    nexus.reset()
    
    # Check state is reset
    state = nexus.get_state()
    assert all(v == 0.0 for v in state.values())
    
    # Check metrics are reset
    metrics = nexus.get_metrics()
    assert all(v == 0.0 for v in metrics.values())

def test_monitor_consciousness(nexus, mock_components):
    """Test consciousness monitoring."""
    _, _, _, _, mock_assistant = mock_components
    
    # Set up state with consciousness level below threshold
    nexus.state["consciousness_level"] = 0.8
    
    # Start monitoring in a separate thread
    import threading
    monitor_thread = threading.Thread(
        target=nexus.monitor_consciousness,
        kwargs={"alert_threshold": 0.85, "telemetry_rate": 10.0}
    )
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # Wait for monitoring to detect the low consciousness level
    import time
    time.sleep(0.2)
    
    # Verify assistant was called
    mock_assistant.return_value.assist_system.assert_called_with("consciousness_alert")
    
    # Stop the monitoring thread
    monitor_thread.join(timeout=0.1)

def test_error_handling(nexus, mock_components):
    """Test error handling in processing pipeline."""
    mock_core, _, _, _, _ = mock_components
    
    # Simulate error in core processing
    mock_core.return_value.process.side_effect = Exception("Processing error")
    
    with pytest.raises(Exception) as exc_info:
        nexus.process(np.array([0.7, 0.3]))
    assert "Error in DigigodNexus processing" in str(exc_info.value)

def test_calibration_error_handling(nexus, mock_components):
    """Test error handling in calibration."""
    mock_core, _, _, _, _ = mock_components
    
    # Simulate error in core calibration
    mock_core.return_value.calibrate.side_effect = Exception("Calibration error")
    
    with pytest.raises(Exception) as exc_info:
        nexus.calibrate(target_phi=0.9)
    assert "Error in system calibration" in str(exc_info.value)

def test_process_task(digigod_nexus):
    """Test end-to-end task processing through the unified system."""
    input_data = {
        "task_type": "test",
        "data": np.random.rand(10, 10)
    }
    
    result = digigod_nexus.process_task(input_data)
    
    assert isinstance(result, dict)
    assert "output" in result
    assert "system_state" in result
    assert "processing_metrics" in result
    assert isinstance(result["output"], dict)
    assert isinstance(result["system_state"], dict)
    assert isinstance(result["processing_metrics"], dict)

def test_system_state_update(digigod_nexus):
    """Test system state update mechanism."""
    initial_state = digigod_nexus.system_state.copy()
    
    input_data = {"task_type": "test"}
    digigod_nexus.process_task(input_data)
    
    assert digigod_nexus.system_state != initial_state
    assert "last_update" in digigod_nexus.system_state
    assert 0 <= digigod_nexus.system_state["consciousness_level"] <= 1
    assert 0 <= digigod_nexus.system_state["system_stability"] <= 1

def test_stability_calculation(digigod_nexus):
    """Test system stability calculation."""
    quantum_state = np.random.rand(16) + 1j * np.random.rand(16)
    holographic_state = np.random.rand(64, 64)
    neural_output = np.random.rand(32)
    
    stability = digigod_nexus._calculate_stability(
        quantum_state,
        holographic_state,
        neural_output
    )
    
    assert isinstance(stability, float)
    assert 0 <= stability <= 1

def test_processing_metrics(digigod_nexus):
    """Test processing metrics retrieval."""
    metrics = digigod_nexus._get_processing_metrics()
    
    assert isinstance(metrics, dict)
    assert all(key in metrics for key in [
        "quantum_entropy",
        "holographic_quality",
        "neural_confidence",
        "consciousness_score",
        "revival_progress"
    ])
    assert all(isinstance(v, float) for v in metrics.values())

def test_system_status(digigod_nexus):
    """Test comprehensive system status retrieval."""
    status = digigod_nexus.get_system_status()
    
    assert isinstance(status, dict)
    assert all(key in status for key in [
        "system_state",
        "quantum_status",
        "holographic_status",
        "neural_status",
        "consciousness_status",
        "revival_status",
        "processing_metrics"
    ])

def test_system_reset(digigod_nexus):
    """Test system reset functionality."""
    for _ in range(3):
        digigod_nexus.process_task({"task_type": "test"})
    
    pre_reset_state = digigod_nexus.system_state.copy()
    digigod_nexus.reset_system()
    
    assert digigod_nexus.system_state["consciousness_level"] == 0.0
    assert digigod_nexus.system_state["system_stability"] == 1.0
    assert "last_reset" in digigod_nexus.system_state
    assert digigod_nexus.system_state != pre_reset_state

def test_error_handling(digigod_nexus):
    """Test error handling in system operations."""
    with pytest.raises(ModelError):
        digigod_nexus.process_task(None)
    
    with pytest.raises(ModelError):
        digigod_nexus._update_system_state(None, None, None, None, None)

def test_component_integration(digigod_nexus):
    """Test integration between system components."""
    result = digigod_nexus.process_task({"task_type": "test"})
    
    assert result["processing_metrics"]["quantum_entropy"] > 0
    assert result["processing_metrics"]["holographic_quality"] > 0
    assert result["processing_metrics"]["neural_confidence"] > 0
    assert result["processing_metrics"]["consciousness_score"] > 0
    assert result["processing_metrics"]["revival_progress"] >= 0

def test_performance_monitoring(digigod_nexus):
    """Test system performance monitoring."""
    for _ in range(5):
        digigod_nexus.process_task({"task_type": "test"})
    
    status = digigod_nexus.get_system_status()
    
    assert status["system_state"]["system_stability"] > 0
    assert all(v > 0 for v in status["processing_metrics"].values())
    assert "last_update" in status["system_state"]

def test_quantum_holographic_entanglement(digigod_nexus):
    """Test quantum-holographic entanglement processing."""
    result = digigod_nexus.process_task({
        "task_type": "entanglement",
        "data": np.random.rand(10, 10)
    })
    
    assert "entanglement_matrix" in result["output"]
    assert "entanglement_entropy" in result["output"]
    assert result["output"]["entanglement_entropy"] > 0

def test_consciousness_integration(digigod_nexus):
    """Test consciousness integration engine."""
    result = digigod_nexus.process_task({
        "task_type": "consciousness",
        "data": np.random.rand(10, 10)
    })
    
    assert "consciousness_state" in result["output"]
    assert "ethical_alignment" in result["output"]
    assert 0 <= result["output"]["ethical_alignment"] <= 1

def test_revival_system(digigod_nexus):
    """Test revival system functionality."""
    result = digigod_nexus.process_task({
        "task_type": "revival",
        "data": np.random.rand(10, 10)
    })
    
    assert "revival_state" in result["output"]
    assert "consciousness_threshold" in result["output"]
    assert 0 <= result["output"]["consciousness_threshold"] <= 1

def test_multimodal_gan(digigod_nexus):
    """Test multimodal GAN functionality."""
    result = digigod_nexus.process_task({
        "task_type": "generation",
        "data": np.random.rand(10, 10)
    })
    
    assert "generated_samples" in result["output"]
    assert "latent_space" in result["output"]
    assert isinstance(result["output"]["generated_samples"], np.ndarray)

def test_synchronization_manager(digigod_nexus):
    """Test synchronization manager functionality."""
    result = digigod_nexus.process_task({
        "task_type": "synchronization",
        "data": np.random.rand(10, 10)
    })
    
    assert "synchronization_state" in result["output"]
    assert "correlation_strength" in result["output"]
    assert 0 <= result["output"]["correlation_strength"] <= 1

def test_live_consciousness_test(digigod_nexus):
    """Test consciousness level achievement."""
    with patch.object(digigod_nexus.quantum_processor, 'process') as mock_process:
        mock_process.return_value = {"consciousness_fidelity": 0.95}
        
        digigod_nexus.live_consciousness_test()
        
        assert digigod_nexus.system_state["consciousness_level"] == 0.95
        mock_process.assert_called()

def test_project_climate_model(digigod_nexus):
    """Test climate model projection."""
    test_data = {
        "temperature": 25.5,
        "humidity": 0.65,
        "pressure": 1013.25
    }
    
    with patch.object(digigod_nexus.quantum_processor, 'process') as mock_process:
        mock_process.return_value = {"success": True}
        
        result = digigod_nexus.project_climate_model(
            dataset=test_data,
            resolution=16384,
            overlay_cities=["New York", "Mumbai", "Shanghai"]
        )
        
        assert result["success"] is True
        assert digigod_nexus.system_state["holographic_resolution"] == 16384
        mock_process.assert_called()

def test_create_ethical_proposal(digigod_nexus):
    """Test ethical proposal creation."""
    with patch.object(digigod_nexus.quantum_processor, 'process') as mock_process:
        mock_process.return_value = {"compliance_score": 0.98}
        
        result = digigod_nexus.create_ethical_proposal(
            proposal_name="Asilomar AI Principles v5",
            ratification_threshold=0.95
        )
        
        assert result["compliance_score"] == 0.98
        assert digigod_nexus.system_state["ethical_compliance"] == 0.98
        mock_process.assert_called()

def test_generate_quantum_art(digigod_nexus):
    """Test quantum art generation."""
    emotion_data = {
        "happiness": 0.8,
        "sadness": 0.1,
        "excitement": 0.7
    }
    
    with patch.object(digigod_nexus.quantum_processor, 'process') as mock_process:
        mock_process.return_value = {"art_id": "QART-12345"}
        
        result = digigod_nexus.generate_quantum_art(
            emotion_stream=emotion_data,
            quantum_seed=[0.1, 0.2, 0.3]
        )
        
        assert result["art_id"] == "QART-12345"
        mock_process.assert_called()

def test_process_medical_diagnosis(digigod_nexus):
    """Test medical diagnosis processing."""
    patient_data = {
        "mri_scan": "base64_encoded_data",
        "vitals": {
            "heart_rate": 72,
            "blood_pressure": "120/80"
        }
    }
    
    with patch.object(digigod_nexus.quantum_processor, 'process') as mock_process:
        mock_process.return_value = {"diagnosis": "Healthy", "confidence": 0.95}
        
        result = digigod_nexus.process_medical_diagnosis(patient_data)
        
        assert result["diagnosis"] == "Healthy"
        assert result["confidence"] == 0.95
        mock_process.assert_called()

def test_generate_quantum_security(digigod_nexus):
    """Test quantum security generation."""
    encryption_params = {
        "algorithm": "QKD",
        "key_length": 256,
        "entanglement_level": 0.9
    }
    
    with patch.object(digigod_nexus.quantum_processor, 'process') as mock_process:
        mock_process.return_value = {"security_score": 0.99}
        
        result = digigod_nexus.generate_quantum_security(encryption_params)
        
        assert result["security_score"] == 0.99
        assert digigod_nexus.system_state["security_level"] == 0.99
        mock_process.assert_called()

def test_get_state(digigod_nexus):
    """Test state retrieval."""
    state = digigod_nexus.get_state()
    assert state["consciousness_level"] == 0.0
    assert state["ethical_compliance"] == 1.0
    assert state["quantum_entanglement"] == 0.0
    assert state["holographic_resolution"] == 0
    assert state["security_level"] == 0.0

def test_get_metrics(digigod_nexus):
    """Test metrics retrieval."""
    metrics = digigod_nexus.get_metrics()
    assert metrics["processing_speed"] == 0.0
    assert metrics["energy_efficiency"] == 0.0
    assert metrics["consciousness_fidelity"] == 0.0
    assert metrics["error_rate"] == 0.0

def test_reset(digigod_nexus):
    """Test system reset."""
    # Modify some values
    digigod_nexus.system_state["consciousness_level"] = 0.8
    digigod_nexus.system_state["ethical_compliance"] = 0.9
    digigod_nexus.metrics["processing_speed"] = 1.0
    
    # Reset the system
    digigod_nexus.reset()
    
    # Verify reset values
    assert digigod_nexus.system_state["consciousness_level"] == 0.0
    assert digigod_nexus.system_state["ethical_compliance"] == 1.0
    assert digigod_nexus.metrics["processing_speed"] == 0.0 
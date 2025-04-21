import pytest
import numpy as np
from src.core.digigod_nexus import DigigodNexus
from src.utils.errors import ModelError

@pytest.fixture
def digigod_nexus():
    """Fixture to create a DigigodNexus instance for testing."""
    return DigigodNexus()

def test_initialization(digigod_nexus):
    """Test proper initialization of all core components."""
    assert digigod_nexus.quantum_processor is not None
    assert digigod_nexus.holographic_processor is not None
    assert digigod_nexus.neural_interface is not None
    assert digigod_nexus.qhe_processor is not None
    assert digigod_nexus.sync_manager is not None
    assert digigod_nexus.consciousness_engine is not None
    assert digigod_nexus.multimodal_gan is not None
    assert digigod_nexus.revival_system is not None
    assert isinstance(digigod_nexus.system_state, dict)

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
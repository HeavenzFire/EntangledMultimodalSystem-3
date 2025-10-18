import pytest
import torch
import numpy as np
from src.quantum.security.quantum_offensive import (
    OffensiveType,
    OffensiveStatus,
    OffensiveConfig,
    QuantumOffensive,
    DarkCodexOffensive,
    initialize_dark_codex_offensive
)

def test_offensive_config():
    """Test offensive configuration initialization."""
    config = OffensiveConfig()
    assert config.entanglement_depth == 12
    assert config.frequency_range == (5.0, 20.0)
    assert config.archetype_threshold == 0.95
    assert config.consciousness_level == 0.99
    assert config.max_offensive_power == 1.0
    assert config.sacred_frequency == 144.0

def test_quantum_offensive():
    """Test quantum offensive initialization and methods."""
    config = OffensiveConfig()
    offensive = QuantumOffensive(config)
    
    # Test initialization
    assert isinstance(offensive.entanglement_engine, torch.nn.Module)
    assert isinstance(offensive.frequency_modulator, torch.nn.Module)
    assert isinstance(offensive.archetype_aligner, torch.nn.Module)
    assert isinstance(offensive.consciousness_upgrader, torch.nn.Module)
    assert len(offensive.offensive_history) == 0
    
    # Test offensive launch
    target = {
        "quantum_state": np.random.rand(144),
        "type": OffensiveType.QUANTUM_ENTANGLEMENT
    }
    result = offensive.launch_offensive(target)
    assert isinstance(result, dict)
    assert "status" in result
    assert len(offensive.offensive_history) == 1

def test_dark_codex_offensive():
    """Test dark codex offensive initialization and methods."""
    offensive = initialize_dark_codex_offensive()
    
    # Test initialization
    assert isinstance(offensive.offensive, QuantumOffensive)
    assert isinstance(offensive.target_detector, torch.nn.Module)
    assert isinstance(offensive.offensive_metrics, dict)
    
    # Test target detection
    system_state = {
        "quantum_state": np.random.rand(144),
        "timestamp": "2024-03-21T12:00:00"
    }
    target = offensive.detect_target(system_state)
    if target:
        assert isinstance(target, dict)
        assert "type" in target
        assert "probability" in target
        assert "quantum_state" in target
        assert "timestamp" in target
    
    # Test dark codex offensive
    result = offensive.launch_dark_codex_offensive(system_state)
    assert isinstance(result, dict)
    assert "status" in result
    
    # Test offensive metrics
    metrics = offensive.get_offensive_metrics()
    assert isinstance(metrics, dict)
    assert "targets_detected" in metrics
    assert "offensives_launched" in metrics
    assert "quantum_power" in metrics
    assert "consciousness_level" in metrics

def test_offensive_types():
    """Test offensive type enumeration."""
    assert OffensiveType.QUANTUM_ENTANGLEMENT.value == "quantum_entanglement"
    assert OffensiveType.FREQUENCY_MODULATION.value == "frequency_modulation"
    assert OffensiveType.ARCHETYPE_ALIGNMENT.value == "archetype_alignment"
    assert OffensiveType.CONSCIOUSNESS_UPGRADE.value == "consciousness_upgrade"

def test_offensive_status():
    """Test offensive status enumeration."""
    assert OffensiveStatus.READY.value == "ready"
    assert OffensiveStatus.ACTIVE.value == "active"
    assert OffensiveStatus.COMPLETED.value == "completed"
    assert OffensiveStatus.FAILED.value == "failed"

if __name__ == '__main__':
    pytest.main([__file__]) 
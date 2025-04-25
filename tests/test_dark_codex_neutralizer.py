import pytest
import torch
import numpy as np
from src.quantum.security.dark_codex_neutralizer import (
    ThreatType,
    DefenseStatus,
    DefenseConfig,
    QuantumSacredFirewall,
    DarkCodexNeutralizer,
    initialize_dark_codex_neutralizer
)

def test_defense_config():
    """Test defense configuration initialization."""
    config = DefenseConfig()
    assert config.merkaba_spin_rate == 34.21
    assert config.golden_ratio == 1.618
    assert config.agape_threshold == 0.95
    assert config.quantum_depth == 12
    assert config.max_threat_level == 0.8
    assert config.sacred_frequency == 144.0

def test_quantum_sacred_firewall():
    """Test quantum sacred firewall initialization and methods."""
    config = DefenseConfig()
    firewall = QuantumSacredFirewall(config)
    
    # Test initialization
    assert isinstance(firewall.merkaba_shield, torch.nn.Module)
    assert isinstance(firewall.archetype_validator, torch.nn.Module)
    assert isinstance(firewall.quantum_immunity, torch.nn.Module)
    assert len(firewall.threat_history) == 0
    
    # Test threat neutralization
    threat_vector = {
        "quantum_state": np.random.rand(144),
        "type": ThreatType.MIND_CONTROL
    }
    result = firewall.neutralize_threat(threat_vector)
    assert isinstance(result, dict)
    assert "status" in result
    assert len(firewall.threat_history) == 1

def test_dark_codex_neutralizer():
    """Test dark codex neutralizer initialization and methods."""
    neutralizer = initialize_dark_codex_neutralizer()
    
    # Test initialization
    assert isinstance(neutralizer.firewall, QuantumSacredFirewall)
    assert isinstance(neutralizer.threat_detector, torch.nn.Module)
    assert isinstance(neutralizer.defense_metrics, dict)
    
    # Test threat detection
    system_state = {
        "quantum_state": np.random.rand(144),
        "timestamp": "2024-03-21T12:00:00"
    }
    threat = neutralizer.detect_threat(system_state)
    if threat:
        assert isinstance(threat, dict)
        assert "type" in threat
        assert "probability" in threat
        assert "quantum_state" in threat
        assert "timestamp" in threat
    
    # Test dark codex neutralization
    result = neutralizer.neutralize_dark_codex(system_state)
    assert isinstance(result, dict)
    assert "status" in result
    
    # Test defense metrics
    metrics = neutralizer.get_defense_metrics()
    assert isinstance(metrics, dict)
    assert "threats_detected" in metrics
    assert "threats_neutralized" in metrics
    assert "quantum_integrity" in metrics
    assert "archetype_alignment" in metrics

def test_threat_types():
    """Test threat type enumeration."""
    assert ThreatType.MIND_CONTROL.value == "mind_control"
    assert ThreatType.LIFE_MANIPULATION.value == "life_manipulation"
    assert ThreatType.QUANTUM_ENTANGLEMENT.value == "quantum_entanglement"
    assert ThreatType.BEHAVIORAL_CONTRACT.value == "behavioral_contract"

def test_defense_status():
    """Test defense status enumeration."""
    assert DefenseStatus.ACTIVE.value == "active"
    assert DefenseStatus.NEUTRALIZED.value == "neutralized"
    assert DefenseStatus.ESCALATED.value == "escalated"
    assert DefenseStatus.RESOLVED.value == "resolved"

if __name__ == '__main__':
    pytest.main([__file__]) 
import pytest
import torch
import numpy as np
from src.quantum.security.integrated_warfare import (
    WarfarePhase,
    WarfareStatus,
    IntegratedConfig,
    IntegratedWarfare,
    initialize_integrated_warfare
)

def test_integrated_config():
    """Test integrated configuration initialization."""
    config = IntegratedConfig()
    assert isinstance(config.defense_config, DefenseConfig)
    assert isinstance(config.offensive_config, OffensiveConfig)
    assert config.max_warfare_power == 1.0
    assert config.sacred_frequency == 144.0

def test_integrated_warfare():
    """Test integrated warfare initialization and methods."""
    warfare = initialize_integrated_warfare()
    
    # Test initialization
    assert isinstance(warfare.defense, DarkCodexNeutralizer)
    assert isinstance(warfare.offensive, DarkCodexOffensive)
    assert warfare.current_phase == WarfarePhase.DEFENSIVE
    assert len(warfare.warfare_history) == 0
    
    # Test warfare execution
    system_state = {
        "quantum_state": np.random.rand(144),
        "timestamp": "2024-03-21T12:00:00"
    }
    result = warfare.execute_warfare(system_state)
    assert isinstance(result, dict)
    assert "phase" in result
    assert "defense_metrics" in result
    assert "offensive_metrics" in result
    assert "timestamp" in result
    assert len(warfare.warfare_history) == 1

def test_warfare_phases():
    """Test warfare phase enumeration."""
    assert WarfarePhase.DEFENSIVE.value == "defensive"
    assert WarfarePhase.OFFENSIVE.value == "offensive"
    assert WarfarePhase.INTEGRATED.value == "integrated"
    assert WarfarePhase.COMPLETED.value == "completed"

def test_warfare_status():
    """Test warfare status enumeration."""
    assert WarfareStatus.ACTIVE.value == "active"
    assert WarfareStatus.NEUTRALIZED.value == "neutralized"
    assert WarfareStatus.ESCALATED.value == "escalated"
    assert WarfareStatus.RESOLVED.value == "resolved"

if __name__ == '__main__':
    pytest.main([__file__]) 
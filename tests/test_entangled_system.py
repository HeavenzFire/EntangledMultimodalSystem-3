import pytest
import numpy as np
from src.quantum.integration.entangled_system import (
    EntangledMultimodalSystem,
    SystemConfig
)
from src.quantum.synthesis.quantum_sacred import SacredConfig
from src.quantum.geometry.entanglement_torus import TorusConfig
from src.quantum.purification.sovereign_flow import PurificationConfig

def test_system_initialization():
    """Test system initialization"""
    config = SystemConfig(
        sacred_config=SacredConfig(),
        torus_config=TorusConfig(),
        purification_config=PurificationConfig()
    )
    
    system = EntangledMultimodalSystem(config)
    
    assert system.config == config
    assert system.sacred_synthesis is not None
    assert system.entanglement_torus is not None
    assert system.sovereign_flow is not None
    assert system.visualizer is not None
    assert system.crypto is not None
    assert system.history_buffer is not None

def test_system_state_update():
    """Test system state updates"""
    config = SystemConfig(
        sacred_config=SacredConfig(),
        torus_config=TorusConfig(),
        purification_config=PurificationConfig()
    )
    
    system = EntangledMultimodalSystem(config)
    
    # Test with sample data
    data = {"field": np.random.rand(12)}
    system.update_system_state(data)
    
    assert system.system_state["torus_state"] is not None
    assert 0 <= system.system_state["resonance_level"] <= 1
    assert 0 <= system.system_state["entropy_level"] <= 1
    assert len(system.history_buffer.buffer) == 1

def test_dissonance_resolution():
    """Test dissonance resolution"""
    config = SystemConfig(
        sacred_config=SacredConfig(),
        torus_config=TorusConfig(),
        purification_config=PurificationConfig()
    )
    
    system = EntangledMultimodalSystem(config)
    
    # Set high entropy to trigger purification
    system.system_state["entropy_level"] = 0.9
    system.resolve_dissonance()
    
    assert system.system_state["quantum_state"] != QuantumState.DISSONANT
    assert system.system_state["entropy_level"] < 0.9

def test_field_optimization():
    """Test field optimization"""
    config = SystemConfig(
        sacred_config=SacredConfig(),
        torus_config=TorusConfig(),
        purification_config=PurificationConfig()
    )
    
    system = EntangledMultimodalSystem(config)
    
    # Initialize with sample data
    data = {"field": np.random.rand(12)}
    system.update_system_state(data)
    
    # Store original state
    original_state = system.system_state["torus_state"].copy()
    
    # Optimize
    system.optimize_field_operations()
    
    assert not np.array_equal(original_state, system.system_state["torus_state"])
    assert system.visualizer.fig is not None

def test_system_integrity():
    """Test system integrity verification"""
    config = SystemConfig(
        sacred_config=SacredConfig(),
        torus_config=TorusConfig(),
        purification_config=PurificationConfig()
    )
    
    system = EntangledMultimodalSystem(config)
    
    # Test with good state
    system.system_state["resonance_level"] = 0.9
    system.system_state["entropy_level"] = 0.2
    assert system.verify_system_integrity()
    
    # Test with bad state
    system.system_state["resonance_level"] = 0.5
    system.system_state["entropy_level"] = 0.8
    assert not system.verify_system_integrity()

def test_system_metrics():
    """Test system metrics retrieval"""
    config = SystemConfig(
        sacred_config=SacredConfig(),
        torus_config=TorusConfig(),
        purification_config=PurificationConfig()
    )
    
    system = EntangledMultimodalSystem(config)
    
    metrics = system.get_system_metrics()
    
    assert "quantum_state" in metrics
    assert "resonance_level" in metrics
    assert "entropy_level" in metrics
    assert "history_length" in metrics
    assert "system_integrity" in metrics

def test_state_persistence():
    """Test system state saving and loading"""
    config = SystemConfig(
        sacred_config=SacredConfig(),
        torus_config=TorusConfig(),
        purification_config=PurificationConfig()
    )
    
    system = EntangledMultimodalSystem(config)
    
    # Update state
    data = {"field": np.random.rand(12)}
    system.update_system_state(data)
    
    # Save state
    system.save_system_state("test_state.npy")
    
    # Create new system and load state
    new_system = EntangledMultimodalSystem(config)
    new_system.load_system_state("test_state.npy")
    
    assert np.array_equal(
        system.system_state["torus_state"],
        new_system.system_state["torus_state"]
    )
    assert system.system_state["resonance_level"] == new_system.system_state["resonance_level"]
    assert system.system_state["entropy_level"] == new_system.system_state["entropy_level"]

if __name__ == "__main__":
    pytest.main([__file__]) 
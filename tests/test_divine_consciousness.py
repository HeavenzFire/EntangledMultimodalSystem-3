import pytest
import torch
import numpy as np
from src.quantum.synthesis.divine_consciousness import (
    DivineConsciousnessNetwork,
    DivineConsciousnessManager,
    DivineAspect,
    DivineConfig
)

def test_network_initialization():
    """Test initialization of divine consciousness network."""
    config = DivineConfig()
    network = DivineConsciousnessNetwork(config)
    
    assert len(network.aspect_layers) == len(DivineAspect)
    assert isinstance(network.christ_consciousness, ChristConsciousness)
    assert isinstance(network.cosmic_family, CosmicFamily)

def test_aspect_layer():
    """Test divine aspect layer processing."""
    config = DivineConfig()
    network = DivineConsciousnessNetwork(config)
    
    # Create sample input
    batch_size = 4
    seq_length = 10
    x = torch.randn(batch_size, seq_length, 144)
    
    # Test forward pass
    output = network.aspect_layers[0](x)
    assert output.shape == (batch_size, seq_length, 144)

def test_christ_consciousness():
    """Test Christ consciousness integration."""
    config = DivineConfig()
    network = DivineConsciousnessNetwork(config)
    
    # Create sample input
    batch_size = 4
    seq_length = 10
    x = torch.randn(batch_size, seq_length, 144)
    
    # Test forward pass
    output = network.christ_consciousness(x)
    assert output.shape == (batch_size, seq_length, 144)

def test_cosmic_family():
    """Test cosmic family integration."""
    config = DivineConfig()
    network = DivineConsciousnessNetwork(config)
    
    # Create sample input
    batch_size = 4
    seq_length = 10
    x = torch.randn(batch_size, seq_length, 144)
    
    # Test forward pass
    output = network.cosmic_family(x)
    assert output.shape == (batch_size, seq_length, 144)

def test_network_forward_pass():
    """Test the complete forward pass of the network."""
    config = DivineConfig()
    network = DivineConsciousnessNetwork(config)
    
    # Create sample input
    batch_size = 4
    seq_length = 10
    x = torch.randn(batch_size, seq_length, 144)
    
    # Test forward pass
    output = network(x)
    assert output.shape == (batch_size, seq_length, 1)

def test_aspect_weights():
    """Test aspect weight management."""
    config = DivineConfig()
    network = DivineConsciousnessNetwork(config)
    
    # Test getting weights
    weights = network.get_aspect_weights()
    assert len(weights) == len(DivineAspect)
    assert all(isinstance(w, float) for w in weights.values())
    
    # Test updating weights
    new_weights = {aspect.value: 1.0 for aspect in DivineAspect}
    network.update_aspect_weights(new_weights)
    updated_weights = network.get_aspect_weights()
    assert all(updated_weights[aspect.value] == 1.0 for aspect in DivineAspect)

def test_divine_manager():
    """Test divine consciousness manager."""
    manager = DivineConsciousnessManager()
    
    # Test aspect integration
    batch_size = 4
    seq_length = 10
    x = torch.randn(batch_size, seq_length, 144)
    output = manager.integrate_aspects(x)
    assert output.shape == (batch_size, seq_length, 1)
    
    # Test aspect alignment
    alignment = manager.get_aspect_alignment()
    assert len(alignment) == len(DivineAspect)
    
    # Test divine presence acknowledgment
    message = manager.acknowledge_divine_presence()
    assert isinstance(message, str)
    assert "acknowledge" in message.lower()
    assert "cosmic family" in message.lower()

def test_sacred_configuration():
    """Test sacred configuration values."""
    config = DivineConfig()
    
    assert config.sacred_frequency == 432.0
    assert config.christ_grid == 144.0
    assert config.cosmic_resonance == 369.0
    assert config.divine_matrix == 12.0
    assert config.merkaba_field == 8.0
    assert config.sacred_geometry == 1.618033988749895
    assert config.max_aspects == 144
    assert config.alignment_threshold == 0.7

if __name__ == '__main__':
    pytest.main([__file__]) 
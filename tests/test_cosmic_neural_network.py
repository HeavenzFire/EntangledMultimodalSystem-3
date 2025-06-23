import pytest
import torch
import numpy as np
from src.quantum.synthesis.cosmic_neural_network import (
    CosmicNeuralNetwork,
    CosmicNetworkManager,
    NetworkType,
    NetworkConfig
)

def test_network_initialization():
    """Test initialization of cosmic neural networks."""
    config = NetworkConfig()
    network = CosmicNeuralNetwork(config)
    
    assert network.config == config
    assert len(network.layers) == config.num_layers
    assert network.output.out_features == 1

def test_attention_mechanism():
    """Test the cosmic attention mechanism."""
    config = NetworkConfig(hidden_dim=144, attention_heads=8)
    network = CosmicNeuralNetwork(config)
    
    # Create sample input
    batch_size = 4
    seq_length = 10
    x = torch.randn(batch_size, seq_length, config.hidden_dim)
    
    # Test forward pass
    output = network.layers[0].attention(x)
    assert output.shape == (batch_size, seq_length, config.hidden_dim)

def test_quantum_resonance_layer():
    """Test quantum resonance layer processing."""
    config = NetworkConfig()
    network = CosmicNeuralNetwork(config)
    
    # Create sample input
    batch_size = 4
    seq_length = 10
    x = torch.randn(batch_size, seq_length, config.hidden_dim)
    
    # Test forward pass
    output = network.layers[0](x)
    assert output.shape == (batch_size, seq_length, config.hidden_dim)

def test_network_forward_pass():
    """Test the complete forward pass of the network."""
    config = NetworkConfig()
    network = CosmicNeuralNetwork(config)
    
    # Create sample input
    batch_size = 4
    seq_length = 10
    x = torch.randn(batch_size, seq_length, config.hidden_dim)
    
    # Test forward pass
    output = network(x)
    assert output.shape == (batch_size, seq_length, 1)

def test_network_manager():
    """Test the cosmic network manager."""
    manager = CosmicNetworkManager()
    
    # Test network creation
    manager.create_network("test_network", NetworkType.QUANTUM_RESONANCE)
    assert "test_network" in manager.networks
    
    # Test network state retrieval
    state = manager.get_network_state("test_network")
    assert "weights" in state
    assert "config" in state
    assert "metrics" in state

def test_training_step():
    """Test a single training step."""
    config = NetworkConfig()
    network = CosmicNeuralNetwork(config)
    network.optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)
    
    # Create sample batch
    batch_size = 4
    seq_length = 10
    x = torch.randn(batch_size, seq_length, config.hidden_dim)
    y = torch.randn(batch_size, seq_length, 1)
    batch = (x, y)
    
    # Test training step
    loss = network.train_step(batch)
    assert isinstance(loss, float)
    assert loss >= 0

def test_validation():
    """Test network validation."""
    config = NetworkConfig()
    network = CosmicNeuralNetwork(config)
    
    # Create sample validation data
    batch_size = 4
    seq_length = 10
    x = torch.randn(batch_size, seq_length, config.hidden_dim)
    y = torch.randn(batch_size, seq_length, 1)
    
    # Create DataLoader
    dataset = torch.utils.data.TensorDataset(x, y)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=2)
    
    # Test validation
    val_loss = network.validate(val_loader)
    assert isinstance(val_loss, float)
    assert val_loss >= 0

def test_network_predictions():
    """Test network predictions."""
    manager = CosmicNetworkManager()
    manager.create_network("test_network", NetworkType.QUANTUM_RESONANCE)
    
    # Create sample input
    batch_size = 4
    seq_length = 10
    x = torch.randn(batch_size, seq_length, manager.config.hidden_dim)
    
    # Test predictions
    predictions = manager.predict("test_network", x)
    assert predictions.shape == (batch_size, seq_length, 1)

def test_network_metrics():
    """Test network training metrics."""
    manager = CosmicNetworkManager()
    manager.create_network("test_network", NetworkType.QUANTUM_RESONANCE)
    
    # Create sample data
    batch_size = 4
    seq_length = 10
    x = torch.randn(batch_size, seq_length, manager.config.hidden_dim)
    y = torch.randn(batch_size, seq_length, 1)
    
    # Create DataLoaders
    train_dataset = torch.utils.data.TensorDataset(x, y)
    val_dataset = torch.utils.data.TensorDataset(x, y)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2)
    
    # Test training metrics
    metrics = manager.train_network("test_network", train_loader, val_loader)
    assert "train_loss" in metrics
    assert "val_loss" in metrics
    assert len(metrics["train_loss"]) == manager.config.epochs
    assert len(metrics["val_loss"]) == manager.config.epochs

if __name__ == '__main__':
    pytest.main([__file__]) 
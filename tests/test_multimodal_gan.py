import pytest
import numpy as np
import torch
from src.core.multimodal_gan import MultimodalGAN
from src.utils.errors import ModelError

@pytest.fixture
def multimodal_gan():
    return MultimodalGAN(
        latent_dim=100,
        quantum_dim=16,
        holographic_dim=(64, 64),
        neural_dim=32
    )

def test_initialization(multimodal_gan):
    """Test multimodal GAN initialization."""
    assert multimodal_gan.latent_dim == 100
    assert multimodal_gan.quantum_dim == 16
    assert multimodal_gan.holographic_dim == (64, 64)
    assert multimodal_gan.neural_dim == 32
    assert isinstance(multimodal_gan.generator, torch.nn.Module)
    assert isinstance(multimodal_gan.discriminator, torch.nn.Module)

def test_generator_architecture(multimodal_gan):
    """Test generator network architecture."""
    batch_size = 4
    latent = torch.randn(batch_size, multimodal_gan.latent_dim)
    output = multimodal_gan.generator(latent)
    
    assert isinstance(output, dict)
    assert all(key in output for key in ['quantum', 'holographic', 'neural'])
    assert output['quantum'].shape == (batch_size, multimodal_gan.quantum_dim)
    assert output['holographic'].shape == (batch_size, 1, *multimodal_gan.holographic_dim)
    assert output['neural'].shape == (batch_size, multimodal_gan.neural_dim)

def test_discriminator_architecture(multimodal_gan):
    """Test discriminator network architecture."""
    batch_size = 4
    fake_data = {
        'quantum': torch.randn(batch_size, multimodal_gan.quantum_dim),
        'holographic': torch.randn(batch_size, 1, *multimodal_gan.holographic_dim),
        'neural': torch.randn(batch_size, multimodal_gan.neural_dim)
    }
    
    output = multimodal_gan.discriminator(fake_data)
    assert output.shape == (batch_size, 1)
    assert torch.all(output >= 0) and torch.all(output <= 1)

def test_generate_samples(multimodal_gan):
    """Test sample generation."""
    num_samples = 4
    samples = multimodal_gan.generate_samples(num_samples)
    
    assert isinstance(samples, dict)
    assert all(key in samples for key in ['quantum', 'holographic', 'neural'])
    assert samples['quantum'].shape == (num_samples, multimodal_gan.quantum_dim)
    assert samples['holographic'].shape == (num_samples, 1, *multimodal_gan.holographic_dim)
    assert samples['neural'].shape == (num_samples, multimodal_gan.neural_dim)

def test_training_step(multimodal_gan):
    """Test single training step."""
    batch_size = 4
    real_data = {
        'quantum': torch.randn(batch_size, multimodal_gan.quantum_dim),
        'holographic': torch.randn(batch_size, 1, *multimodal_gan.holographic_dim),
        'neural': torch.randn(batch_size, multimodal_gan.neural_dim)
    }
    
    g_loss, d_loss = multimodal_gan.training_step(real_data)
    assert isinstance(g_loss, float)
    assert isinstance(d_loss, float)
    assert not np.isnan(g_loss)
    assert not np.isnan(d_loss)

def test_latent_interpolation(multimodal_gan):
    """Test latent space interpolation."""
    start_point = torch.randn(1, multimodal_gan.latent_dim)
    end_point = torch.randn(1, multimodal_gan.latent_dim)
    num_steps = 5
    
    interpolated = multimodal_gan.interpolate_latent(start_point, end_point, num_steps)
    assert isinstance(interpolated, dict)
    assert all(key in interpolated for key in ['quantum', 'holographic', 'neural'])
    assert interpolated['quantum'].shape[0] == num_steps
    assert interpolated['holographic'].shape[0] == num_steps
    assert interpolated['neural'].shape[0] == num_steps

def test_get_gan_status(multimodal_gan):
    """Test retrieval of GAN status."""
    status = multimodal_gan.get_gan_status()
    assert isinstance(status, dict)
    assert all(key in status for key in [
        'generator_state',
        'discriminator_state',
        'training_metrics',
        'sample_quality'
    ])

def test_error_handling(multimodal_gan):
    """Test error handling in GAN operations."""
    # Test invalid dimensions
    with pytest.raises(ModelError):
        MultimodalGAN(latent_dim=-1)
    
    with pytest.raises(ModelError):
        MultimodalGAN(quantum_dim=0)
    
    # Test invalid input shapes
    with pytest.raises(ModelError):
        multimodal_gan.discriminator({
            'quantum': torch.randn(4, multimodal_gan.quantum_dim + 1)
        })

def test_reset_gan(multimodal_gan):
    """Test reset of GAN models."""
    # Get initial weights
    initial_g_weights = multimodal_gan.generator.state_dict()
    initial_d_weights = multimodal_gan.discriminator.state_dict()
    
    # Train for one step
    fake_data = {
        'quantum': torch.randn(4, multimodal_gan.quantum_dim),
        'holographic': torch.randn(4, 1, *multimodal_gan.holographic_dim),
        'neural': torch.randn(4, multimodal_gan.neural_dim)
    }
    multimodal_gan.training_step(fake_data)
    
    # Reset
    multimodal_gan.reset_gan()
    
    # Check if weights are different
    current_g_weights = multimodal_gan.generator.state_dict()
    current_d_weights = multimodal_gan.discriminator.state_dict()
    
    assert not all(torch.equal(initial_g_weights[key], current_g_weights[key]) 
                  for key in initial_g_weights.keys())
    assert not all(torch.equal(initial_d_weights[key], current_d_weights[key]) 
                  for key in initial_d_weights.keys()) 
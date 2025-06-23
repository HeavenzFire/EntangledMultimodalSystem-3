import pytest
import numpy as np
from src.core.holographic_processor import HolographicProcessor
from src.utils.errors import ModelError

@pytest.fixture
def holographic_processor():
    return HolographicProcessor(
        wavelength=633e-9,  # Red laser wavelength
        pixel_size=10e-6,   # 10 micron pixels
        distance=0.1        # 10cm propagation distance
    )

def test_initialization(holographic_processor):
    """Test holographic processor initialization."""
    assert holographic_processor.wavelength == 633e-9
    assert holographic_processor.pixel_size == 10e-6
    assert holographic_processor.distance == 0.1
    assert holographic_processor.grid_size == (1024, 1024)  # Default size

def test_create_hologram(holographic_processor):
    """Test hologram creation."""
    hologram = holographic_processor.create_hologram()
    assert hologram is not None
    assert isinstance(hologram, np.ndarray)
    assert hologram.shape == holographic_processor.grid_size
    assert np.all(np.abs(hologram) <= 1.0)  # Check normalization

def test_propagate_wave(holographic_processor):
    """Test wave propagation."""
    initial_wave = holographic_processor.create_plane_wave()
    propagated_wave = holographic_processor.propagate_wave(initial_wave)
    assert propagated_wave is not None
    assert isinstance(propagated_wave, np.ndarray)
    assert propagated_wave.shape == initial_wave.shape
    assert np.all(np.isfinite(propagated_wave))  # Check for numerical stability

def test_reconstruct_hologram(holographic_processor):
    """Test hologram reconstruction."""
    hologram = holographic_processor.create_hologram()
    reconstruction = holographic_processor.reconstruct_hologram(hologram)
    assert reconstruction is not None
    assert isinstance(reconstruction, np.ndarray)
    assert reconstruction.shape == hologram.shape
    assert np.all(np.isfinite(reconstruction))

def test_create_point_source(holographic_processor):
    """Test point source creation."""
    point_source = holographic_processor.create_point_source(x=0, y=0)
    assert point_source is not None
    assert isinstance(point_source, np.ndarray)
    assert point_source.shape == holographic_processor.grid_size
    assert np.all(np.isfinite(point_source))

def test_create_plane_wave(holographic_processor):
    """Test plane wave creation."""
    plane_wave = holographic_processor.create_plane_wave(angle_x=0.1, angle_y=0.1)
    assert plane_wave is not None
    assert isinstance(plane_wave, np.ndarray)
    assert plane_wave.shape == holographic_processor.grid_size
    assert np.all(np.abs(plane_wave) - 1.0 < 1e-10)  # Check unit amplitude

def test_interference_pattern(holographic_processor):
    """Test interference pattern generation."""
    reference = holographic_processor.create_plane_wave()
    object_wave = holographic_processor.create_point_source(x=0, y=0)
    interference = holographic_processor.create_interference_pattern(reference, object_wave)
    assert interference is not None
    assert isinstance(interference, np.ndarray)
    assert interference.shape == holographic_processor.grid_size
    assert np.all(np.isreal(interference))  # Interference pattern should be real

def test_get_hologram_info(holographic_processor):
    """Test retrieval of hologram information."""
    hologram = holographic_processor.create_hologram()
    info = holographic_processor.get_hologram_info(hologram)
    assert isinstance(info, dict)
    assert "shape" in info
    assert "max_amplitude" in info
    assert "phase_range" in info

def test_error_handling(holographic_processor):
    """Test error handling in holographic operations."""
    # Test invalid wavelength
    with pytest.raises(ModelError):
        HolographicProcessor(wavelength=-1)
    
    # Test invalid pixel size
    with pytest.raises(ModelError):
        HolographicProcessor(wavelength=633e-9, pixel_size=0)
    
    # Test invalid distance
    with pytest.raises(ModelError):
        HolographicProcessor(wavelength=633e-9, pixel_size=10e-6, distance=-1)

def test_reset_processor(holographic_processor):
    """Test reset of holographic processor."""
    # Create some holograms
    hologram1 = holographic_processor.create_hologram()
    
    # Reset
    holographic_processor.reset_processor()
    
    # Create new hologram
    hologram2 = holographic_processor.create_hologram()
    
    # Verify different random states
    assert not np.array_equal(hologram1, hologram2) 
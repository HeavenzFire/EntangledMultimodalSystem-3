import pytest
import numpy as np
from src.quantum.geometry.entanglement_torus import (
    QuantumEntanglementTorus,
    TorusConfig,
    TorusState
)

# --- Initialization Tests ---

def test_torus_initialization_valid_dimensions():
    """Tests successful initialization with valid dimensions."""
    try:
        config = TorusConfig(dimensions=12)
        torus = QuantumEntanglementTorus(config)
        assert torus.config.dimensions == 12
        assert torus.state == TorusState.HARMONIC
        assert len(torus.torus_field) == 144  # 12 * 12 for phi resonance
    except Exception as e:
        pytest.fail(f"Initialization with valid dimensions failed: {e}")

@pytest.mark.parametrize("invalid_dims", [
    0,      # Zero dimension
    -1,     # Negative dimension
    2.5,    # Float dimension
    "12",   # Wrong type (string)
    None,   # Wrong type (None)
])
def test_torus_initialization_invalid_dimensions(invalid_dims):
    """Tests that initialization fails with various invalid dimension specifications."""
    with pytest.raises((ValueError, TypeError)):
        config = TorusConfig(dimensions=invalid_dims)
        QuantumEntanglementTorus(config)

def test_torus_initialization_default_config():
    """Tests initialization with default configuration."""
    torus = QuantumEntanglementTorus()
    assert torus.config.dimensions == 12
    assert torus.config.phi_resonance == pytest.approx(1.618033988749895)
    assert torus.config.harmonic_threshold == 0.9
    assert torus.config.ascension_threshold == 0.99
    assert torus.config.max_iterations == 144

# --- Field Harmonization Tests ---

def test_harmonize_field_valid_input():
    """Tests field harmonization with valid input."""
    torus = QuantumEntanglementTorus()
    consciousness = np.random.rand(12) + 1j * np.random.rand(12)
    harmonized = torus.harmonize_field(consciousness)
    assert isinstance(harmonized, np.ndarray)
    assert harmonized.shape[0] > consciousness.shape[0]  # Should be scaled by phi

@pytest.mark.parametrize("invalid_input", [
    np.random.rand(10),  # Wrong dimension
    np.random.rand(12, 12),  # Wrong shape
    "not an array",  # Wrong type
    None,  # None input
])
def test_harmonize_field_invalid_input(invalid_input):
    """Tests field harmonization with invalid inputs."""
    torus = QuantumEntanglementTorus()
    with pytest.raises(ValueError):
        torus.harmonize_field(invalid_input)

# --- State Transition Tests ---

def test_state_transitions():
    """Tests state transitions based on harmonic scores."""
    torus = QuantumEntanglementTorus()
    
    # Test initial state
    assert torus.state == TorusState.HARMONIC
    
    # Create patterns with known characteristics
    dissonant_pattern = np.zeros(12, dtype=np.complex128)  # Should cause dissonance
    transitional_pattern = np.ones(12, dtype=np.complex128)  # Should be transitional
    resonant_pattern = np.array([torus.config.phi_resonance ** i for i in range(12)], 
                               dtype=np.complex128)  # Should resonate
    
    # Test transition to dissonant
    torus.harmonize_field(dissonant_pattern)
    assert torus.state == TorusState.DISSONANT
    
    # Test transition to transitional
    torus.harmonize_field(transitional_pattern)
    assert torus.state == TorusState.TRANSITIONAL
    
    # Test transition to resonant
    for _ in range(5):  # Multiple iterations to achieve stability
        torus.harmonize_field(resonant_pattern)
    assert torus.state == TorusState.RESONANT

def test_stability_thresholds():
    """Tests state stability thresholds and hysteresis."""
    torus = QuantumEntanglementTorus()
    
    # Create oscillating pattern to test stability
    patterns = []
    for i in range(10):
        if i % 2 == 0:
            pattern = np.ones(12, dtype=np.complex128)
        else:
            pattern = np.zeros(12, dtype=np.complex128)
        patterns.append(pattern)
    
    # Apply oscillating patterns
    for pattern in patterns:
        torus.harmonize_field(pattern)
    
    # Should be in dissonant state due to high instability
    assert torus.state == TorusState.DISSONANT
    
    # Apply stable pattern repeatedly
    stable_pattern = np.array([torus.config.phi_resonance ** i for i in range(12)], 
                            dtype=np.complex128)
    for _ in range(5):
        torus.harmonize_field(stable_pattern)
    
    # Should transition to resonant state due to stability
    assert torus.state == TorusState.RESONANT

def test_moving_average():
    """Tests the moving average calculation in state transitions."""
    torus = QuantumEntanglementTorus()
    
    # Create pattern that gradually improves harmony
    patterns = []
    for i in range(10):
        pattern = np.array([torus.config.phi_resonance ** (i/10 * j) for j in range(12)], 
                         dtype=np.complex128)
        patterns.append(pattern)
    
    states = []
    for pattern in patterns:
        torus.harmonize_field(pattern)
        states.append(torus.state)
    
    # Verify state progression
    assert TorusState.DISSONANT in states
    assert TorusState.TRANSITIONAL in states
    assert TorusState.RESONANT in states
    
    # Verify final state is stable
    final_states = states[-3:]
    assert all(state == final_states[0] for state in final_states)

# --- Additional Edge Case Tests ---

def test_empty_history():
    """Tests behavior with empty harmonic history."""
    torus = QuantumEntanglementTorus()
    assert len(torus.harmonic_history) == 0
    assert torus.state == TorusState.HARMONIC

def test_history_length_limit():
    """Tests that history length is properly limited."""
    torus = QuantumEntanglementTorus()
    pattern = np.ones(12, dtype=np.complex128)
    
    # Generate more iterations than the history length limit
    for _ in range(200):
        torus.harmonize_field(pattern)
    
    # Verify history length is maintained
    assert len(torus.harmonic_history) <= 144  # max_iterations from config

# --- Harmonic History Tests ---

def test_harmonic_history():
    """Tests harmonic history tracking."""
    torus = QuantumEntanglementTorus()
    consciousness = np.random.rand(12) + 1j * np.random.rand(12)
    
    initial_length = len(torus.harmonic_history)
    torus.harmonize_field(consciousness)
    assert len(torus.harmonic_history) == initial_length + 1
    assert 0 <= torus.harmonic_history[-1] <= 1

# --- Reset Tests ---

def test_reset_torus():
    """Tests torus reset functionality."""
    torus = QuantumEntanglementTorus()
    consciousness = np.random.rand(12) + 1j * np.random.rand(12)
    
    # Perform some operations
    torus.harmonize_field(consciousness)
    initial_state = torus.state
    initial_history_length = len(torus.harmonic_history)
    
    # Reset the torus
    torus.reset_torus()
    
    # Verify reset
    assert torus.state == TorusState.HARMONIC
    assert len(torus.harmonic_history) == 0
    assert len(torus.torus_field) == 144 

# --- Phi Resonance Tests ---

def test_phi_resonance_calculation():
    """Tests phi resonance calculation with known patterns."""
    torus = QuantumEntanglementTorus()
    
    # Create a pattern that should resonate with phi
    phi = torus.config.phi_resonance
    pattern = np.array([phi ** i for i in range(torus.config.dimensions)], dtype=np.complex128)
    
    harmonized = torus.harmonize_field(pattern)
    score = torus.harmonic_history[-1]
    
    # Pattern aligned with phi should have high harmonic score
    assert score > 0.8
    assert torus.state == TorusState.RESONANT or torus.state == TorusState.ASCENDED

def test_phi_scaling_effects():
    """Tests the effects of phi scaling on tensor dimensions."""
    torus = QuantumEntanglementTorus()
    pattern = np.random.rand(12) + 1j * np.random.rand(12)
    
    harmonized = torus.harmonize_field(pattern)
    
    # Verify phi scaling
    original_size = pattern.shape[0]
    scaled_size = harmonized.shape[0]
    expected_size = int(original_size * torus.config.phi_resonance)
    
    assert scaled_size == expected_size
    assert scaled_size > original_size

def test_phi_resonance_edge_cases():
    """Tests phi resonance with edge cases and boundary conditions."""
    torus = QuantumEntanglementTorus()
    
    # Test with zero pattern
    zero_pattern = np.zeros(torus.config.dimensions, dtype=np.complex128)
    harmonized_zero = torus.harmonize_field(zero_pattern)
    score_zero = torus.harmonic_history[-1]
    assert score_zero < 0.1  # Zero pattern should have low harmonic score
    
    # Test with constant pattern
    const_pattern = np.ones(torus.config.dimensions, dtype=np.complex128)
    harmonized_const = torus.harmonize_field(const_pattern)
    score_const = torus.harmonic_history[-1]
    assert 0.3 <= score_const <= 0.7  # Constant pattern should have moderate score
    
    # Test with alternating pattern
    alt_pattern = np.array([1 if i % 2 == 0 else -1 for i in range(torus.config.dimensions)], 
                          dtype=np.complex128)
    harmonized_alt = torus.harmonize_field(alt_pattern)
    score_alt = torus.harmonic_history[-1]
    assert 0.4 <= score_alt <= 0.8  # Alternating pattern should have moderate to high score

def test_phi_resonance_custom_config():
    """Tests phi resonance with custom configuration."""
    # Create custom config with different phi resonance
    custom_phi = 1.5  # Different from golden ratio
    config = TorusConfig(
        dimensions=12,
        phi_resonance=custom_phi,
        harmonic_threshold=0.8,
        ascension_threshold=0.95
    )
    torus = QuantumEntanglementTorus(config)
    
    # Create pattern aligned with custom phi
    pattern = np.array([custom_phi ** i for i in range(config.dimensions)], 
                      dtype=np.complex128)
    
    harmonized = torus.harmonize_field(pattern)
    score = torus.harmonic_history[-1]
    
    # Pattern should resonate with custom phi
    assert score > 0.7
    assert torus.state == TorusState.RESONANT or torus.state == TorusState.ASCENDED

def test_phi_resonance_phase_sensitivity():
    """Tests sensitivity to phase in phi resonance patterns."""
    torus = QuantumEntanglementTorus()
    phi = torus.config.phi_resonance
    
    # Create base pattern
    base_pattern = np.array([phi ** i for i in range(torus.config.dimensions)], 
                          dtype=np.complex128)
    
    # Test different phase shifts
    for phase in [0, np.pi/4, np.pi/2, np.pi]:
        phase_pattern = base_pattern * np.exp(1j * phase)
        harmonized = torus.harmonize_field(phase_pattern)
        score = torus.harmonic_history[-1]
        
        # Phase should not significantly affect resonance
        assert abs(score - torus.harmonic_history[-2]) < 0.1 if len(torus.harmonic_history) > 1 else True 

# --- Performance and Stress Tests ---

def test_large_scale_harmonization():
    """Tests system performance with large-scale harmonization operations."""
    torus = QuantumEntanglementTorus()
    num_iterations = 1000
    
    start_time = np.datetime64('now')
    pattern = np.array([complex(np.cos(i), np.sin(i)) for i in range(12)])
    
    for _ in range(num_iterations):
        harmonized = torus.harmonize_field(pattern)
        assert harmonized is not None
        assert isinstance(harmonized, np.ndarray)
    
    end_time = np.datetime64('now')
    processing_time = (end_time - start_time) / np.timedelta64(1, 's')
    
    # Ensure reasonable performance (adjust threshold as needed)
    assert processing_time / num_iterations < 0.01  # 10ms per iteration max

def test_memory_stability():
    """Tests memory stability during repeated operations."""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    
    torus = QuantumEntanglementTorus()
    pattern = np.array([complex(np.cos(i), np.sin(i)) for i in range(12)])
    
    # Perform multiple operations
    for _ in range(1000):
        harmonized = torus.harmonize_field(pattern)
        del harmonized  # Ensure cleanup
    
    final_memory = process.memory_info().rss
    memory_growth = (final_memory - initial_memory) / initial_memory
    
    # Ensure memory growth is reasonable (less than 10%)
    assert memory_growth < 0.1

@pytest.mark.parametrize("field_size", [12, 24, 48, 96])
def test_scaling_performance(field_size):
    """Tests performance scaling with different field sizes."""
    config = TorusConfig(dimensions=field_size)
    torus = QuantumEntanglementTorus(config)
    
    pattern = np.array([complex(np.cos(i), np.sin(i)) for i in range(field_size)])
    
    start_time = np.datetime64('now')
    harmonized = torus.harmonize_field(pattern)
    end_time = np.datetime64('now')
    
    processing_time = (end_time - start_time) / np.timedelta64(1, 's')
    
    # Processing time should scale roughly linearly with field size
    # Allow for some overhead in smaller sizes
    expected_max_time = (field_size / 12) * 0.01  # Base processing time for size 12
    assert processing_time < expected_max_time

def test_concurrent_stability():
    """Tests stability under concurrent operations."""
    import threading
    import queue
    
    results_queue = queue.Queue()
    num_threads = 4
    iterations_per_thread = 100
    
    def worker():
        torus = QuantumEntanglementTorus()
        pattern = np.array([complex(np.cos(i), np.sin(i)) for i in range(12)])
        try:
            for _ in range(iterations_per_thread):
                harmonized = torus.harmonize_field(pattern)
                assert harmonized is not None
            results_queue.put(True)
        except Exception as e:
            results_queue.put(e)
    
    threads = []
    for _ in range(num_threads):
        thread = threading.Thread(target=worker)
        thread.start()
        threads.append(thread)
    
    for thread in threads:
        thread.join()
    
    # Check results
    while not results_queue.empty():
        result = results_queue.get()
        assert result is True

if __name__ == '__main__':
    pytest.main([__file__]) 
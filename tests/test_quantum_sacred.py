import pytest
import numpy as np
from src.quantum.synthesis.quantum_sacred import (
    QuantumSacredSynthesis,
    SacredConfig,
    QuantumState,
    VortexHistoryBuffer,
    VortexPrimes
)

def test_vortex_prime_sequence():
    """Test vortex prime sequence alignment"""
    primes = VortexPrimes([3, 6, 9])
    assert primes.is_vortex_prime(3)
    assert primes.is_vortex_prime(6)
    assert primes.is_vortex_prime(9)
    assert not primes.is_vortex_prime(4)
    assert not primes.is_vortex_prime(7)

def test_history_buffer_entropy():
    """Test history buffer entropy management"""
    config = SacredConfig()
    buffer = VortexHistoryBuffer(config)
    
    # Add states with varying entropy
    low_entropy_state = {"a": 0.9, "b": 0.1}
    high_entropy_state = {"a": 0.5, "b": 0.5}
    
    buffer.add_state(low_entropy_state)
    buffer.add_state(high_entropy_state)
    buffer.purge_entropy()
    
    # High entropy state should be purged
    assert len(buffer.buffer) == 1
    assert buffer.buffer[0] == low_entropy_state

def test_transition_matrix_update():
    """Test transition matrix update with sacred geometry"""
    synthesis = QuantumSacredSynthesis()
    initial_matrix = synthesis.transition_matrix.copy()
    
    # Update with high coherence and low entropy
    synthesis.update_transition_matrix(coherence_level=0.9, field_entropy=0.1)
    
    # Verify probabilities are within sacred geometry bounds
    assert np.all(synthesis.transition_matrix >= 0.7)
    assert np.all(synthesis.transition_matrix <= 0.8)
    assert not np.array_equal(initial_matrix, synthesis.transition_matrix)

def test_dissonance_resolution():
    """Test quantum-sacred escape mechanism"""
    synthesis = QuantumSacredSynthesis()
    synthesis.current_state = QuantumState.DISSONANT
    
    # Apply resolution protocol
    synthesis.resolve_dissonance()
    
    # Should transition to resonant state
    assert synthesis.current_state == QuantumState.RESONANT
    assert synthesis.dissonance_cycles == 0

def test_field_optimization():
    """Test quantum-parallel field optimization"""
    synthesis = QuantumSacredSynthesis()
    
    # Create test field
    field = np.random.rand(144) + 1j * np.random.rand(144)
    
    # Optimize field
    optimized = synthesis.optimize_field_operations(field)
    
    # Verify optimization properties
    assert optimized.shape == field.shape
    assert np.all(np.abs(optimized) <= 1.0)  # Normalized
    assert not np.array_equal(field, optimized)  # Field should be modified

def test_photon_stargate_activation():
    """Test emergency photon stargate protocol"""
    synthesis = QuantumSacredSynthesis()
    synthesis.current_state = QuantumState.DISSONANT
    synthesis.dissonance_cycles = 145  # Exceed max history
    
    # Activate stargate
    synthesis.resolve_dissonance()
    
    # Should transition to merkaba state
    assert synthesis.current_state == QuantumState.MERKABA
    assert synthesis.dissonance_cycles == 0
    assert synthesis.merkaba_rotation == 0.0

def test_christos_harmonic():
    """Test Christos grid harmonic application"""
    synthesis = QuantumSacredSynthesis()
    
    # Apply harmonic
    resonance = synthesis._apply_christos_harmonic()
    
    # Verify resonance properties
    assert 0 <= resonance <= 1.0
    assert isinstance(resonance, float)

def test_toroidal_recombination():
    """Test toroidal recombination of field partitions"""
    synthesis = QuantumSacredSynthesis()
    
    # Create test partitions
    partitions = [np.random.rand(12) + 1j * np.random.rand(12) for _ in range(12)]
    
    # Recombine
    recombined = synthesis._recombine_toroid(partitions)
    
    # Verify recombination properties
    assert recombined.shape == (12,)
    assert np.all(np.abs(recombined) <= 1.0)  # Normalized

def test_multi_resonance_gate():
    """Test multi-resonance quantum gate application"""
    synthesis = QuantumSacredSynthesis()
    
    # Create test partition
    partition = np.random.rand(12) + 1j * np.random.rand(12)
    
    # Apply gate
    processed = synthesis._apply_multi_resonance(partition)
    
    # Verify processing properties
    assert processed.shape == partition.shape
    assert not np.array_equal(partition, processed)  # Should be modified

def test_sacred_config_defaults():
    """Test sacred configuration defaults"""
    config = SacredConfig()
    
    assert config.phi_resonance == pytest.approx(1.618033988749895)
    assert config.vortex_sequence == (3, 6, 9)
    assert config.max_history == 144
    assert config.entropy_threshold == 3.69
    assert config.torsion_field == 369.0
    assert config.christos_frequency == 432.0

if __name__ == '__main__':
    pytest.main([__file__]) 
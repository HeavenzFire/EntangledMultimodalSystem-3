import pytest
import numpy as np
from src.quantum.resonance.quantum_resonance import (
    QuantumResonance,
    FrequencyType,
    ResonancePattern
)

def test_pattern_generation():
    """Test generation of all resonance patterns."""
    resonance = QuantumResonance()
    
    # Test Sacred pattern
    sacred = resonance.generate_pattern(FrequencyType.SACRED)
    assert sacred.pattern_type == FrequencyType.SACRED
    assert len(sacred.frequencies) == 5
    assert len(sacred.amplitudes) == 5
    assert len(sacred.phases) == 5
    assert sacred.energy_level == 1.0
    
    # Test Harmonic pattern
    harmonic = resonance.generate_pattern(FrequencyType.HARMONIC)
    assert harmonic.pattern_type == FrequencyType.HARMONIC
    assert len(harmonic.frequencies) == 8
    assert len(harmonic.amplitudes) == 8
    assert len(harmonic.phases) == 8
    assert harmonic.energy_level == 0.8
    
    # Test Quantum pattern
    quantum = resonance.generate_pattern(FrequencyType.QUANTUM)
    assert quantum.pattern_type == FrequencyType.QUANTUM
    assert len(quantum.frequencies) == 7
    assert len(quantum.amplitudes) == 7
    assert len(quantum.phases) == 7
    assert quantum.energy_level == 0.9
    
    # Test Divine pattern
    divine = resonance.generate_pattern(FrequencyType.DIVINE)
    assert divine.pattern_type == FrequencyType.DIVINE
    assert len(divine.frequencies) == 9
    assert len(divine.amplitudes) == 9
    assert len(divine.phases) == 9
    assert divine.energy_level == 1.0

def test_pattern_transformation():
    """Test transformations on resonance patterns."""
    resonance = QuantumResonance()
    pattern = resonance.generate_pattern(FrequencyType.SACRED)
    
    # Test frequency scaling
    transformed = resonance.transform_pattern(pattern, frequency_scale=2.0)
    assert transformed.pattern_type == pattern.pattern_type
    assert len(transformed.frequencies) == len(pattern.frequencies)
    assert all(tf == 2.0 * pf for tf, pf in zip(transformed.frequencies, pattern.frequencies))
    
    # Test amplitude scaling
    transformed = resonance.transform_pattern(pattern, amplitude_scale=0.5)
    assert all(ta == 0.5 * pa for ta, pa in zip(transformed.amplitudes, pattern.amplitudes))
    
    # Test phase shifting
    phase_shift = np.pi/2
    transformed = resonance.transform_pattern(pattern, phase_shift=phase_shift)
    assert all((tp - pp) % (2*np.pi) == phase_shift % (2*np.pi)
              for tp, pp in zip(transformed.phases, pattern.phases))

def test_resonance_calculation():
    """Test resonance value calculations."""
    resonance = QuantumResonance()
    resonance.activate_pattern(FrequencyType.SACRED)
    
    # Test at different time points
    times = [0.0, 0.1, 0.5, 1.0]
    for t in times:
        value = resonance.calculate_resonance(t)
        assert isinstance(value, float)
        assert -1.0 <= value <= 1.0
    
    # Test with different resonance factors
    resonance.update_resonance_factor(0.5)
    value = resonance.calculate_resonance(0.0)
    assert abs(value) <= 0.5

def test_energy_alignment():
    """Test energy alignment checks."""
    resonance = QuantumResonance()
    
    # Test with no active pattern
    assert not resonance.check_energy_alignment()
    
    # Test with different patterns
    resonance.activate_pattern(FrequencyType.SACRED)
    assert resonance.check_energy_alignment()  # energy_level = 1.0 > threshold = 0.5
    
    resonance.activate_pattern(FrequencyType.HARMONIC)
    assert resonance.check_energy_alignment()  # energy_level = 0.8 > threshold = 0.5
    
    # Test with updated threshold
    resonance.energy_threshold = 0.9
    assert not resonance.check_energy_alignment()  # energy_level = 0.8 < threshold = 0.9

def test_invalid_pattern():
    """Test handling of invalid pattern types."""
    resonance = QuantumResonance()
    with pytest.raises(ValueError):
        resonance.generate_pattern("invalid_pattern")  # type: ignore 
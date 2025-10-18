import pytest
import numpy as np
from src.quantum.synthesis.quantum_sacred import (
    QuantumSacredSynthesis,
    SynthesisMetrics,
    SacredConfig,
    VortexHistoryBuffer
)
from src.quantum.geometry.sacred_geometry import PatternType
from src.quantum.resonance.quantum_resonance import FrequencyType

def test_synthesis_creation():
    """Test creation of new syntheses."""
    synthesis = QuantumSacredSynthesis()
    
    # Test creating a synthesis
    synthesis.create_synthesis(
        "test_synthesis",
        PatternType.FLOWER_OF_LIFE,
        FrequencyType.SACRED
    )
    
    assert "test_synthesis" in synthesis.metrics
    assert synthesis.active_synthesis == "test_synthesis"
    assert synthesis.geometry.active_pattern is not None
    assert synthesis.resonance.active_pattern is not None
    
    # Test metrics calculation
    metrics = synthesis.metrics["test_synthesis"]
    assert isinstance(metrics, SynthesisMetrics)
    assert 0.0 <= metrics.geometric_alignment <= 1.0
    assert 0.0 <= metrics.resonance_strength <= 1.0
    assert 0.0 <= metrics.energy_level
    assert -1.0 <= metrics.phase_alignment <= 1.0

def test_synthesis_update():
    """Test updating existing syntheses."""
    synthesis = QuantumSacredSynthesis()
    synthesis.create_synthesis(
        "test_synthesis",
        PatternType.FLOWER_OF_LIFE,
        FrequencyType.SACRED
    )
    
    # Test geometric transformation
    rotation = np.array([
        [np.cos(np.pi/4), -np.sin(np.pi/4), 0],
        [np.sin(np.pi/4), np.cos(np.pi/4), 0],
        [0, 0, 1]
    ])
    scale = 2.0
    translation = np.array([1.0, 1.0, 1.0])
    
    synthesis.update_synthesis(
        "test_synthesis",
        geometric_transform=(rotation, scale, translation)
    )
    
    # Test resonance transformation
    synthesis.update_synthesis(
        "test_synthesis",
        resonance_transform=(2.0, 0.5, np.pi/2)
    )
    
    # Verify metrics were updated
    assert "test_synthesis" in synthesis.metrics
    metrics = synthesis.metrics["test_synthesis"]
    assert isinstance(metrics, SynthesisMetrics)

def test_alignment_check():
    """Test alignment threshold checks."""
    synthesis = QuantumSacredSynthesis()
    synthesis.create_synthesis(
        "test_synthesis",
        PatternType.FLOWER_OF_LIFE,
        FrequencyType.SACRED
    )
    
    # Test with default threshold
    result = synthesis.check_alignment("test_synthesis")
    assert isinstance(result, bool)
    
    # Test with non-existent synthesis
    assert not synthesis.check_alignment("non_existent")
    
    # Test with adjusted threshold
    synthesis.alignment_threshold = 1.1  # Impossible to achieve
    assert not synthesis.check_alignment("test_synthesis")

def test_synthesis_state():
    """Test retrieval of synthesis state."""
    synthesis = QuantumSacredSynthesis()
    synthesis.create_synthesis(
        "test_synthesis",
        PatternType.FLOWER_OF_LIFE,
        FrequencyType.SACRED
    )
    
    # Test state retrieval
    state = synthesis.get_synthesis_state("test_synthesis")
    assert len(state) == 3
    assert state[2] == synthesis.metrics["test_synthesis"]
    
    # Test with non-existent synthesis
    with pytest.raises(ValueError):
        synthesis.get_synthesis_state("non_existent")

def test_phase_update():
    """Test phase updates for both patterns."""
    synthesis = QuantumSacredSynthesis()
    synthesis.create_synthesis(
        "test_synthesis",
        PatternType.FLOWER_OF_LIFE,
        FrequencyType.SACRED
    )
    
    initial_phase = synthesis.geometry.phase
    initial_resonance_phases = synthesis.resonance.active_pattern.phases.copy()
    
    # Update phase
    delta_time = 0.1
    synthesis.update_phase(delta_time)
    
    assert synthesis.geometry.phase != initial_phase
    assert not np.array_equal(
        synthesis.resonance.active_pattern.phases,
        initial_resonance_phases
    )

def test_energy_field():
    """Test energy field calculations."""
    synthesis = QuantumSacredSynthesis()
    synthesis.create_synthesis(
        "test_synthesis",
        PatternType.FLOWER_OF_LIFE,
        FrequencyType.SACRED
    )
    
    # Test with sample points
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    
    energy_field = synthesis.calculate_energy_field(points)
    assert len(energy_field) == len(points)
    assert np.all(energy_field >= 0.0)  # Energy should be non-negative
    
    # Test with no active patterns
    synthesis.geometry.active_pattern = None
    synthesis.resonance.active_pattern = None
    energy_field = synthesis.calculate_energy_field(points)
    assert np.all(energy_field == 0.0)

def test_history_buffer():
    """Test history buffer functionality."""
    config = SacredConfig()
    buffer = VortexHistoryBuffer(config)
    
    # Test adding states
    state1 = {"a": 0.8, "b": 0.2}
    state2 = {"a": 0.6, "b": 0.4}
    
    buffer.add_state(state1)
    buffer.add_state(state2)
    
    assert len(buffer.buffer) == 2
    
    # Test entropy calculation
    entropy = buffer._calculate_entropy(state1)
    assert 0.0 <= entropy <= 1.0
    
    # Test purging high entropy states
    buffer.purge_entropy()
    assert len(buffer.buffer) > 0

def test_sacred_config():
    """Test sacred configuration values."""
    config = SacredConfig()
    
    assert config.phi_resonance == 1.618033988749895
    assert config.torsion_field == np.pi / 12
    assert config.christos_frequency == 432.0
    assert config.max_history == 144
    assert config.entropy_threshold == 0.5

def test_transition_matrix():
    """Test transition matrix operations."""
    synthesis = QuantumSacredSynthesis()
    
    # Test initial state
    assert synthesis.transition_matrix.shape == (12, 12)
    assert np.allclose(synthesis.transition_matrix, np.eye(12))
    
    # Test update
    synthesis.update_transition_matrix(0.9, 0.1)
    assert synthesis.transition_matrix.shape == (12, 12)
    assert np.allclose(np.trace(synthesis.transition_matrix), 1.0)

def test_christos_harmonic():
    """Test Christos harmonic pattern generation."""
    synthesis = QuantumSacredSynthesis()
    
    pattern = synthesis._apply_christos_harmonic()
    assert len(pattern) == 12
    assert np.allclose(np.abs(pattern), 1/np.sqrt(12))

def test_resolution_field():
    """Test resolution field generation."""
    synthesis = QuantumSacredSynthesis()
    
    field = synthesis._generate_resolution_field()
    assert len(field) == 144
    assert np.allclose(np.abs(field), 1.0)

def test_multi_resonance():
    """Test multi-resonance quantum gate."""
    synthesis = QuantumSacredSynthesis()
    
    partition = np.ones(12, dtype=np.complex128)
    processed = synthesis._apply_multi_resonance(partition)
    assert len(processed) == 12
    assert np.allclose(np.abs(processed), 1.0)

if __name__ == '__main__':
    pytest.main([__file__]) 
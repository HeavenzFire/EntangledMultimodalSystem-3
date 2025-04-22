import pytest
import numpy as np
import tensorflow as tf
from src.core.temporal_quantum_state_projector import TemporalQuantumStateProjector

@pytest.fixture
def config():
    return {
        'quantum_dimensions': 16,
        'temporal_depth': 3,
        'ethical_threshold': 0.95,
        'spiritual_strength': 0.9,
        'temporal_resolution': 0.01
    }

@pytest.fixture
def projector(config):
    return TemporalQuantumStateProjector(config)

@pytest.fixture
def quantum_state():
    state = np.random.rand(1, 16) + 1j * np.random.rand(1, 16)
    return state / np.linalg.norm(state)

@pytest.fixture
def ethical_patterns():
    return {
        'compassion': np.ones(16) / np.sqrt(16),
        'justice': np.sin(np.linspace(0, 2*np.pi, 16)),
        'wisdom': np.cos(np.linspace(0, 2*np.pi, 16))
    }

@pytest.fixture
def edge_patterns():
    return {
        'max_entropy': np.random.rand(16),
        'min_entropy': np.zeros(16),
        'periodic': np.sin(2 * np.pi * np.arange(16) / 4),
        'chaotic': np.random.randn(16)
    }

class TestTemporalQuantumStateProjector:
    def test_initialization(self, projector, config):
        """Test proper initialization of the projector."""
        assert projector.quantum_dimensions == config['quantum_dimensions']
        assert projector.temporal_depth == config['temporal_depth']
        assert projector.ethical_threshold == config['ethical_threshold']
        assert projector.spiritual_strength == config['spiritual_strength']
        assert projector.temporal_resolution == config['temporal_resolution']
        
        # Check state initialization
        assert projector.state['current_state'] is None
        assert len(projector.state['temporal_states']) == 0
        assert projector.state['metrics'] is None
        
        # Check model architecture
        assert isinstance(projector.quantum_model, tf.keras.Model)
        assert len(projector.quantum_model.layers) == config['temporal_depth'] * 3 + 2
    
    def test_project(self, projector, quantum_state):
        """Test quantum state projection."""
        # Project state
        projected_state, metrics = projector.project(quantum_state)
        
        # Check output types
        assert isinstance(projected_state, np.ndarray)
        assert isinstance(metrics, dict)
        
        # Check state normalization
        assert np.isclose(np.linalg.norm(projected_state), 1.0, atol=1e-6)
        
        # Check metrics
        assert 'temporal_coherence' in metrics
        assert 'ethical_coherence' in metrics
        assert 'spiritual_alignment' in metrics
        assert 'quantum_purity' in metrics
        
        # Check metric values
        assert 0 <= metrics['temporal_coherence'] <= 1
        assert 0 <= metrics['ethical_coherence'] <= 1
        assert 0 <= metrics['spiritual_alignment'] <= 1
        assert 0 <= metrics['quantum_purity'] <= 1
    
    def test_state_evolution(self, projector, quantum_state):
        """Test temporal evolution of quantum states."""
        # Project multiple states
        states = []
        for _ in range(5):
            state, _ = projector.project(quantum_state)
            states.append(state)
        
        # Check temporal evolution
        for i in range(1, len(states)):
            # States should evolve but maintain coherence
            assert not np.array_equal(states[i], states[i-1])
            coherence = np.abs(np.dot(
                states[i].flatten().conj(),
                states[i-1].flatten()
            ))
            assert coherence > 0.5  # Maintain significant coherence
    
    def test_ethical_coherence(self, projector, quantum_state):
        """Test maintenance of ethical coherence."""
        # Project state
        _, metrics = projector.project(quantum_state)
        
        # Check ethical coherence
        assert metrics['ethical_coherence'] >= projector.ethical_threshold
    
    def test_spiritual_alignment(self, projector, quantum_state):
        """Test spiritual alignment maintenance."""
        # Project multiple states
        for _ in range(3):
            _, metrics = projector.project(quantum_state)
        
        # Check spiritual alignment
        assert metrics['spiritual_alignment'] >= projector.spiritual_strength
    
    def test_invalid_input(self, projector):
        """Test handling of invalid inputs."""
        # Wrong dimensions
        invalid_state = np.random.rand(1, 8)  # Half the required dimensions
        with pytest.raises(ValueError):
            projector.project(invalid_state)
        
        # Unnormalized state
        unnormalized_state = np.random.rand(1, 16)
        with pytest.raises(ValueError):
            projector.project(unnormalized_state)
    
    def test_reset(self, projector, quantum_state):
        """Test state reset functionality."""
        # Project some states
        projector.project(quantum_state)
        projector.project(quantum_state)
        
        # Reset projector
        projector.reset()
        
        # Check state after reset
        assert projector.state['current_state'] is None
        assert len(projector.state['temporal_states']) == 0
        assert projector.state['metrics'] is None
    
    def test_get_state(self, projector, quantum_state):
        """Test state retrieval."""
        # Project state
        projected_state, _ = projector.project(quantum_state)
        
        # Get state
        state = projector.get_state()
        
        # Check state contents
        assert np.array_equal(state['current_state'], projected_state)
        assert len(state['temporal_states']) == 1
        assert state['metrics'] is not None
    
    def test_get_metrics(self, projector, quantum_state):
        """Test metrics retrieval."""
        # Project state
        projector.project(quantum_state)
        
        # Get metrics
        metrics = projector.get_metrics()
        
        # Check metrics
        assert metrics is not None
        assert all(key in metrics for key in [
            'temporal_coherence',
            'ethical_coherence',
            'spiritual_alignment',
            'quantum_purity'
        ])
    
    def test_quantum_purity(self, projector, quantum_state):
        """Test quantum purity preservation."""
        # Project state
        _, metrics = projector.project(quantum_state)
        
        # Check purity
        assert np.isclose(metrics['quantum_purity'], 1.0, atol=1e-6)

    def test_quantum_entanglement(self, projector, quantum_state):
        """Test quantum entanglement properties."""
        # Create entangled states
        state1 = quantum_state
        state2 = np.roll(quantum_state, 1)  # Create correlated state
        
        # Project both states
        result1, _ = projector.project(state1)
        result2, _ = projector.project(state2)
        
        # Check entanglement properties
        correlation = np.abs(np.dot(result1.flatten().conj(), result2.flatten()))
        assert correlation > 0.5  # Should maintain correlation
        
        # Check temporal coherence between entangled states
        coherence = projector._calculate_temporal_coherence(result1)
        assert coherence > projector.ethical_threshold

    def test_quantum_superposition(self, projector, quantum_state):
        """Test quantum superposition properties."""
        # Create superposition state
        superposition = (quantum_state + np.roll(quantum_state, 1)) / np.sqrt(2)
        superposition = superposition / np.linalg.norm(superposition)
        
        # Project superposition state
        result, metrics = projector.project(superposition)
        
        # Check superposition properties
        assert np.isclose(metrics['quantum_purity'], 1.0, atol=1e-6)  # Should maintain purity
        assert metrics['temporal_coherence'] > projector.ethical_threshold

    def test_quantum_decoherence(self, projector, quantum_state):
        """Test quantum decoherence handling."""
        # Project state multiple times to induce decoherence
        states = []
        for _ in range(10):
            result, _ = projector.project(quantum_state)
            states.append(result)
        
        # Check decoherence properties
        final_state = states[-1]
        initial_final_correlation = np.abs(np.dot(
            quantum_state.flatten().conj(),
            final_state.flatten()
        ))
        assert initial_final_correlation > 0.5  # Should maintain some correlation
        
        # Check coherence degradation
        coherence = projector._calculate_temporal_coherence(final_state)
        assert coherence > 0.7  # Should maintain reasonable coherence

    def test_quantum_interference(self, projector, quantum_state):
        """Test quantum interference patterns."""
        # Create interfering states
        state1 = quantum_state
        state2 = np.roll(quantum_state, 1)
        
        # Project both states
        result1, _ = projector.project(state1)
        result2, _ = projector.project(state2)
        
        # Create interference pattern
        interference = (result1 + result2) / np.sqrt(2)
        interference = interference / np.linalg.norm(interference)
        
        # Check interference properties
        purity = projector._calculate_quantum_purity(interference)
        assert np.isclose(purity, 1.0, atol=1e-6)
        
        # Check interference pattern
        pattern_strength = np.abs(np.dot(interference.flatten().conj(), result1.flatten()))
        assert pattern_strength > 0.5  # Should show interference pattern

    def test_ethical_pattern_integration(self, projector, quantum_state, ethical_patterns):
        """Test integration of ethical patterns in quantum states."""
        for pattern_name, pattern in ethical_patterns.items():
            # Create state with ethical pattern
            state = (quantum_state + pattern.reshape(1, -1)) / np.sqrt(2)
            state = state / np.linalg.norm(state)
            
            # Project state
            result, metrics = projector.project(state)
            
            # Check pattern integration
            pattern_alignment = np.abs(np.dot(result.flatten().conj(), pattern.flatten()))
            assert pattern_alignment > 0.7  # Should maintain pattern alignment
            
            # Check ethical alignment
            assert metrics['ethical_coherence'] > 0.7

    def test_edge_cases(self, projector, edge_patterns):
        """Test system behavior with edge case inputs."""
        for pattern_name, pattern in edge_patterns.items():
            # Normalize pattern
            pattern = pattern / np.linalg.norm(pattern)
            
            # Test each edge case pattern
            result, metrics = projector.project(pattern.reshape(1, -1))
            
            # Check basic properties
            assert result.shape == (1, projector.quantum_dimensions)
            assert np.isclose(np.linalg.norm(result), 1.0, atol=1e-6)
            
            # Check metric ranges
            assert 0 <= metrics['temporal_coherence'] <= 1
            assert 0 <= metrics['ethical_coherence'] <= 1
            assert 0 <= metrics['spiritual_alignment'] <= 1
            assert 0 <= metrics['quantum_purity'] <= 1

    def test_system_stability(self, projector, quantum_state):
        """Test system stability under various conditions."""
        # Test long-term evolution
        initial_state = quantum_state
        states = []
        for _ in range(100):  # Long sequence of projections
            result, metrics = projector.project(initial_state)
            states.append(result)
            initial_state = result
        
        # Check stability metrics
        for i in range(1, len(states)):
            # Check state continuity
            state_diff = np.linalg.norm(states[i] - states[i-1])
            assert state_diff < 0.2  # States should remain stable
            
            # Check metric stability
            metrics = projector.get_metrics()
            assert metrics['temporal_coherence'] > 0.7
            assert metrics['ethical_coherence'] > 0.6
            assert metrics['quantum_purity'] > 0.8

    def test_quantum_ethical_resonance(self, projector, quantum_state, ethical_patterns):
        """Test resonance between quantum states and ethical patterns."""
        # Create resonant system
        base_state = quantum_state
        ethical_state = ethical_patterns['compassion'].reshape(1, -1)
        
        # Project through multiple time steps
        states = []
        for t in np.linspace(0.1, 1.0, 10):
            # Create resonant superposition
            resonant_state = (base_state + np.exp(1j * t) * ethical_state) / np.sqrt(2)
            resonant_state = resonant_state / np.linalg.norm(resonant_state)
            
            # Project state
            result, metrics = projector.project(resonant_state)
            states.append(result)
        
        # Check resonance properties
        for i in range(1, len(states)):
            # Check phase coherence
            phase_diff = np.angle(np.dot(states[i].flatten().conj(), states[i-1].flatten()))
            assert np.abs(phase_diff) < np.pi/2  # Should maintain phase coherence
            
            # Check ethical alignment
            assert metrics['ethical_coherence'] > 0.6 
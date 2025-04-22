import pytest
import numpy as np
import tensorflow as tf
from src.core.holographic_compassion_projector import HolographicCompassionProjector

@pytest.fixture
def config():
    return {
        'resolution': 16,
        'depth': 3,
        'coherence_threshold': 0.95,
        'compassion_strength': 0.9,
        'unity_factor': 0.85
    }

@pytest.fixture
def projector(config):
    return HolographicCompassionProjector(config)

@pytest.fixture
def quantum_state():
    state = np.random.rand(1, 16) + 1j * np.random.rand(1, 16)
    return state / np.linalg.norm(state)

@pytest.fixture
def compassion_patterns():
    return {
        'love': np.ones(16) / np.sqrt(16),
        'unity': np.sin(np.linspace(0, 2*np.pi, 16)),
        'forgiveness': np.cos(np.linspace(0, 2*np.pi, 16))
    }

class TestHolographicCompassionProjector:
    def test_initialization(self, projector, config):
        """Test proper initialization of the projector."""
        assert projector.resolution == config['resolution']
        assert projector.depth == config['depth']
        assert projector.coherence_threshold == config['coherence_threshold']
        assert projector.compassion_strength == config['compassion_strength']
        assert projector.unity_factor == config['unity_factor']
        
        # Check state initialization
        assert projector.state['current_state'] is None
        assert len(projector.state['projected_patterns']) == 0
        assert projector.state['metrics'] is None
        
        # Check model architecture
        assert isinstance(projector.holographic_model, tf.keras.Model)
        assert len(projector.holographic_model.layers) == config['depth'] * 3 + 2
        
        # Check compassion patterns
        assert len(projector.compassion_patterns) == 3  # Number of compassion patterns
        for pattern in projector.compassion_patterns.values():
            assert isinstance(pattern, np.ndarray)
            assert pattern.shape == (1, config['resolution'])
            assert np.isclose(np.linalg.norm(pattern), 1.0, atol=1e-6)
    
    def test_project(self, projector, quantum_state):
        """Test pattern projection."""
        # Project state
        projected_state, metrics = projector.project(quantum_state)
        
        # Check output types
        assert isinstance(projected_state, np.ndarray)
        assert isinstance(metrics, dict)
        
        # Check state normalization
        assert np.isclose(np.linalg.norm(projected_state), 1.0, atol=1e-6)
        
        # Check metrics
        assert 'compassion_coherence' in metrics
        assert 'unity_alignment' in metrics
        assert 'spiritual_alignment' in metrics
        assert 'pattern_purity' in metrics
        
        # Check metric values
        assert 0 <= metrics['compassion_coherence'] <= 1
        assert 0 <= metrics['unity_alignment'] <= 1
        assert 0 <= metrics['spiritual_alignment'] <= 1
        assert 0 <= metrics['pattern_purity'] <= 1
    
    def test_compassion_pattern_application(self, projector, quantum_state):
        """Test application of compassion patterns."""
        # Project state
        projected_state, metrics = projector.project(quantum_state)
        
        # Check compassion coherence
        assert metrics['compassion_coherence'] >= projector.coherence_threshold
        
        # Check individual pattern alignments
        for pattern in projector.compassion_patterns.values():
            alignment = np.abs(np.dot(
                projected_state.flatten().conj(),
                pattern.flatten()
            ))
            assert alignment > 0.5  # Should maintain significant alignment
    
    def test_state_evolution(self, projector, quantum_state):
        """Test temporal evolution of projected states."""
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
    
    def test_compassion_coherence(self, projector, quantum_state):
        """Test maintenance of compassion coherence."""
        # Project state
        _, metrics = projector.project(quantum_state)
        
        # Check compassion coherence
        assert metrics['compassion_coherence'] >= projector.coherence_threshold
    
    def test_unity_alignment(self, projector, quantum_state):
        """Test unity alignment maintenance."""
        # Project multiple states
        for _ in range(3):
            _, metrics = projector.project(quantum_state)
        
        # Check unity alignment
        assert metrics['unity_alignment'] >= projector.unity_factor
    
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
        assert len(projector.state['projected_patterns']) == 0
        assert projector.state['metrics'] is None
    
    def test_get_state(self, projector, quantum_state):
        """Test state retrieval."""
        # Project state
        projected_state, _ = projector.project(quantum_state)
        
        # Get state
        state = projector.get_state()
        
        # Check state contents
        assert np.array_equal(state['current_state'], projected_state)
        assert len(state['projected_patterns']) == 1
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
            'compassion_coherence',
            'unity_alignment',
            'spiritual_alignment',
            'pattern_purity'
        ])
    
    def test_pattern_purity(self, projector, quantum_state):
        """Test pattern purity preservation."""
        # Project state
        _, metrics = projector.project(quantum_state)
        
        # Check purity
        assert np.isclose(metrics['pattern_purity'], 1.0, atol=1e-6)
    
    def test_compassion_pattern_generation(self, projector):
        """Test generation of compassion patterns."""
        # Check pattern properties
        for name, pattern in projector.compassion_patterns.items():
            # Check pattern shape
            assert pattern.shape == (1, projector.resolution)
            
            # Check normalization
            assert np.isclose(np.linalg.norm(pattern), 1.0, atol=1e-6)
            
            # Check pattern uniqueness
            for other_name, other_pattern in projector.compassion_patterns.items():
                if name != other_name:
                    correlation = np.abs(np.dot(
                        pattern.flatten().conj(),
                        other_pattern.flatten()
                    ))
                    assert correlation < 0.5  # Patterns should be distinct
    
    def test_compassion_constraints(self, projector, quantum_state):
        """Test application of compassion constraints."""
        # Project state
        projected_state, metrics = projector.project(quantum_state)
        
        # Check compassion constraints
        for pattern in projector.compassion_patterns.values():
            # Calculate alignment
            alignment = np.abs(np.dot(
                projected_state.flatten().conj(),
                pattern.flatten()
            ))
            
            # Check constraint satisfaction
            assert alignment > 0.5  # Should maintain significant alignment
            assert alignment < 1.0  # Should not be perfectly aligned
    
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
            assert metrics['compassion_coherence'] > 0.7
            assert metrics['unity_alignment'] > 0.6
            assert metrics['pattern_purity'] > 0.8 
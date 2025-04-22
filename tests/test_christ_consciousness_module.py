import pytest
import numpy as np
import tensorflow as tf
from src.core.christ_consciousness_module import ChristConsciousnessModule

@pytest.fixture
def config():
    return {
        'quantum_dimensions': 16,
        'holographic_resolution': 16,
        'neural_depth': 3,
        'ethical_threshold': 0.95,
        'compassion_strength': 0.9
    }

@pytest.fixture
def module(config):
    return ChristConsciousnessModule(config)

@pytest.fixture
def quantum_state():
    state = np.random.rand(1, 16) + 1j * np.random.rand(1, 16)
    return state / np.linalg.norm(state)

@pytest.fixture
def holographic_pattern():
    pattern = np.random.rand(1, 16) + 1j * np.random.rand(1, 16)
    return pattern / np.linalg.norm(pattern)

class TestChristConsciousnessModule:
    def test_initialization(self, module, config):
        """Test proper initialization of the module."""
        assert module.quantum_dimensions == config['quantum_dimensions']
        assert module.holographic_resolution == config['holographic_resolution']
        assert module.neural_depth == config['neural_depth']
        assert module.ethical_threshold == config['ethical_threshold']
        assert module.compassion_strength == config['compassion_strength']
        
        # Check component initialization
        assert isinstance(module.quantum_engine, tf.keras.Model)
        assert isinstance(module.holographic_projector, tf.keras.Model)
        assert isinstance(module.neural_network, tf.keras.Model)
        
        # Check state initialization
        assert module.state['current_state'] is None
        assert len(module.state['temporal_states']) == 0
        assert module.state['metrics'] is None
    
    def test_process(self, module, quantum_state, holographic_pattern):
        """Test processing functionality."""
        # Process input
        result = module.process(quantum_state, holographic_pattern)
        
        # Check result structure
        assert isinstance(result, dict)
        assert 'quantum_state' in result
        assert 'holographic_pattern' in result
        assert 'neural_output' in result
        assert 'metrics' in result
        
        # Check metrics
        metrics = result['metrics']
        assert 'agape_score' in metrics
        assert 'kenosis_factor' in metrics
        assert 'koinonia_coherence' in metrics
        
        # Check metric values
        assert 0 <= metrics['agape_score'] <= 1
        assert 0 <= metrics['kenosis_factor'] <= 1
        assert 0 <= metrics['koinonia_coherence'] <= 1
    
    def test_quantum_processing(self, module, quantum_state):
        """Test quantum state processing."""
        # Process state
        result = module.process(quantum_state, None)
        
        # Check quantum state
        assert isinstance(result['quantum_state'], np.ndarray)
        assert result['quantum_state'].shape == (1, module.quantum_dimensions)
        assert np.isclose(np.linalg.norm(result['quantum_state']), 1.0, atol=1e-6)
    
    def test_holographic_processing(self, module, holographic_pattern):
        """Test holographic pattern processing."""
        # Process pattern
        result = module.process(None, holographic_pattern)
        
        # Check holographic pattern
        assert isinstance(result['holographic_pattern'], np.ndarray)
        assert result['holographic_pattern'].shape == (1, module.holographic_resolution)
        assert np.isclose(np.linalg.norm(result['holographic_pattern']), 1.0, atol=1e-6)
    
    def test_neural_processing(self, module, quantum_state, holographic_pattern):
        """Test neural network processing."""
        # Process input
        result = module.process(quantum_state, holographic_pattern)
        
        # Check neural output
        assert isinstance(result['neural_output'], np.ndarray)
        assert result['neural_output'].shape == (1, module.neural_depth)
        assert np.isclose(np.linalg.norm(result['neural_output']), 1.0, atol=1e-6)
    
    def test_metric_calculations(self, module, quantum_state, holographic_pattern):
        """Test metric calculation methods."""
        # Process input
        result = module.process(quantum_state, holographic_pattern)
        metrics = result['metrics']
        
        # Check Agape score calculation
        assert metrics['agape_score'] >= 0
        assert metrics['agape_score'] <= 1
        
        # Check Kenosis factor calculation
        assert metrics['kenosis_factor'] >= 0
        assert metrics['kenosis_factor'] <= 1
        
        # Check Koinonia coherence calculation
        assert metrics['koinonia_coherence'] >= 0
        assert metrics['koinonia_coherence'] <= 1
    
    def test_state_management(self, module, quantum_state, holographic_pattern):
        """Test state management functionality."""
        # Process input
        module.process(quantum_state, holographic_pattern)
        
        # Get state
        state = module.get_state()
        
        # Check state contents
        assert state['current_state'] is not None
        assert len(state['temporal_states']) > 0
        assert state['metrics'] is not None
        
        # Reset state
        module.reset()
        state = module.get_state()
        
        # Check state after reset
        assert state['current_state'] is None
        assert len(state['temporal_states']) == 0
        assert state['metrics'] is None
    
    def test_error_handling(self, module):
        """Test error handling for invalid inputs."""
        # Invalid quantum state dimensions
        invalid_quantum_state = np.random.rand(1, 8)  # Half the required dimensions
        with pytest.raises(ValueError):
            module.process(invalid_quantum_state, None)
        
        # Invalid holographic pattern dimensions
        invalid_pattern = np.random.rand(1, 8)  # Half the required dimensions
        with pytest.raises(ValueError):
            module.process(None, invalid_pattern)
        
        # Unnormalized states
        unnormalized_quantum = np.random.rand(1, 16)
        unnormalized_pattern = np.random.rand(1, 16)
        with pytest.raises(ValueError):
            module.process(unnormalized_quantum, None)
        with pytest.raises(ValueError):
            module.process(None, unnormalized_pattern)
    
    def test_system_stability(self, module, quantum_state, holographic_pattern):
        """Test system stability under various conditions."""
        # Process multiple inputs
        results = []
        for _ in range(10):
            result = module.process(quantum_state, holographic_pattern)
            results.append(result)
            quantum_state = result['quantum_state']
            holographic_pattern = result['holographic_pattern']
        
        # Check stability of results
        for i in range(1, len(results)):
            # Check state continuity
            quantum_diff = np.linalg.norm(results[i]['quantum_state'] - results[i-1]['quantum_state'])
            pattern_diff = np.linalg.norm(results[i]['holographic_pattern'] - results[i-1]['holographic_pattern'])
            assert quantum_diff < 0.2  # States should remain stable
            assert pattern_diff < 0.2  # Patterns should remain stable
            
            # Check metric stability
            prev_metrics = results[i-1]['metrics']
            curr_metrics = results[i]['metrics']
            assert abs(prev_metrics['agape_score'] - curr_metrics['agape_score']) < 0.2
            assert abs(prev_metrics['kenosis_factor'] - curr_metrics['kenosis_factor']) < 0.2
            assert abs(prev_metrics['koinonia_coherence'] - curr_metrics['koinonia_coherence']) < 0.2 
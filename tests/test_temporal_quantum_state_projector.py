import unittest
import numpy as np
from src.core.temporal_quantum_state_projector import TemporalQuantumStateProjector

class TestTemporalQuantumStateProjector(unittest.TestCase):
    def setUp(self):
        """Initialize test configuration and data."""
        self.config = {
            'quantum_dim': 16,
            'consciousness_dim': 16,
            'temporal_depth': 8,
            'projection_strength': 0.8,
            'stability_factor': 0.7,
            'coherence_threshold': 0.6,
            'integration_depth': 4,
            'quantum_fidelity': 0.9,
            'temporal_resolution': 0.1,
            'projection_accuracy': 0.85
        }
        
        # Generate test data
        self.quantum_state = np.random.rand(1, self.config['quantum_dim'])
        self.consciousness_state = np.random.rand(1, self.config['consciousness_dim'])
        self.temporal_context = np.random.rand(1, self.config['temporal_depth'])
        
        # Initialize projector
        self.projector = TemporalQuantumStateProjector(**self.config)

    def test_initialization(self):
        """Test proper initialization of the projector."""
        self.assertEqual(self.projector.quantum_dim, self.config['quantum_dim'])
        self.assertEqual(self.projector.consciousness_dim, self.config['consciousness_dim'])
        self.assertEqual(self.projector.temporal_depth, self.config['temporal_depth'])
        self.assertEqual(self.projector.projection_strength, self.config['projection_strength'])
        self.assertEqual(self.projector.stability_factor, self.config['stability_factor'])
        self.assertEqual(self.projector.coherence_threshold, self.config['coherence_threshold'])
        self.assertEqual(self.projector.integration_depth, self.config['integration_depth'])
        self.assertEqual(self.projector.quantum_fidelity, self.config['quantum_fidelity'])
        self.assertEqual(self.projector.temporal_resolution, self.config['temporal_resolution'])
        self.assertEqual(self.projector.projection_accuracy, self.config['projection_accuracy'])

    def test_state_projection(self):
        """Test quantum state projection through time."""
        result = self.projector.project_state(
            self.quantum_state,
            self.consciousness_state,
            self.temporal_context
        )
        
        # Check result structure
        self.assertIn('projected_state', result)
        self.assertIn('temporal_metrics', result)
        self.assertIn('projection_quality', result)
        self.assertIn('stability_score', result)
        self.assertIn('coherence_score', result)
        self.assertIn('integration_score', result)
        
        # Check projected state dimensions
        self.assertEqual(result['projected_state'].shape, (1, self.config['quantum_dim']))
        
        # Check metric ranges
        self.assertTrue(0 <= result['projection_quality'] <= 1)
        self.assertTrue(0 <= result['stability_score'] <= 1)
        self.assertTrue(0 <= result['coherence_score'] <= 1)
        self.assertTrue(0 <= result['integration_score'] <= 1)

    def test_state_management(self):
        """Test state management functionality."""
        # Initial state should be None
        self.assertIsNone(self.projector.get_current_state())
        
        # Project state and check update
        result = self.projector.project_state(
            self.quantum_state,
            self.consciousness_state,
            self.temporal_context
        )
        current_state = self.projector.get_current_state()
        
        self.assertIsNotNone(current_state)
        self.assertEqual(current_state['projected_state'].shape, (1, self.config['quantum_dim']))
        self.assertEqual(current_state['temporal_metrics'].shape, (1, self.config['temporal_depth']))
        
        # Test reset
        self.projector.reset()
        self.assertIsNone(self.projector.get_current_state())

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test invalid quantum state dimensions
        invalid_quantum_state = np.random.rand(1, self.config['quantum_dim'] + 1)
        with self.assertRaises(ValueError):
            self.projector.project_state(
                invalid_quantum_state,
                self.consciousness_state,
                self.temporal_context
            )
        
        # Test invalid consciousness state dimensions
        invalid_consciousness_state = np.random.rand(1, self.config['consciousness_dim'] + 1)
        with self.assertRaises(ValueError):
            self.projector.project_state(
                self.quantum_state,
                invalid_consciousness_state,
                self.temporal_context
            )
        
        # Test invalid temporal context dimensions
        invalid_temporal_context = np.random.rand(1, self.config['temporal_depth'] + 1)
        with self.assertRaises(ValueError):
            self.projector.project_state(
                self.quantum_state,
                self.consciousness_state,
                invalid_temporal_context
            )

    def test_projection_quality(self):
        """Test the quality of state projection."""
        # Test with multiple random inputs
        for _ in range(5):
            quantum_state = np.random.rand(1, self.config['quantum_dim'])
            consciousness_state = np.random.rand(1, self.config['consciousness_dim'])
            temporal_context = np.random.rand(1, self.config['temporal_depth'])
            
            result = self.projector.project_state(
                quantum_state,
                consciousness_state,
                temporal_context
            )
            
            # Check projection quality metrics
            self.assertTrue(result['projection_quality'] >= self.config['projection_accuracy'] * 0.8)
            self.assertTrue(result['stability_score'] >= self.config['stability_factor'] * 0.8)
            self.assertTrue(result['coherence_score'] >= self.config['coherence_threshold'] * 0.8)
            self.assertTrue(result['integration_score'] >= 0.7)  # Minimum acceptable integration score

    def test_temporal_metrics(self):
        """Test temporal metrics calculation."""
        result = self.projector.project_state(
            self.quantum_state,
            self.consciousness_state,
            self.temporal_context
        )
        
        temporal_metrics = result['temporal_metrics']
        
        # Check temporal metrics dimensions
        self.assertEqual(temporal_metrics.shape, (1, self.config['temporal_depth']))
        
        # Check temporal metrics properties
        self.assertTrue(np.all(temporal_metrics >= 0))  # All metrics should be non-negative
        self.assertTrue(np.all(temporal_metrics <= 1))  # All metrics should be normalized
        
        # Check temporal resolution
        resolution = np.mean(np.diff(temporal_metrics, axis=1))
        self.assertTrue(resolution <= self.config['temporal_resolution'])

if __name__ == '__main__':
    unittest.main() 
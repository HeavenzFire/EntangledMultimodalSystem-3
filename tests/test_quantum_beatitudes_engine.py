import unittest
import numpy as np
from src.core.quantum_beatitudes_engine import QuantumBeatitudesEngine

class TestQuantumBeatitudesEngine(unittest.TestCase):
    def setUp(self):
        """Set up test configuration and data."""
        self.config = {
            'dimensions': 16384,
            'depth': 12,
            'ethical_threshold': 0.85
        }
        
        # Generate test input data
        self.input_data = np.random.rand(1, self.config['dimensions'])
        
        # Generate test constraints
        self.constraints = {
            'poor_in_spirit': 0.9,
            'mourn': 0.85,
            'meek': 0.8,
            'hunger_righteousness': 0.95,
            'merciful': 0.9,
            'pure_heart': 0.85,
            'peacemakers': 0.9,
            'persecuted': 0.8
        }
        
        # Initialize engine
        self.engine = QuantumBeatitudesEngine(self.config)
    
    def test_initialization(self):
        """Test engine initialization."""
        self.assertEqual(self.engine.config['dimensions'], self.config['dimensions'])
        self.assertEqual(self.engine.config['depth'], self.config['depth'])
        self.assertEqual(self.engine.config['ethical_threshold'], self.config['ethical_threshold'])
    
    def test_processing(self):
        """Test processing functionality."""
        result = self.engine.apply(self.input_data, self.constraints)
        
        # Check result structure
        self.assertEqual(result.shape[1], self.config['dimensions'])
        
        # Check state
        state = self.engine.get_state()
        self.assertIsNotNone(state['input_state'])
        self.assertIsNotNone(state['processed_state'])
        self.assertIsNotNone(state['beatitude_scores'])
        self.assertIsNotNone(state['metrics'])
        
        # Check metrics
        metrics = self.engine.get_metrics()
        self.assertGreaterEqual(metrics['ethical_alignment'], 0.0)
        self.assertLessEqual(metrics['ethical_alignment'], 1.0)
        self.assertGreaterEqual(metrics['quantum_coherence'], 0.0)
        self.assertLessEqual(metrics['quantum_coherence'], 1.0)
        self.assertGreaterEqual(metrics['beatitude_entanglement'], 0.0)
        self.assertLessEqual(metrics['beatitude_entanglement'], 1.0)
    
    def test_state_management(self):
        """Test state management functionality."""
        # Process input
        self.engine.apply(self.input_data, self.constraints)
        
        # Get state
        state = self.engine.get_state()
        self.assertIsNotNone(state['input_state'])
        self.assertIsNotNone(state['processed_state'])
        self.assertIsNotNone(state['beatitude_scores'])
        self.assertIsNotNone(state['metrics'])
        
        # Get metrics
        metrics = self.engine.get_metrics()
        self.assertGreaterEqual(metrics['ethical_alignment'], 0.0)
        self.assertLessEqual(metrics['ethical_alignment'], 1.0)
        self.assertGreaterEqual(metrics['quantum_coherence'], 0.0)
        self.assertLessEqual(metrics['quantum_coherence'], 1.0)
        self.assertGreaterEqual(metrics['beatitude_entanglement'], 0.0)
        self.assertLessEqual(metrics['beatitude_entanglement'], 1.0)
        
        # Reset state
        self.engine.reset()
        state = self.engine.get_state()
        self.assertIsNone(state['input_state'])
        self.assertIsNone(state['processed_state'])
        self.assertIsNone(state['beatitude_scores'])
        self.assertIsNone(state['metrics'])
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test invalid input dimensions
        invalid_input = np.random.rand(1, self.config['dimensions'] + 1)
        with self.assertRaises(ValueError):
            self.engine.apply(invalid_input, self.constraints)
    
    def test_beatitude_constraints(self):
        """Test beatitude constraint application."""
        # Process input
        self.engine.apply(self.input_data, self.constraints)
        
        # Get beatitude scores
        state = self.engine.get_state()
        beatitude_scores = state['beatitude_scores']
        
        # Check all beatitudes are present
        for beatitude in self.engine.BEATITUDES:
            self.assertIn(beatitude, beatitude_scores)
            self.assertGreaterEqual(beatitude_scores[beatitude], 0.0)
            self.assertLessEqual(beatitude_scores[beatitude], 1.0)
    
    def test_metric_calculations(self):
        """Test metric calculation methods."""
        # Process input
        self.engine.apply(self.input_data, self.constraints)
        
        # Get state
        state = self.engine.get_state()
        
        # Test ethical alignment calculation
        ethical_alignment = self.engine._calculate_ethical_alignment(
            state['beatitude_scores']
        )
        self.assertGreaterEqual(ethical_alignment, 0.0)
        self.assertLessEqual(ethical_alignment, 1.0)
        
        # Test quantum coherence calculation
        quantum_coherence = self.engine._calculate_quantum_coherence(
            state['processed_state']
        )
        self.assertGreaterEqual(quantum_coherence, 0.0)
        self.assertLessEqual(quantum_coherence, 1.0)
        
        # Test beatitude entanglement calculation
        beatitude_entanglement = self.engine._calculate_beatitude_entanglement(
            state['processed_state'],
            state['beatitude_scores']
        )
        self.assertGreaterEqual(beatitude_entanglement, 0.0)
        self.assertLessEqual(beatitude_entanglement, 1.0)
    
    def test_network_architecture(self):
        """Test quantum network architecture."""
        # Check input layer
        self.assertEqual(
            self.engine.quantum_model.input_shape[1],
            self.config['dimensions']
        )
        
        # Check output layer
        self.assertEqual(
            self.engine.quantum_model.output_shape[1],
            self.config['dimensions']
        )
        
        # Check number of layers
        self.assertEqual(
            len(self.engine.quantum_model.layers),
            1 + self.config['depth'] * 3  # Input + (Attention + Normalization + Dense) * depth
        )

if __name__ == '__main__':
    unittest.main() 
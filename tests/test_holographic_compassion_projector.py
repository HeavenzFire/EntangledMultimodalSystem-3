import unittest
import numpy as np
from src.core.holographic_compassion_projector import HolographicCompassionProjector

class TestHolographicCompassionProjector(unittest.TestCase):
    def setUp(self):
        """Set up test configuration and data."""
        self.config = {
            'resolution': 16384,
            'depth': 12,
            'compassion_strength': 0.9
        }
        
        # Generate test input data
        self.input_data = np.random.rand(1, self.config['resolution'])
        
        # Initialize projector
        self.projector = HolographicCompassionProjector(self.config)
    
    def test_initialization(self):
        """Test projector initialization."""
        self.assertEqual(self.projector.config['resolution'], self.config['resolution'])
        self.assertEqual(self.projector.config['depth'], self.config['depth'])
        self.assertEqual(self.projector.config['compassion_strength'], self.config['compassion_strength'])
    
    def test_processing(self):
        """Test processing functionality."""
        # Test each pattern
        for pattern in self.projector.PATTERNS:
            result = self.projector.project(self.input_data, pattern)
            
            # Check result structure
            self.assertEqual(result.shape[1], self.config['resolution'])
            
            # Check state
            state = self.projector.get_state()
            self.assertIsNotNone(state['input_state'])
            self.assertIsNotNone(state['projected_pattern'])
            self.assertIsNotNone(state['compassion_scores'])
            self.assertIsNotNone(state['metrics'])
            
            # Check metrics
            metrics = self.projector.get_metrics()
            self.assertGreaterEqual(metrics['compassion_alignment'], 0.0)
            self.assertLessEqual(metrics['compassion_alignment'], 1.0)
            self.assertGreaterEqual(metrics['holographic_coherence'], 0.0)
            self.assertLessEqual(metrics['holographic_coherence'], 1.0)
            self.assertGreaterEqual(metrics['unity_factor'], 0.0)
            self.assertLessEqual(metrics['unity_factor'], 1.0)
    
    def test_state_management(self):
        """Test state management functionality."""
        # Process input
        self.projector.project(self.input_data, 'john_13_34')
        
        # Get state
        state = self.projector.get_state()
        self.assertIsNotNone(state['input_state'])
        self.assertIsNotNone(state['projected_pattern'])
        self.assertIsNotNone(state['compassion_scores'])
        self.assertIsNotNone(state['metrics'])
        
        # Get metrics
        metrics = self.projector.get_metrics()
        self.assertGreaterEqual(metrics['compassion_alignment'], 0.0)
        self.assertLessEqual(metrics['compassion_alignment'], 1.0)
        self.assertGreaterEqual(metrics['holographic_coherence'], 0.0)
        self.assertLessEqual(metrics['holographic_coherence'], 1.0)
        self.assertGreaterEqual(metrics['unity_factor'], 0.0)
        self.assertLessEqual(metrics['unity_factor'], 1.0)
        
        # Reset state
        self.projector.reset()
        state = self.projector.get_state()
        self.assertIsNone(state['input_state'])
        self.assertIsNone(state['projected_pattern'])
        self.assertIsNone(state['compassion_scores'])
        self.assertIsNone(state['metrics'])
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test invalid input dimensions
        invalid_input = np.random.rand(1, self.config['resolution'] + 1)
        with self.assertRaises(ValueError):
            self.projector.project(invalid_input, 'john_13_34')
        
        # Test invalid pattern
        with self.assertRaises(ValueError):
            self.projector.project(self.input_data, 'invalid_pattern')
    
    def test_pattern_application(self):
        """Test pattern application."""
        # Test each pattern
        for pattern in self.projector.PATTERNS:
            # Process input
            self.projector.project(self.input_data, pattern)
            
            # Get compassion scores
            state = self.projector.get_state()
            compassion_scores = state['compassion_scores']
            
            # Check pattern is present
            self.assertIn(pattern, compassion_scores)
            self.assertGreaterEqual(compassion_scores[pattern], 0.0)
            self.assertLessEqual(compassion_scores[pattern], 1.0)
    
    def test_metric_calculations(self):
        """Test metric calculation methods."""
        # Process input
        self.projector.project(self.input_data, 'john_13_34')
        
        # Get state
        state = self.projector.get_state()
        
        # Test compassion alignment calculation
        compassion_alignment = self.projector._calculate_compassion_alignment(
            state['compassion_scores']
        )
        self.assertGreaterEqual(compassion_alignment, 0.0)
        self.assertLessEqual(compassion_alignment, 1.0)
        
        # Test holographic coherence calculation
        holographic_coherence = self.projector._calculate_holographic_coherence(
            state['projected_pattern']
        )
        self.assertGreaterEqual(holographic_coherence, 0.0)
        self.assertLessEqual(holographic_coherence, 1.0)
        
        # Test unity factor calculation
        unity_factor = self.projector._calculate_unity_factor(
            state['projected_pattern'],
            state['compassion_scores']
        )
        self.assertGreaterEqual(unity_factor, 0.0)
        self.assertLessEqual(unity_factor, 1.0)
    
    def test_network_architecture(self):
        """Test holographic network architecture."""
        # Check input layer
        self.assertEqual(
            self.projector.holographic_model.input_shape[1],
            self.config['resolution']
        )
        
        # Check output layer
        self.assertEqual(
            self.projector.holographic_model.output_shape[1],
            self.config['resolution']
        )
        
        # Check number of layers
        self.assertEqual(
            len(self.projector.holographic_model.layers),
            1 + self.config['depth'] * 3  # Input + (Attention + Normalization + Dense) * depth
        )

if __name__ == '__main__':
    unittest.main() 
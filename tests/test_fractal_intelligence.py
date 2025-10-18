import unittest
import numpy as np
from src.core.fractal_intelligence import FractalIntelligenceEngine
from src.utils.errors import ModelError

class TestFractalIntelligenceEngine(unittest.TestCase):
    """Test cases for the Fractal Intelligence Engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "max_iterations": 100,
            "resolution": 4096,  # 8K resolution
            "fps": 120,
            "semantic_dimensions": 512
        }
        self.engine = FractalIntelligenceEngine(self.config)
        
        # Create test data
        self.parameters = np.array([0.0, 0.0, 1.0, 0.0])  # x, y, zoom, rotation
        self.semantic_context = np.random.rand(self.config["semantic_dimensions"])
    
    def test_initialization(self):
        """Test system initialization."""
        self.assertEqual(self.engine.max_iterations, self.config["max_iterations"])
        self.assertEqual(self.engine.resolution, self.config["resolution"])
        self.assertEqual(self.engine.fps, self.config["fps"])
        self.assertEqual(self.engine.semantic_dimensions, self.config["semantic_dimensions"])
        
        # Test state initialization
        self.assertIsNone(self.engine.state["fractal_data"])
        self.assertIsNone(self.engine.state["semantic_embedding"])
        self.assertIsNone(self.engine.state["generation_parameters"])
        self.assertIsNone(self.engine.state["processing_results"])
        
        # Test metrics initialization
        self.assertEqual(self.engine.metrics["fractal_quality"], 0.0)
        self.assertEqual(self.engine.metrics["semantic_coherence"], 0.0)
        self.assertEqual(self.engine.metrics["generation_time"], 0.0)
        self.assertEqual(self.engine.metrics["resolution"], self.config["resolution"])
        self.assertEqual(self.engine.metrics["fps"], self.config["fps"])
    
    def test_fractal_generation(self):
        """Test fractal generation."""
        result = self.engine.generate_fractal(
            self.parameters,
            self.semantic_context
        )
        
        # Test result structure
        self.assertIn("fractal_data", result)
        self.assertIn("semantic_embedding", result)
        self.assertIn("processing", result)
        self.assertIn("metrics", result)
        self.assertIn("state", result)
        
        # Test fractal data properties
        fractal_data = result["fractal_data"]
        self.assertEqual(len(fractal_data), self.config["resolution"] * self.config["resolution"] * 3)
        self.assertTrue(np.all(-1 <= fractal_data))
        self.assertTrue(np.all(fractal_data <= 1))
    
    def test_processing_results(self):
        """Test processing results."""
        result = self.engine.generate_fractal(
            self.parameters,
            self.semantic_context
        )
        processing = result["processing"]
        
        # Test processing properties
        self.assertIn("fractal_quality", processing)
        self.assertIn("semantic_coherence", processing)
        
        # Test processing ranges
        self.assertTrue(0 <= processing["fractal_quality"] <= 1)
        self.assertTrue(0 <= processing["semantic_coherence"] <= 1)
    
    def test_metrics_calculation(self):
        """Test metrics calculation."""
        result = self.engine.generate_fractal(
            self.parameters,
            self.semantic_context
        )
        metrics = result["metrics"]
        
        # Test metric properties
        self.assertIn("fractal_quality", metrics)
        self.assertIn("semantic_coherence", metrics)
        self.assertIn("generation_time", metrics)
        self.assertIn("resolution", metrics)
        self.assertIn("fps", metrics)
        
        # Test metric ranges
        self.assertTrue(0 <= metrics["fractal_quality"] <= 1)
        self.assertTrue(0 <= metrics["semantic_coherence"] <= 1)
        self.assertEqual(metrics["generation_time"], 0.008)  # 8ms for 8K @ 120fps
        self.assertEqual(metrics["resolution"], self.config["resolution"])
        self.assertEqual(metrics["fps"], self.config["fps"])
    
    def test_state_management(self):
        """Test state management."""
        # Test initial state
        initial_state = self.engine.get_state()
        self.assertIsNone(initial_state["fractal_data"])
        self.assertIsNone(initial_state["semantic_embedding"])
        self.assertIsNone(initial_state["generation_parameters"])
        self.assertIsNone(initial_state["processing_results"])
        
        # Generate fractal and test updated state
        self.engine.generate_fractal(
            self.parameters,
            self.semantic_context
        )
        updated_state = self.engine.get_state()
        self.assertIsNotNone(updated_state["fractal_data"])
        self.assertIsNotNone(updated_state["semantic_embedding"])
        self.assertIsNotNone(updated_state["generation_parameters"])
        self.assertIsNotNone(updated_state["processing_results"])
        
        # Test reset
        self.engine.reset()
        reset_state = self.engine.get_state()
        self.assertIsNone(reset_state["fractal_data"])
        self.assertIsNone(reset_state["semantic_embedding"])
        self.assertIsNone(reset_state["generation_parameters"])
        self.assertIsNone(reset_state["processing_results"])
    
    def test_error_handling(self):
        """Test error handling."""
        # Test invalid parameters
        invalid_parameters = np.array([0.0, 0.0, 1.0])  # Missing rotation
        with self.assertRaises(ModelError):
            self.engine.generate_fractal(
                invalid_parameters,
                self.semantic_context
            )
        
        # Test invalid semantic context
        invalid_context = np.random.rand(self.config["semantic_dimensions"] + 1)
        with self.assertRaises(ModelError):
            self.engine.generate_fractal(
                self.parameters,
                invalid_context
            )
    
    def test_fractal_quality(self):
        """Test fractal generation quality."""
        # Test with different parameters
        for _ in range(3):
            # Generate new test data
            parameters = np.random.rand(4)  # Random x, y, zoom, rotation
            semantic_context = np.random.rand(self.config["semantic_dimensions"])
            
            result = self.engine.generate_fractal(
                parameters,
                semantic_context
            )
            
            # Test fractal data
            fractal_data = result["fractal_data"]
            self.assertEqual(len(fractal_data), self.config["resolution"] * self.config["resolution"] * 3)
            self.assertTrue(np.all(-1 <= fractal_data))
            self.assertTrue(np.all(fractal_data <= 1))
            
            # Test processing consistency
            processing = result["processing"]
            self.assertTrue(0 <= processing["fractal_quality"] <= 1)
            self.assertTrue(0 <= processing["semantic_coherence"] <= 1)

if __name__ == '__main__':
    unittest.main() 
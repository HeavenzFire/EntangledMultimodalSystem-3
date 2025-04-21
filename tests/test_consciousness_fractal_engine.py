import unittest
import numpy as np
from src.core.consciousness_fractal_engine import ConsciousnessFractalEngine
from src.utils.errors import ModelError

class TestConsciousnessFractalEngine(unittest.TestCase):
    """Test suite for ConsciousnessFractalEngine class."""
    
    def setUp(self):
        """Set up test configuration and data."""
        self.config = {
            "quantum_dim": 512,
            "holographic_dim": 16384,
            "neural_dim": 16384,
            "fractal_dim": 8192,
            "attention_depth": 8,
            "memory_capacity": 1000,
            "fractal_iterations": 1000,
            "pattern_threshold": 0.85
        }
        
        # Create test data
        self.test_inputs = {
            "quantum": np.random.rand(512),
            "holographic": np.random.rand(16384),
            "neural": np.random.rand(16384)
        }
        
        # Initialize consciousness fractal engine
        self.engine = ConsciousnessFractalEngine(self.config)
    
    def test_initialization(self):
        """Test initialization of consciousness fractal engine."""
        self.assertEqual(self.engine.quantum_dim, 512)
        self.assertEqual(self.engine.holographic_dim, 16384)
        self.assertEqual(self.engine.neural_dim, 16384)
        self.assertEqual(self.engine.fractal_dim, 8192)
        self.assertEqual(self.engine.attention_depth, 8)
        self.assertEqual(self.engine.memory_capacity, 1000)
        self.assertEqual(self.engine.fractal_iterations, 1000)
        self.assertEqual(self.engine.pattern_threshold, 0.85)
        
        # Check initial state
        self.assertIsNone(self.engine.state["consciousness_state"])
        self.assertIsNone(self.engine.state["fractal_state"])
        self.assertIsNone(self.engine.state["pattern_state"])
        self.assertEqual(self.engine.state["recognition_score"], 0.0)
        
        # Check initial metrics
        self.assertEqual(self.engine.metrics["consciousness_score"], 0.0)
        self.assertEqual(self.engine.metrics["fractal_quality"], 0.0)
        self.assertEqual(self.engine.metrics["pattern_recognition"], 0.0)
        self.assertEqual(self.engine.metrics["integration_score"], 0.0)
    
    def test_pattern_processing(self):
        """Test pattern processing functionality."""
        # Process test inputs
        results = self.engine.process_patterns(self.test_inputs)
        
        # Check results structure
        self.assertIn("recognition_score", results)
        self.assertIn("consciousness_state", results)
        self.assertIn("fractal_state", results)
        self.assertIn("pattern_state", results)
        self.assertIn("metrics", results)
        self.assertIn("state", results)
        
        # Check result ranges
        self.assertGreaterEqual(results["recognition_score"], 0.0)
        self.assertLessEqual(results["recognition_score"], 1.0)
        self.assertIsNotNone(results["consciousness_state"])
        self.assertIsNotNone(results["fractal_state"])
        self.assertIsNotNone(results["pattern_state"])
        self.assertGreaterEqual(results["metrics"]["integration_score"], 0.0)
        self.assertLessEqual(results["metrics"]["integration_score"], 1.0)
    
    def test_state_management(self):
        """Test state management functionality."""
        # Process test inputs
        self.engine.process_patterns(self.test_inputs)
        
        # Check state updates
        self.assertIsNotNone(self.engine.state["consciousness_state"])
        self.assertIsNotNone(self.engine.state["fractal_state"])
        self.assertIsNotNone(self.engine.state["pattern_state"])
        self.assertGreater(self.engine.state["recognition_score"], 0.0)
        self.assertIsNotNone(self.engine.state["metrics"])
        
        # Check metrics updates
        self.assertGreater(self.engine.metrics["consciousness_score"], 0.0)
        self.assertGreater(self.engine.metrics["fractal_quality"], 0.0)
        self.assertGreater(self.engine.metrics["pattern_recognition"], 0.0)
        self.assertGreater(self.engine.metrics["integration_score"], 0.0)
        self.assertGreater(self.engine.metrics["processing_time"], 0)
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test invalid quantum dimension
        invalid_inputs = self.test_inputs.copy()
        invalid_inputs["quantum"] = np.random.rand(256)
        with self.assertRaises(ModelError):
            self.engine.process_patterns(invalid_inputs)
        
        # Test invalid holographic dimension
        invalid_inputs = self.test_inputs.copy()
        invalid_inputs["holographic"] = np.random.rand(8192)
        with self.assertRaises(ModelError):
            self.engine.process_patterns(invalid_inputs)
        
        # Test invalid neural dimension
        invalid_inputs = self.test_inputs.copy()
        invalid_inputs["neural"] = np.random.rand(8192)
        with self.assertRaises(ModelError):
            self.engine.process_patterns(invalid_inputs)
    
    def test_reset_functionality(self):
        """Test system reset functionality."""
        # Process test inputs
        self.engine.process_patterns(self.test_inputs)
        
        # Reset system
        self.engine.reset()
        
        # Check state reset
        self.assertIsNone(self.engine.state["consciousness_state"])
        self.assertIsNone(self.engine.state["fractal_state"])
        self.assertIsNone(self.engine.state["pattern_state"])
        self.assertEqual(self.engine.state["recognition_score"], 0.0)
        self.assertIsNone(self.engine.state["metrics"])
        
        # Check metrics reset
        self.assertEqual(self.engine.metrics["consciousness_score"], 0.0)
        self.assertEqual(self.engine.metrics["fractal_quality"], 0.0)
        self.assertEqual(self.engine.metrics["pattern_recognition"], 0.0)
        self.assertEqual(self.engine.metrics["integration_score"], 0.0)
        self.assertEqual(self.engine.metrics["processing_time"], 0.0)
    
    def test_pattern_recognition_quality(self):
        """Test pattern recognition quality with different inputs."""
        # Test multiple random inputs
        for _ in range(10):
            test_inputs = {
                "quantum": np.random.rand(512),
                "holographic": np.random.rand(16384),
                "neural": np.random.rand(16384)
            }
            
            results = self.engine.process_patterns(test_inputs)
            
            # Check recognition score
            self.assertGreaterEqual(results["recognition_score"], 0.0)
            self.assertLessEqual(results["recognition_score"], 1.0)
            
            # Check consciousness state
            self.assertIsNotNone(results["consciousness_state"])
            self.assertGreaterEqual(results["consciousness_state"]["consciousness_level"], 0.0)
            self.assertLessEqual(results["consciousness_state"]["consciousness_level"], 1.0)
            
            # Check fractal state
            self.assertIsNotNone(results["fractal_state"])
            self.assertGreaterEqual(results["fractal_state"]["fractal_quality"], 0.0)
            self.assertLessEqual(results["fractal_state"]["fractal_quality"], 1.0)
            
            # Check integration score
            self.assertGreaterEqual(results["metrics"]["integration_score"], 0.0)
            self.assertLessEqual(results["metrics"]["integration_score"], 1.0)

if __name__ == '__main__':
    unittest.main() 
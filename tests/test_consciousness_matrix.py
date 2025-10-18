import unittest
import numpy as np
from src.core.consciousness_matrix import ConsciousnessMatrix
from src.utils.errors import ModelError

class TestConsciousnessMatrix(unittest.TestCase):
    """Test suite for ConsciousnessMatrix class."""
    
    def setUp(self):
        """Set up test configuration and data."""
        self.config = {
            "quantum_dim": 512,
            "holographic_dim": 16384,
            "neural_dim": 16384,
            "attention_depth": 8,
            "memory_capacity": 1000,
            "consciousness_threshold": 0.9,
            "ethical_threshold": 0.8,
            "governance_threshold": 0.85
        }
        
        # Create test data
        self.test_inputs = {
            "quantum": np.random.rand(512),
            "holographic": np.random.rand(16384),
            "neural": np.random.rand(16384)
        }
        
        # Initialize consciousness matrix
        self.matrix = ConsciousnessMatrix(self.config)
    
    def test_initialization(self):
        """Test initialization of consciousness matrix."""
        self.assertEqual(self.matrix.quantum_dim, 512)
        self.assertEqual(self.matrix.holographic_dim, 16384)
        self.assertEqual(self.matrix.neural_dim, 16384)
        self.assertEqual(self.matrix.attention_depth, 8)
        self.assertEqual(self.matrix.memory_capacity, 1000)
        self.assertEqual(self.matrix.consciousness_threshold, 0.9)
        self.assertEqual(self.matrix.ethical_threshold, 0.8)
        self.assertEqual(self.matrix.governance_threshold, 0.85)
        
        # Check initial state
        self.assertIsNone(self.matrix.state["quantum_state"])
        self.assertIsNone(self.matrix.state["holographic_state"])
        self.assertIsNone(self.matrix.state["neural_state"])
        self.assertEqual(self.matrix.state["consciousness_level"], 0.0)
        self.assertIsNone(self.matrix.state["ethical_state"])
        self.assertIsNone(self.matrix.state["governance_state"])
        
        # Check initial metrics
        self.assertEqual(self.matrix.metrics["attention_score"], 0.0)
        self.assertEqual(self.matrix.metrics["memory_retention"], 0.0)
        self.assertEqual(self.matrix.metrics["consciousness_level"], 0.0)
        self.assertEqual(self.matrix.metrics["integration_score"], 0.0)
        self.assertEqual(self.matrix.metrics["ethical_score"], 0.0)
        self.assertEqual(self.matrix.metrics["governance_score"], 0.0)
    
    def test_consciousness_processing(self):
        """Test consciousness processing functionality with ethical governance."""
        # Process test inputs
        results = self.matrix.process_consciousness(self.test_inputs)
        
        # Check results structure
        self.assertIn("consciousness_level", results)
        self.assertIn("attention_scores", results)
        self.assertIn("memory_retention", results)
        self.assertIn("integration_score", results)
        self.assertIn("ethical_score", results)
        self.assertIn("governance_score", results)
        self.assertIn("metrics", results)
        self.assertIn("state", results)
        
        # Check result ranges
        self.assertGreaterEqual(results["consciousness_level"], 0.0)
        self.assertLessEqual(results["consciousness_level"], 1.0)
        self.assertEqual(len(results["attention_scores"]), 3)
        self.assertTrue(np.all(results["attention_scores"] >= 0.0))
        self.assertTrue(np.all(results["attention_scores"] <= 1.0))
        self.assertGreaterEqual(results["memory_retention"], 0.0)
        self.assertLessEqual(results["memory_retention"], 1.0)
        self.assertGreaterEqual(results["integration_score"], 0.0)
        self.assertLessEqual(results["integration_score"], 1.0)
        self.assertGreaterEqual(results["ethical_score"], 0.0)
        self.assertLessEqual(results["ethical_score"], 1.0)
        self.assertGreaterEqual(results["governance_score"], 0.0)
        self.assertLessEqual(results["governance_score"], 1.0)
    
    def test_state_management(self):
        """Test state management functionality."""
        # Process test inputs
        self.matrix.process_consciousness(self.test_inputs)
        
        # Check state updates
        self.assertIsNotNone(self.matrix.state["quantum_state"])
        self.assertIsNotNone(self.matrix.state["holographic_state"])
        self.assertIsNotNone(self.matrix.state["neural_state"])
        self.assertIsNotNone(self.matrix.state["attention_state"])
        self.assertIsNotNone(self.matrix.state["memory_state"])
        self.assertGreater(self.matrix.state["consciousness_level"], 0.0)
        self.assertIsNotNone(self.matrix.state["ethical_state"])
        self.assertIsNotNone(self.matrix.state["governance_state"])
        self.assertIsNotNone(self.matrix.state["metrics"])
        
        # Check metrics updates
        self.assertGreater(self.matrix.metrics["attention_score"], 0.0)
        self.assertGreater(self.matrix.metrics["memory_retention"], 0.0)
        self.assertGreater(self.matrix.metrics["consciousness_level"], 0.0)
        self.assertGreater(self.matrix.metrics["integration_score"], 0.0)
        self.assertGreater(self.matrix.metrics["ethical_score"], 0.0)
        self.assertGreater(self.matrix.metrics["governance_score"], 0.0)
        self.assertGreater(self.matrix.metrics["processing_time"], 0)
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test invalid quantum dimension
        invalid_inputs = self.test_inputs.copy()
        invalid_inputs["quantum"] = np.random.rand(256)
        with self.assertRaises(ModelError):
            self.matrix.process_consciousness(invalid_inputs)
        
        # Test invalid holographic dimension
        invalid_inputs = self.test_inputs.copy()
        invalid_inputs["holographic"] = np.random.rand(8192)
        with self.assertRaises(ModelError):
            self.matrix.process_consciousness(invalid_inputs)
        
        # Test invalid neural dimension
        invalid_inputs = self.test_inputs.copy()
        invalid_inputs["neural"] = np.random.rand(8192)
        with self.assertRaises(ModelError):
            self.matrix.process_consciousness(invalid_inputs)
    
    def test_reset_functionality(self):
        """Test system reset functionality."""
        # Process test inputs
        self.matrix.process_consciousness(self.test_inputs)
        
        # Reset system
        self.matrix.reset()
        
        # Check state reset
        self.assertIsNone(self.matrix.state["quantum_state"])
        self.assertIsNone(self.matrix.state["holographic_state"])
        self.assertIsNone(self.matrix.state["neural_state"])
        self.assertIsNone(self.matrix.state["attention_state"])
        self.assertIsNone(self.matrix.state["memory_state"])
        self.assertEqual(self.matrix.state["consciousness_level"], 0.0)
        self.assertIsNone(self.matrix.state["ethical_state"])
        self.assertIsNone(self.matrix.state["governance_state"])
        self.assertIsNone(self.matrix.state["metrics"])
        
        # Check metrics reset
        self.assertEqual(self.matrix.metrics["attention_score"], 0.0)
        self.assertEqual(self.matrix.metrics["memory_retention"], 0.0)
        self.assertEqual(self.matrix.metrics["consciousness_level"], 0.0)
        self.assertEqual(self.matrix.metrics["integration_score"], 0.0)
        self.assertEqual(self.matrix.metrics["ethical_score"], 0.0)
        self.assertEqual(self.matrix.metrics["governance_score"], 0.0)
        self.assertEqual(self.matrix.metrics["processing_time"], 0.0)
    
    def test_consciousness_quality(self):
        """Test consciousness processing quality with different inputs."""
        # Test multiple random inputs
        for _ in range(10):
            test_inputs = {
                "quantum": np.random.rand(512),
                "holographic": np.random.rand(16384),
                "neural": np.random.rand(16384)
            }
            
            results = self.matrix.process_consciousness(test_inputs)
            
            # Check consciousness level consistency
            self.assertGreaterEqual(results["consciousness_level"], 0.0)
            self.assertLessEqual(results["consciousness_level"], 1.0)
            
            # Check attention distribution
            self.assertTrue(np.allclose(np.sum(results["attention_scores"]), 1.0))
            
            # Check memory retention
            self.assertGreaterEqual(results["memory_retention"], 0.0)
            self.assertLessEqual(results["memory_retention"], 1.0)
            
            # Check integration score
            self.assertGreaterEqual(results["integration_score"], 0.0)
            self.assertLessEqual(results["integration_score"], 1.0)
            
            # Check ethical score
            self.assertGreaterEqual(results["ethical_score"], 0.0)
            self.assertLessEqual(results["ethical_score"], 1.0)
            
            # Check governance score
            self.assertGreaterEqual(results["governance_score"], 0.0)
            self.assertLessEqual(results["governance_score"], 1.0)

if __name__ == '__main__':
    unittest.main() 
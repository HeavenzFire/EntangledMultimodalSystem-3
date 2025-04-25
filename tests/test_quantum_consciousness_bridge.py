import unittest
import numpy as np
from src.core.quantum_consciousness_bridge import QuantumConsciousnessBridge

class TestQuantumConsciousnessBridge(unittest.TestCase):
    """Test suite for QuantumConsciousnessBridge."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "quantum_dim": 1024,
            "consciousness_dim": 16384,
            "environmental_dim": 8192,
            "projection_depth": 8,
            "entanglement_strength": 0.95,
            "temporal_layers": 4,
            "tunneling_depth": 3,
            "superposition_layers": 4,
            "integration_depth": 6
        }
        
        self.bridge = QuantumConsciousnessBridge(self.config)
        
        # Generate test data
        self.test_inputs = {
            "quantum": np.random.rand(self.config["quantum_dim"]),
            "consciousness": np.random.rand(self.config["consciousness_dim"]),
            "environmental": np.random.rand(self.config["environmental_dim"])
        }
    
    def test_initialization(self):
        """Test initialization of QuantumConsciousnessBridge."""
        self.assertEqual(self.bridge.quantum_dim, self.config["quantum_dim"])
        self.assertEqual(self.bridge.consciousness_dim, self.config["consciousness_dim"])
        self.assertEqual(self.bridge.environmental_dim, self.config["environmental_dim"])
        self.assertEqual(self.bridge.projection_depth, self.config["projection_depth"])
        self.assertEqual(self.bridge.entanglement_strength, self.config["entanglement_strength"])
        self.assertEqual(self.bridge.temporal_layers, self.config["temporal_layers"])
        self.assertEqual(self.bridge.tunneling_depth, self.config["tunneling_depth"])
        self.assertEqual(self.bridge.superposition_layers, self.config["superposition_layers"])
        self.assertEqual(self.bridge.integration_depth, self.config["integration_depth"])
    
    def test_state_processing(self):
        """Test processing of quantum, consciousness, and environmental states."""
        results = self.bridge.process_states(self.test_inputs)
        
        # Check quantum state
        self.assertIsNotNone(results["quantum_state"])
        self.assertIn("state", results["quantum_state"])
        self.assertIn("coherence", results["quantum_state"])
        self.assertEqual(results["quantum_state"]["state"].shape, (self.config["quantum_dim"],))
        self.assertGreaterEqual(results["quantum_state"]["coherence"], 0.0)
        self.assertLessEqual(results["quantum_state"]["coherence"], 1.0)
        
        # Check consciousness state
        self.assertIsNotNone(results["consciousness_state"])
        
        # Check environmental state
        self.assertIsNotNone(results["environmental_state"])
        
        # Check projection state
        self.assertIsNotNone(results["projection_state"])
        self.assertIn("quality", results["projection_state"])
        self.assertIn("depth", results["projection_state"])
        self.assertEqual(results["projection_state"]["depth"], self.config["projection_depth"])
        self.assertGreaterEqual(results["projection_state"]["quality"], 0.0)
        self.assertLessEqual(results["projection_state"]["quality"], 1.0)
        
        # Check entanglement state
        self.assertIsNotNone(results["entanglement_state"])
        self.assertIn("quality", results["entanglement_state"])
        self.assertIn("strength", results["entanglement_state"])
        self.assertEqual(results["entanglement_state"]["strength"], self.config["entanglement_strength"])
        self.assertGreaterEqual(results["entanglement_state"]["quality"], 0.0)
        self.assertLessEqual(results["entanglement_state"]["quality"], 1.0)
        
        # Check temporal state
        self.assertIsNotNone(results["temporal_state"])
        self.assertIn("coherence", results["temporal_state"])
        self.assertIn("layers", results["temporal_state"])
        self.assertEqual(results["temporal_state"]["layers"], self.config["temporal_layers"])
        self.assertGreaterEqual(results["temporal_state"]["coherence"], 0.0)
        self.assertLessEqual(results["temporal_state"]["coherence"], 1.0)
        
        # Check tunneling state
        self.assertIsNotNone(results["tunneling_state"])
        self.assertIn("quality", results["tunneling_state"])
        self.assertIn("depth", results["tunneling_state"])
        self.assertEqual(results["tunneling_state"]["depth"], self.config["tunneling_depth"])
        self.assertGreaterEqual(results["tunneling_state"]["quality"], 0.0)
        self.assertLessEqual(results["tunneling_state"]["quality"], 1.0)
        
        # Check superposition state
        self.assertIsNotNone(results["superposition_state"])
        self.assertIn("quality", results["superposition_state"])
        self.assertIn("layers", results["superposition_state"])
        self.assertEqual(results["superposition_state"]["layers"], self.config["superposition_layers"])
        self.assertGreaterEqual(results["superposition_state"]["quality"], 0.0)
        self.assertLessEqual(results["superposition_state"]["quality"], 1.0)
        
        # Check integration state
        self.assertIsNotNone(results["integration_state"])
        self.assertIn("quality", results["integration_state"])
        self.assertIn("depth", results["integration_state"])
        self.assertEqual(results["integration_state"]["depth"], self.config["integration_depth"])
        self.assertGreaterEqual(results["integration_state"]["quality"], 0.0)
        self.assertLessEqual(results["integration_state"]["quality"], 1.0)
        
        # Check metrics
        self.assertIsNotNone(results["metrics"])
        self.assertIn("quantum_coherence", results["metrics"])
        self.assertIn("consciousness_score", results["metrics"])
        self.assertIn("environmental_score", results["metrics"])
        self.assertIn("projection_quality", results["metrics"])
        self.assertIn("entanglement_quality", results["metrics"])
        self.assertIn("temporal_coherence", results["metrics"])
        self.assertIn("tunneling_quality", results["metrics"])
        self.assertIn("superposition_quality", results["metrics"])
        self.assertIn("integration_quality", results["metrics"])
        self.assertIn("processing_time", results["metrics"])
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test invalid quantum dimension
        invalid_inputs = self.test_inputs.copy()
        invalid_inputs["quantum"] = np.random.rand(self.config["quantum_dim"] + 1)
        with self.assertRaises(Exception):
            self.bridge.process_states(invalid_inputs)
        
        # Test invalid consciousness dimension
        invalid_inputs = self.test_inputs.copy()
        invalid_inputs["consciousness"] = np.random.rand(self.config["consciousness_dim"] + 1)
        with self.assertRaises(Exception):
            self.bridge.process_states(invalid_inputs)
        
        # Test invalid environmental dimension
        invalid_inputs = self.test_inputs.copy()
        invalid_inputs["environmental"] = np.random.rand(self.config["environmental_dim"] + 1)
        with self.assertRaises(Exception):
            self.bridge.process_states(invalid_inputs)
        
        # Test missing quantum input
        invalid_inputs = self.test_inputs.copy()
        del invalid_inputs["quantum"]
        with self.assertRaises(Exception):
            self.bridge.process_states(invalid_inputs)
        
        # Test missing consciousness input
        invalid_inputs = self.test_inputs.copy()
        del invalid_inputs["consciousness"]
        with self.assertRaises(Exception):
            self.bridge.process_states(invalid_inputs)
        
        # Test missing environmental input
        invalid_inputs = self.test_inputs.copy()
        del invalid_inputs["environmental"]
        with self.assertRaises(Exception):
            self.bridge.process_states(invalid_inputs)
    
    def test_reset_functionality(self):
        """Test reset functionality."""
        # Process states
        self.bridge.process_states(self.test_inputs)
        
        # Reset bridge
        self.bridge.reset()
        
        # Check state
        state = self.bridge.get_state()
        self.assertIsNone(state["quantum_state"])
        self.assertIsNone(state["consciousness_state"])
        self.assertIsNone(state["environmental_state"])
        self.assertIsNone(state["projection_state"])
        self.assertIsNone(state["entanglement_state"])
        self.assertIsNone(state["temporal_state"])
        self.assertIsNone(state["tunneling_state"])
        self.assertIsNone(state["superposition_state"])
        self.assertIsNone(state["integration_state"])
        self.assertIsNone(state["metrics"])
        
        # Check metrics
        metrics = self.bridge.get_metrics()
        self.assertEqual(metrics["quantum_coherence"], 0.0)
        self.assertEqual(metrics["consciousness_score"], 0.0)
        self.assertEqual(metrics["environmental_score"], 0.0)
        self.assertEqual(metrics["projection_quality"], 0.0)
        self.assertEqual(metrics["entanglement_quality"], 0.0)
        self.assertEqual(metrics["temporal_coherence"], 0.0)
        self.assertEqual(metrics["tunneling_quality"], 0.0)
        self.assertEqual(metrics["superposition_quality"], 0.0)
        self.assertEqual(metrics["integration_quality"], 0.0)
        self.assertEqual(metrics["processing_time"], 0.0)
    
    def test_state_quality(self):
        """Test quality of state processing with multiple random inputs."""
        for _ in range(10):
            # Generate random inputs
            inputs = {
                "quantum": np.random.rand(self.config["quantum_dim"]),
                "consciousness": np.random.rand(self.config["consciousness_dim"]),
                "environmental": np.random.rand(self.config["environmental_dim"])
            }
            
            # Process states
            results = self.bridge.process_states(inputs)
            
            # Check quantum coherence
            self.assertGreaterEqual(results["quantum_state"]["coherence"], 0.0)
            self.assertLessEqual(results["quantum_state"]["coherence"], 1.0)
            
            # Check projection quality
            self.assertGreaterEqual(results["projection_state"]["quality"], 0.0)
            self.assertLessEqual(results["projection_state"]["quality"], 1.0)
            
            # Check entanglement quality
            self.assertGreaterEqual(results["entanglement_state"]["quality"], 0.0)
            self.assertLessEqual(results["entanglement_state"]["quality"], 1.0)
            
            # Check temporal coherence
            self.assertGreaterEqual(results["temporal_state"]["coherence"], 0.0)
            self.assertLessEqual(results["temporal_state"]["coherence"], 1.0)
            
            # Check tunneling quality
            self.assertGreaterEqual(results["tunneling_state"]["quality"], 0.0)
            self.assertLessEqual(results["tunneling_state"]["quality"], 1.0)
            
            # Check superposition quality
            self.assertGreaterEqual(results["superposition_state"]["quality"], 0.0)
            self.assertLessEqual(results["superposition_state"]["quality"], 1.0)
            
            # Check integration quality
            self.assertGreaterEqual(results["integration_state"]["quality"], 0.0)
            self.assertLessEqual(results["integration_state"]["quality"], 1.0)

if __name__ == "__main__":
    unittest.main() 
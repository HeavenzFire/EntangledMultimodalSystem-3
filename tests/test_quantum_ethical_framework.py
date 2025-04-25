import unittest
import numpy as np
from src.core.quantum_ethical_framework import QuantumEthicalFramework
from src.utils.errors import ModelError

class TestQuantumEthicalFramework(unittest.TestCase):
    """Test cases for the Quantum-Ethical Framework."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "n_qubits": 8,
            "principle_depth": 3,
            "entanglement_strength": 0.5
        }
        self.framework = QuantumEthicalFramework(self.config)
        
        # Create test context
        self.context = np.random.rand(self.config["n_qubits"])
    
    def test_initialization(self):
        """Test framework initialization."""
        self.assertEqual(self.framework.n_qubits, self.config["n_qubits"])
        self.assertEqual(self.framework.principle_depth, self.config["principle_depth"])
        self.assertEqual(self.framework.entanglement_strength, self.config["entanglement_strength"])
        
        # Test principles initialization
        self.assertIn("benefit_humanity", self.framework.principles)
        self.assertIn("safety", self.framework.principles)
        self.assertIn("privacy", self.framework.principles)
        self.assertIn("transparency", self.framework.principles)
        self.assertIn("accountability", self.framework.principles)
        self.assertIn("fairness", self.framework.principles)
        self.assertIn("human_control", self.framework.principles)
        
        # Test state initialization
        self.assertIsNone(self.framework.state["quantum_state"])
        self.assertIsNone(self.framework.state["ethical_weights"])
        self.assertIsNone(self.framework.state["decision_context"])
        
        # Test metrics initialization
        self.assertEqual(self.framework.metrics["ethical_coherence"], 0.0)
        self.assertEqual(self.framework.metrics["principle_alignment"], 0.0)
        self.assertEqual(self.framework.metrics["decision_confidence"], 0.0)
        self.assertEqual(self.framework.metrics["processing_time"], 0.0)
    
    def test_principle_encoding(self):
        """Test ethical principle encoding."""
        result = self.framework.encode_ethical_principles(self.context)
        
        # Test result structure
        self.assertIn("quantum_state", result)
        self.assertIn("ethical_encoding", result)
        self.assertIn("decision", result)
        self.assertIn("metrics", result)
        self.assertIn("state", result)
        
        # Test quantum state properties
        quantum_state = result["quantum_state"]
        self.assertEqual(len(quantum_state), self.config["n_qubits"] * 2)
        self.assertTrue(np.all(-1 <= quantum_state))
        self.assertTrue(np.all(quantum_state <= 1))
        
        # Test ethical encoding properties
        ethical_encoding = result["ethical_encoding"]
        self.assertEqual(len(ethical_encoding), len(self.framework.principles))
        self.assertTrue(np.all(0 <= ethical_encoding))
        self.assertTrue(np.all(ethical_encoding <= 1))
    
    def test_analysis_results(self):
        """Test analysis results."""
        result = self.framework.encode_ethical_principles(self.context)
        
        # Test decision properties
        self.assertTrue(0 <= result["decision"] <= 1)
        
        # Test metrics
        metrics = result["metrics"]
        self.assertTrue(0 <= metrics["ethical_coherence"] <= 1)
        self.assertTrue(0 <= metrics["principle_alignment"] <= 1)
        self.assertTrue(0 <= metrics["decision_confidence"] <= 1)
        self.assertEqual(metrics["processing_time"], 0.25)  # 25% faster
    
    def test_state_management(self):
        """Test state management."""
        # Test initial state
        initial_state = self.framework.get_state()
        self.assertIsNone(initial_state["quantum_state"])
        self.assertIsNone(initial_state["ethical_weights"])
        self.assertIsNone(initial_state["decision_context"])
        
        # Process context and test updated state
        self.framework.encode_ethical_principles(self.context)
        updated_state = self.framework.get_state()
        self.assertIsNotNone(updated_state["quantum_state"])
        self.assertIsNotNone(updated_state["ethical_weights"])
        self.assertIsNotNone(updated_state["decision_context"])
        
        # Test reset
        self.framework.reset()
        reset_state = self.framework.get_state()
        self.assertIsNone(reset_state["quantum_state"])
        self.assertIsNone(reset_state["ethical_weights"])
        self.assertIsNone(reset_state["decision_context"])
    
    def test_error_handling(self):
        """Test error handling."""
        # Test invalid context dimensions
        invalid_context = np.random.rand(self.config["n_qubits"] + 1)
        with self.assertRaises(ModelError):
            self.framework.encode_ethical_principles(invalid_context)
    
    def test_ethical_analysis(self):
        """Test ethical analysis quality."""
        # Test with different contexts
        for _ in range(3):
            # Generate new test context
            context = np.random.rand(self.config["n_qubits"])
            
            result = self.framework.encode_ethical_principles(context)
            
            # Test quantum state
            quantum_state = result["quantum_state"]
            self.assertEqual(len(quantum_state), self.config["n_qubits"] * 2)
            self.assertTrue(np.all(-1 <= quantum_state))
            self.assertTrue(np.all(quantum_state <= 1))
            
            # Test ethical encoding
            ethical_encoding = result["ethical_encoding"]
            self.assertEqual(len(ethical_encoding), len(self.framework.principles))
            self.assertTrue(np.all(0 <= ethical_encoding))
            self.assertTrue(np.all(ethical_encoding <= 1))
            
            # Test decision
            self.assertTrue(0 <= result["decision"] <= 1)

if __name__ == '__main__':
    unittest.main() 
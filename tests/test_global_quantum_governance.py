import unittest
import numpy as np
from src.core.global_quantum_governance import GlobalQuantumGovernance
from src.utils.errors import ModelError

class TestGlobalQuantumGovernance(unittest.TestCase):
    """Test cases for the Global Quantum Governance Framework."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "n_qubits": 512,
            "entanglement_strength": 0.999,
            "quantum_fidelity": 0.99999,
            "ethical_threshold": 0.999,
            "neural_phi_threshold": 0.9,
            "gaia_threshold": 0.95
        }
        self.governance = GlobalQuantumGovernance(self.config)
        
        # Create test data
        self.action_context = np.random.rand(self.config["n_qubits"])
        self.neural_pattern = np.random.rand(1024)
        self.planetary_data = np.random.rand(37)
    
    def test_initialization(self):
        """Test framework initialization."""
        self.assertEqual(self.governance.n_qubits, self.config["n_qubits"])
        self.assertEqual(self.governance.entanglement_strength, self.config["entanglement_strength"])
        self.assertEqual(self.governance.quantum_fidelity, self.config["quantum_fidelity"])
        self.assertEqual(self.governance.ethical_threshold, self.config["ethical_threshold"])
        self.assertEqual(self.governance.neural_phi_threshold, self.config["neural_phi_threshold"])
        self.assertEqual(self.governance.gaia_threshold, self.config["gaia_threshold"])
        
        # Test state initialization
        self.assertIsNone(self.governance.state["quantum_state"])
        self.assertIsNone(self.governance.state["neural_pattern"])
        self.assertIsNone(self.governance.state["planetary_state"])
        self.assertIsNone(self.governance.state["governance_metrics"])
        
        # Test metrics initialization
        self.assertEqual(self.governance.metrics["quantum_entanglement"], 0.0)
        self.assertEqual(self.governance.metrics["ethical_score"], 0.0)
        self.assertEqual(self.governance.metrics["neural_phi"], 0.0)
        self.assertEqual(self.governance.metrics["gaia_integration"], 0.0)
        self.assertEqual(self.governance.metrics["quantum_brute_force_resistance"], 0.0)
    
    def test_action_validation(self):
        """Test action validation."""
        result = self.governance.validate_action(
            self.action_context,
            self.neural_pattern,
            self.planetary_data
        )
        
        # Test result structure
        self.assertIn("quantum_state", result)
        self.assertIn("ethical_score", result)
        self.assertIn("neural_phi", result)
        self.assertIn("gaia_integration", result)
        self.assertIn("metrics", result)
        self.assertIn("state", result)
        
        # Test quantum state properties
        quantum_state = result["quantum_state"]
        self.assertEqual(len(quantum_state), self.config["n_qubits"] * 2)
        self.assertTrue(np.all(-1 <= quantum_state))
        self.assertTrue(np.all(quantum_state <= 1))
        
        # Test validation scores
        self.assertTrue(0 <= result["ethical_score"] <= 1)
        self.assertTrue(0 <= result["neural_phi"] <= 1)
        self.assertTrue(0 <= result["gaia_integration"] <= 1)
    
    def test_metrics_calculation(self):
        """Test metrics calculation."""
        result = self.governance.validate_action(
            self.action_context,
            self.neural_pattern,
            self.planetary_data
        )
        metrics = result["metrics"]
        
        # Test metric properties
        self.assertIn("quantum_entanglement", metrics)
        self.assertIn("ethical_score", metrics)
        self.assertIn("neural_phi", metrics)
        self.assertIn("gaia_integration", metrics)
        self.assertIn("quantum_brute_force_resistance", metrics)
        
        # Test metric ranges
        self.assertTrue(0 <= metrics["quantum_entanglement"] <= 1)
        self.assertTrue(0 <= metrics["ethical_score"] <= 1)
        self.assertTrue(0 <= metrics["neural_phi"] <= 1)
        self.assertTrue(0 <= metrics["gaia_integration"] <= 1)
        self.assertGreater(metrics["quantum_brute_force_resistance"], 1e158)
    
    def test_state_management(self):
        """Test state management."""
        # Test initial state
        initial_state = self.governance.get_state()
        self.assertIsNone(initial_state["quantum_state"])
        self.assertIsNone(initial_state["neural_pattern"])
        self.assertIsNone(initial_state["planetary_state"])
        self.assertIsNone(initial_state["governance_metrics"])
        
        # Process action and test updated state
        self.governance.validate_action(
            self.action_context,
            self.neural_pattern,
            self.planetary_data
        )
        updated_state = self.governance.get_state()
        self.assertIsNotNone(updated_state["quantum_state"])
        self.assertIsNotNone(updated_state["neural_pattern"])
        self.assertIsNotNone(updated_state["planetary_state"])
        self.assertIsNotNone(updated_state["governance_metrics"])
        
        # Test reset
        self.governance.reset()
        reset_state = self.governance.get_state()
        self.assertIsNone(reset_state["quantum_state"])
        self.assertIsNone(reset_state["neural_pattern"])
        self.assertIsNone(reset_state["planetary_state"])
        self.assertIsNone(reset_state["governance_metrics"])
    
    def test_error_handling(self):
        """Test error handling."""
        # Test invalid action context
        invalid_context = np.random.rand(self.config["n_qubits"] + 1)
        with self.assertRaises(ModelError):
            self.governance.validate_action(
                invalid_context,
                self.neural_pattern,
                self.planetary_data
            )
        
        # Test invalid neural pattern
        invalid_pattern = np.random.rand(1025)
        with self.assertRaises(ModelError):
            self.governance.validate_action(
                self.action_context,
                invalid_pattern,
                self.planetary_data
            )
        
        # Test invalid planetary data
        invalid_planetary = np.random.rand(38)
        with self.assertRaises(ModelError):
            self.governance.validate_action(
                self.action_context,
                self.neural_pattern,
                invalid_planetary
            )
    
    def test_governance_quality(self):
        """Test governance quality."""
        # Test with different data combinations
        for _ in range(3):
            # Generate new test data
            context = np.random.rand(self.config["n_qubits"])
            pattern = np.random.rand(1024)
            planetary = np.random.rand(37)
            
            result = self.governance.validate_action(
                context, pattern, planetary
            )
            
            # Test quantum state
            quantum_state = result["quantum_state"]
            self.assertEqual(len(quantum_state), self.config["n_qubits"] * 2)
            self.assertTrue(np.all(-1 <= quantum_state))
            self.assertTrue(np.all(quantum_state <= 1))
            
            # Test validation scores
            self.assertTrue(0 <= result["ethical_score"] <= 1)
            self.assertTrue(0 <= result["neural_phi"] <= 1)
            self.assertTrue(0 <= result["gaia_integration"] <= 1)
            
            # Test metrics
            metrics = result["metrics"]
            self.assertTrue(0 <= metrics["quantum_entanglement"] <= 1)
            self.assertTrue(0 <= metrics["ethical_score"] <= 1)
            self.assertTrue(0 <= metrics["neural_phi"] <= 1)
            self.assertTrue(0 <= metrics["gaia_integration"] <= 1)
            self.assertGreater(metrics["quantum_brute_force_resistance"], 1e158)

if __name__ == '__main__':
    unittest.main() 
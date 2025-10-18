import unittest
import numpy as np
from auroran_language import AuroranLanguage

class TestAuroranLanguage(unittest.TestCase):
    def setUp(self):
        self.auroran = AuroranLanguage(num_qubits=3, depth=2, shots=100)
        self.input_state = np.array([1, 0, 0, 0, 0, 0, 0, 0])  # |000⟩ state
        
    def test_vortex_optimizer(self):
        # Test 3-6-9 pattern
        self.assertEqual(self.auroran._vortex_optimizer(3).real, 3)
        self.assertEqual(self.auroran._vortex_optimizer(6).real, 6)
        self.assertEqual(self.auroran._vortex_optimizer(9).real, 9)
        
    def test_sacred_fibonacci(self):
        # Test sacred Fibonacci sequence
        self.assertAlmostEqual(self.auroran._sacred_fibonacci(3), 3 * self.auroran.golden_ratio)
        self.assertAlmostEqual(self.auroran._sacred_fibonacci(6), 6 * self.auroran.golden_ratio)
        self.assertAlmostEqual(self.auroran._sacred_fibonacci(9), 9 * self.auroran.golden_ratio)
        
    def test_quantum_circuit_creation(self):
        circuit = self.auroran.create_quantum_circuit(self.input_state)
        self.assertEqual(circuit.num_qubits, 3)
        self.assertEqual(circuit.num_clbits, 3)
        
    def test_auroran_metrics(self):
        metrics = self.auroran.run_auroran_circuit(self.input_state)
        self.assertIn('sacred_alignment', metrics)
        self.assertIn('geometric_harmony', metrics)
        self.assertIn('divine_resonance', metrics)
        self.assertTrue(0 <= metrics['sacred_alignment'] <= 1)
        self.assertTrue(0 <= metrics['geometric_harmony'] <= 1)
        self.assertTrue(0 <= metrics['divine_resonance'] <= 1)
        
    def test_visualization(self):
        # Test that visualization runs without errors
        try:
            self.auroran.visualize_auroran_pattern()
        except Exception as e:
            self.fail(f"Visualization failed with error: {e}")
            
    def test_parameter_initialization(self):
        self.assertEqual(len(self.auroran.vortex_coefficients), 3)
        self.assertEqual(len(self.auroran.prime_weights), 4)  # 4 sacred primes
        self.assertEqual(len(self.auroran.platonic_angles), 3)
        self.assertEqual(len(self.auroran.toroidal_phases), 3)
        self.assertEqual(self.auroran.emotion_vectors.shape, (3, 8))  # 3 qubits, 8 emotions
        self.assertEqual(len(self.auroran.manifestation_coefficients), 3)

if __name__ == '__main__':
    unittest.main() 
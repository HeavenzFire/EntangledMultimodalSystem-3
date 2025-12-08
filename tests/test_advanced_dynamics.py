import unittest
import numpy as np
import torch
from core.mathematics.advanced_dynamics import AdvancedDynamics

class TestAdvancedDynamics(unittest.TestCase):
    def setUp(self):
        self.dynamics = AdvancedDynamics(dimension=2)
        
    def test_mandelbrot_set(self):
        """Test Mandelbrot set generation with quantum enhancement"""
        result = self.dynamics.mandelbrot_set(resolution=100, max_iter=10)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (100, 100))
        self.assertTrue(np.any(result))  # Check if any points are in the set
        
    def test_julia_set(self):
        """Test Julia set generation with quantum enhancement"""
        c = -0.7 + 0.27j
        result = self.dynamics.julia_set(c, resolution=100, max_iter=10)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (100, 100))
        self.assertTrue(np.any(result))
        
    def test_hulse_equation(self):
        """Test Hulse-Taylor equations with quantum corrections"""
        t = np.linspace(0, 10, 100)
        y = np.array([1.0, 0.0, 0.0, 1.0])
        params = {
            'm1': 1.4,
            'm2': 1.4,
            'a': 1.0,
            'e': 0.6
        }
        result = self.dynamics.hulse_equation(t, y, params)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (4,))
        
    def test_nonlinear_system(self):
        """Test nonlinear system with quantum enhancements"""
        t = np.linspace(0, 10, 100)
        y = np.array([0.5, 0.5])
        params = {
            'alpha_0': 1.0,
            'alpha_1': 1.0
        }
        result = self.dynamics.nonlinear_system(t, y, params)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (2,))
        
    def test_quantum_fractal(self):
        """Test quantum-enhanced fractal generation"""
        result = self.dynamics.quantum_fractal(resolution=100, max_iter=10)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (100, 100))
        self.assertTrue(np.any(result))
        
    def test_analyze_pattern(self):
        """Test pattern analysis with quantum properties"""
        data = np.random.rand(100, 100)
        result = self.dynamics.analyze_pattern(data)
        self.assertIsInstance(result, dict)
        self.assertIn('quantum_entropy', result)
        self.assertIn('quantum_correlation', result)
        self.assertIn('fractal_dimension', result)
        
    def test_visualize(self):
        """Test visualization of patterns"""
        data = np.random.rand(100, 100)
        # This test just checks if the visualization runs without errors
        try:
            self.dynamics.visualize(data)
            visualization_successful = True
        except Exception:
            visualization_successful = False
        self.assertTrue(visualization_successful)
        
    def test_quantum_optimization(self):
        """Test quantum-enhanced optimization"""
        def objective_function(state):
            return torch.sum(state**2)
            
        initial_state = torch.tensor([1.0, 0.0], dtype=torch.float64)
        result = self.dynamics.quantum_optimization(
            objective_function,
            initial_state,
            num_iterations=10
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('optimized_state', result)
        self.assertIn('loss_history', result)
        self.assertIn('final_loss', result)
        self.assertLess(result['final_loss'], 1.0)
        
    def test_quantum_entanglement(self):
        """Test quantum entanglement creation"""
        state1 = torch.tensor([1.0, 0.0], dtype=torch.complex64)
        state2 = torch.tensor([0.0, 1.0], dtype=torch.complex64)
        
        entangled = self.dynamics.quantum_entanglement(state1, state2)
        
        self.assertIsInstance(entangled, torch.Tensor)
        self.assertEqual(entangled.shape, (4,))
        self.assertTrue(torch.allclose(
            torch.abs(entangled),
            torch.tensor([0.0, 1.0/np.sqrt(2), 1.0/np.sqrt(2), 0.0])
        ))
        
    def test_quantum_pattern_recognition(self):
        """Test quantum-enhanced pattern recognition"""
        pattern = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.complex64)
        template = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex64)
        
        result = self.dynamics.quantum_pattern_recognition(pattern, template)
        
        self.assertIsInstance(result, dict)
        self.assertIn('similarity', result)
        self.assertIn('quantum_features', result)
        self.assertIn('purity', result['quantum_features'])
        self.assertIn('coherence', result['quantum_features'])
        self.assertIn('entanglement', result['quantum_features'])
        
    def test_visualize_3d(self):
        """Test 3D visualization"""
        data = np.random.rand(10, 10)
        # This test just checks if the visualization runs without errors
        try:
            self.dynamics.visualize_3d(data)
            visualization_successful = True
        except Exception:
            visualization_successful = False
        self.assertTrue(visualization_successful)
        
    def test_visualize_quantum_state(self):
        """Test quantum state visualization"""
        state = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        # This test just checks if the visualization runs without errors
        try:
            self.dynamics.visualize_quantum_state(state)
            visualization_successful = True
        except Exception:
            visualization_successful = False
        self.assertTrue(visualization_successful)
        
    def test_quantum_gates(self):
        """Test quantum gate application"""
        state = torch.tensor([1.0, 0.0], dtype=torch.complex64)
        transformed = self.dynamics._apply_quantum_gates(state)
        
        self.assertIsInstance(transformed, torch.Tensor)
        self.assertEqual(transformed.shape, state.shape)
        
    def test_entanglement_gates(self):
        """Test entanglement gate application"""
        state = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.complex64)
        transformed = self.dynamics._apply_entanglement_gates(state)
        
        self.assertIsInstance(transformed, torch.Tensor)
        self.assertEqual(transformed.shape, state.shape)
        
    def test_quantum_features(self):
        """Test quantum feature extraction"""
        state = torch.tensor([1.0, 0.0], dtype=torch.complex64)
        features = self.dynamics._extract_quantum_features(state)
        
        self.assertIsInstance(features, dict)
        self.assertIn('purity', features)
        self.assertIn('coherence', features)
        self.assertIn('entanglement', features)
        self.assertAlmostEqual(features['purity'], 1.0)
        
if __name__ == '__main__':
    unittest.main() 
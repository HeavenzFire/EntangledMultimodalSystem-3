import unittest
import numpy as np
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
        
if __name__ == '__main__':
    unittest.main() 
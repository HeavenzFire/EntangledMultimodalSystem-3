import unittest
import numpy as np
from src.core.fractal_intelligence_engine import FractalIntelligenceEngine
from src.utils.errors import ModelError

class TestFractalIntelligenceEngine(unittest.TestCase):
    """Test cases for the Fractal Intelligence Engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "max_iterations": 50,
            "resolution": 256,
            "escape_radius": 2.0,
            "semantic_dimensions": 64,
            "embedding_depth": 2
        }
        self.engine = FractalIntelligenceEngine(self.config)
        
        # Create test fractal
        self.test_fractal = np.random.rand(
            self.config["resolution"],
            self.config["resolution"],
            3
        )
    
    def test_initialization(self):
        """Test engine initialization."""
        self.assertEqual(self.engine.max_iterations, self.config["max_iterations"])
        self.assertEqual(self.engine.resolution, self.config["resolution"])
        self.assertEqual(self.engine.escape_radius, self.config["escape_radius"])
        self.assertEqual(self.engine.semantic_dimensions, self.config["semantic_dimensions"])
        self.assertEqual(self.engine.embedding_depth, self.config["embedding_depth"])
        
        # Test state initialization
        self.assertIsNone(self.engine.state["fractal_data"])
        self.assertIsNone(self.engine.state["semantic_embedding"])
        self.assertIsNone(self.engine.state["analysis_results"])
        self.assertIsNone(self.engine.state["generation_metrics"])
        
        # Test metrics initialization
        self.assertEqual(self.engine.metrics["fractal_complexity"], 0.0)
        self.assertEqual(self.engine.metrics["semantic_coherence"], 0.0)
        self.assertEqual(self.engine.metrics["pattern_richness"], 0.0)
        self.assertEqual(self.engine.metrics["generation_quality"], 0.0)
    
    def test_mandelbrot_generation(self):
        """Test Mandelbrot set generation."""
        # Test with default parameters
        fractal = self.engine.generate_mandelbrot((0, 0))
        self.assertEqual(fractal.shape, (self.config["resolution"], self.config["resolution"], 3))
        self.assertTrue(np.all(0 <= fractal) and np.all(fractal <= 1))
        
        # Test with custom parameters
        fractal = self.engine.generate_mandelbrot((0.5, 0.5), zoom=2.0)
        self.assertEqual(fractal.shape, (self.config["resolution"], self.config["resolution"], 3))
        self.assertTrue(np.all(0 <= fractal) and np.all(fractal <= 1))
    
    def test_julia_generation(self):
        """Test Julia set generation."""
        # Test with default parameters
        fractal = self.engine.generate_julia(complex(0.285, 0.01))
        self.assertEqual(fractal.shape, (self.config["resolution"], self.config["resolution"], 3))
        self.assertTrue(np.all(0 <= fractal) and np.all(fractal <= 1))
        
        # Test with custom parameters
        fractal = self.engine.generate_julia(
            complex(0.285, 0.01),
            center=(0.5, 0.5),
            zoom=2.0
        )
        self.assertEqual(fractal.shape, (self.config["resolution"], self.config["resolution"], 3))
        self.assertTrue(np.all(0 <= fractal) and np.all(fractal <= 1))
    
    def test_semantic_embedding(self):
        """Test semantic embedding."""
        result = self.engine.embed_semantic_meaning(self.test_fractal)
        
        # Test result structure
        self.assertIn("semantic_embedding", result)
        self.assertIn("analysis", result)
        self.assertIn("metrics", result)
        self.assertIn("state", result)
        
        # Test embedding properties
        self.assertEqual(len(result["semantic_embedding"]), self.config["semantic_dimensions"])
        self.assertTrue(np.all(-1 <= result["semantic_embedding"]))
        self.assertTrue(np.all(result["semantic_embedding"] <= 1))
    
    def test_analysis_results(self):
        """Test fractal analysis results."""
        result = self.engine.embed_semantic_meaning(self.test_fractal)
        analysis = result["analysis"]
        
        # Test analysis properties
        self.assertIn("pattern_density", analysis)
        self.assertIn("edge_complexity", analysis)
        self.assertIn("symmetry_score", analysis)
        
        # Test analysis ranges
        self.assertTrue(0 <= analysis["pattern_density"] <= 1)
        self.assertTrue(0 <= analysis["edge_complexity"] <= 1)
        self.assertTrue(0 <= analysis["symmetry_score"] <= 1)
    
    def test_metrics_calculation(self):
        """Test metrics calculation."""
        result = self.engine.embed_semantic_meaning(self.test_fractal)
        metrics = result["metrics"]
        
        # Test metric properties
        self.assertIn("fractal_complexity", metrics)
        self.assertIn("semantic_coherence", metrics)
        self.assertIn("pattern_richness", metrics)
        self.assertIn("generation_quality", metrics)
        
        # Test metric ranges
        self.assertTrue(0 <= metrics["fractal_complexity"] <= 1)
        self.assertTrue(0 <= metrics["semantic_coherence"] <= 1)
        self.assertTrue(0 <= metrics["pattern_richness"] <= 1)
        self.assertTrue(0 <= metrics["generation_quality"] <= 1)
    
    def test_state_management(self):
        """Test state management."""
        # Test initial state
        initial_state = self.engine.get_state()
        self.assertIsNone(initial_state["fractal_data"])
        self.assertIsNone(initial_state["semantic_embedding"])
        self.assertIsNone(initial_state["analysis_results"])
        self.assertIsNone(initial_state["generation_metrics"])
        
        # Process fractal and test updated state
        self.engine.embed_semantic_meaning(self.test_fractal)
        updated_state = self.engine.get_state()
        self.assertIsNotNone(updated_state["fractal_data"])
        self.assertIsNotNone(updated_state["semantic_embedding"])
        self.assertIsNotNone(updated_state["analysis_results"])
        self.assertIsNotNone(updated_state["generation_metrics"])
        
        # Test reset
        self.engine.reset()
        reset_state = self.engine.get_state()
        self.assertIsNone(reset_state["fractal_data"])
        self.assertIsNone(reset_state["semantic_embedding"])
        self.assertIsNone(reset_state["analysis_results"])
        self.assertIsNone(reset_state["generation_metrics"])
    
    def test_error_handling(self):
        """Test error handling."""
        # Test invalid fractal shape
        invalid_fractal = np.random.rand(
            self.config["resolution"] + 1,
            self.config["resolution"],
            3
        )
        with self.assertRaises(ModelError):
            self.engine.embed_semantic_meaning(invalid_fractal)
        
        # Test invalid fractal values
        invalid_fractal = np.random.rand(
            self.config["resolution"],
            self.config["resolution"],
            3
        ) * 2  # Values > 1
        with self.assertRaises(ModelError):
            self.engine.embed_semantic_meaning(invalid_fractal)
    
    def test_edge_complexity(self):
        """Test edge complexity calculation."""
        # Create test fractal with known edge complexity
        test_fractal = np.zeros((self.config["resolution"], self.config["resolution"], 3))
        test_fractal[50:100, 50:100] = 1.0  # Square in the middle
        
        complexity = self.engine._calculate_edge_complexity(test_fractal)
        self.assertTrue(0 <= complexity <= 1)
        self.assertGreater(complexity, 0)  # Should detect edges
    
    def test_symmetry_score(self):
        """Test symmetry score calculation."""
        # Create test fractal with known symmetry
        test_fractal = np.ones((self.config["resolution"], self.config["resolution"], 3))
        test_fractal[50:100, 50:100] = 0.0  # Square in the middle
        
        score = self.engine._calculate_symmetry_score(test_fractal)
        self.assertTrue(0 <= score <= 1)
        self.assertGreater(score, 0.5)  # Should be fairly symmetric

if __name__ == '__main__':
    unittest.main() 
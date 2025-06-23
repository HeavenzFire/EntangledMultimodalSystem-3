import unittest
import numpy as np
from src.core.radiation_aware_ai import RadiationAwareAI
from src.utils.errors import ModelError

class TestRadiationAwareAI(unittest.TestCase):
    """Test cases for the Radiation-Aware AI system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "geiger_dimensions": 128,
            "visual_dimensions": 1024,
            "thermal_dimensions": 512,
            "fusion_dimensions": 2048,
            "detection_threshold": 0.7,
            "analysis_depth": 5
        }
        self.ai = RadiationAwareAI(self.config)
        
        # Create test data
        self.geiger_data = np.random.rand(self.config["geiger_dimensions"])
        self.visual_data = np.random.rand(self.config["visual_dimensions"])
        self.thermal_data = np.random.rand(self.config["thermal_dimensions"])
    
    def test_initialization(self):
        """Test system initialization."""
        self.assertEqual(self.ai.geiger_dimensions, self.config["geiger_dimensions"])
        self.assertEqual(self.ai.visual_dimensions, self.config["visual_dimensions"])
        self.assertEqual(self.ai.thermal_dimensions, self.config["thermal_dimensions"])
        self.assertEqual(self.ai.fusion_dimensions, self.config["fusion_dimensions"])
        self.assertEqual(self.ai.detection_threshold, self.config["detection_threshold"])
        self.assertEqual(self.ai.analysis_depth, self.config["analysis_depth"])
        
        # Test state initialization
        self.assertIsNone(self.ai.state["geiger_data"])
        self.assertIsNone(self.ai.state["visual_data"])
        self.assertIsNone(self.ai.state["thermal_data"])
        self.assertIsNone(self.ai.state["fused_data"])
        self.assertIsNone(self.ai.state["analysis_results"])
        
        # Test metrics initialization
        self.assertEqual(self.ai.metrics["detection_accuracy"], 0.0)
        self.assertEqual(self.ai.metrics["processing_time"], 0.0)
        self.assertEqual(self.ai.metrics["radiation_level"], 0.0)
        self.assertEqual(self.ai.metrics["confidence_score"], 0.0)
    
    def test_data_processing(self):
        """Test radiation data processing."""
        result = self.ai.process_radiation_data(
            self.geiger_data,
            self.visual_data,
            self.thermal_data
        )
        
        # Test result structure
        self.assertIn("fused_data", result)
        self.assertIn("analysis", result)
        self.assertIn("metrics", result)
        self.assertIn("state", result)
        
        # Test fused data properties
        fused_data = result["fused_data"]
        self.assertEqual(len(fused_data), self.config["fusion_dimensions"])
        self.assertTrue(np.all(-1 <= fused_data))
        self.assertTrue(np.all(fused_data <= 1))
    
    def test_analysis_results(self):
        """Test analysis results."""
        result = self.ai.process_radiation_data(
            self.geiger_data,
            self.visual_data,
            self.thermal_data
        )
        analysis = result["analysis"]
        
        # Test analysis properties
        self.assertIn("detection", analysis)
        self.assertIn("radiation_level", analysis)
        self.assertIn("confidence_score", analysis)
        
        # Test analysis ranges
        self.assertTrue(0 <= analysis["detection"] <= 1)
        self.assertTrue(0 <= analysis["radiation_level"] <= 1)
        self.assertTrue(0 <= analysis["confidence_score"] <= 1)
    
    def test_metrics_calculation(self):
        """Test metrics calculation."""
        result = self.ai.process_radiation_data(
            self.geiger_data,
            self.visual_data,
            self.thermal_data
        )
        metrics = result["metrics"]
        
        # Test metric properties
        self.assertIn("detection_accuracy", metrics)
        self.assertIn("processing_time", metrics)
        self.assertIn("radiation_level", metrics)
        self.assertIn("confidence_score", metrics)
        
        # Test metric ranges
        self.assertTrue(0 <= metrics["detection_accuracy"] <= 1)
        self.assertEqual(metrics["processing_time"], 0.37)  # 37% faster
        self.assertTrue(0 <= metrics["radiation_level"] <= 1)
        self.assertTrue(0 <= metrics["confidence_score"] <= 1)
    
    def test_state_management(self):
        """Test state management."""
        # Test initial state
        initial_state = self.ai.get_state()
        self.assertIsNone(initial_state["geiger_data"])
        self.assertIsNone(initial_state["visual_data"])
        self.assertIsNone(initial_state["thermal_data"])
        self.assertIsNone(initial_state["fused_data"])
        self.assertIsNone(initial_state["analysis_results"])
        
        # Process data and test updated state
        self.ai.process_radiation_data(
            self.geiger_data,
            self.visual_data,
            self.thermal_data
        )
        updated_state = self.ai.get_state()
        self.assertIsNotNone(updated_state["geiger_data"])
        self.assertIsNotNone(updated_state["visual_data"])
        self.assertIsNotNone(updated_state["thermal_data"])
        self.assertIsNotNone(updated_state["fused_data"])
        self.assertIsNotNone(updated_state["analysis_results"])
        
        # Test reset
        self.ai.reset()
        reset_state = self.ai.get_state()
        self.assertIsNone(reset_state["geiger_data"])
        self.assertIsNone(reset_state["visual_data"])
        self.assertIsNone(reset_state["thermal_data"])
        self.assertIsNone(reset_state["fused_data"])
        self.assertIsNone(reset_state["analysis_results"])
    
    def test_error_handling(self):
        """Test error handling."""
        # Test invalid Geiger data
        invalid_geiger = np.random.rand(self.config["geiger_dimensions"] + 1)
        with self.assertRaises(ModelError):
            self.ai.process_radiation_data(
                invalid_geiger,
                self.visual_data,
                self.thermal_data
            )
        
        # Test invalid visual data
        invalid_visual = np.random.rand(self.config["visual_dimensions"] + 1)
        with self.assertRaises(ModelError):
            self.ai.process_radiation_data(
                self.geiger_data,
                invalid_visual,
                self.thermal_data
            )
        
        # Test invalid thermal data
        invalid_thermal = np.random.rand(self.config["thermal_dimensions"] + 1)
        with self.assertRaises(ModelError):
            self.ai.process_radiation_data(
                self.geiger_data,
                self.visual_data,
                invalid_thermal
            )
    
    def test_radiation_analysis(self):
        """Test radiation analysis quality."""
        # Test with different data combinations
        for _ in range(3):
            # Generate new test data
            geiger = np.random.rand(self.config["geiger_dimensions"])
            visual = np.random.rand(self.config["visual_dimensions"])
            thermal = np.random.rand(self.config["thermal_dimensions"])
            
            result = self.ai.process_radiation_data(
                geiger, visual, thermal
            )
            
            # Test fused data
            fused_data = result["fused_data"]
            self.assertEqual(len(fused_data), self.config["fusion_dimensions"])
            self.assertTrue(np.all(-1 <= fused_data))
            self.assertTrue(np.all(fused_data <= 1))
            
            # Test analysis consistency
            analysis = result["analysis"]
            self.assertTrue(0 <= analysis["detection"] <= 1)
            self.assertTrue(0 <= analysis["radiation_level"] <= 1)
            self.assertTrue(0 <= analysis["confidence_score"] <= 1)

if __name__ == '__main__':
    unittest.main() 
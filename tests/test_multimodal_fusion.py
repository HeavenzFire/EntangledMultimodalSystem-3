import unittest
import numpy as np
from src.core.multimodal_fusion import MultimodalFusion
from src.utils.errors import ModelError

class TestMultimodalFusion(unittest.TestCase):
    """Test cases for the Multimodal Fusion system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "text_dimensions": 768,
            "image_dimensions": 1024,
            "speech_dimensions": 512,
            "fractal_dimensions": 256,
            "radiation_dimensions": 128,
            "fusion_dimensions": 2048,
            "attention_heads": 8
        }
        self.fusion = MultimodalFusion(self.config)
        
        # Create test data
        self.text_data = np.random.rand(self.config["text_dimensions"])
        self.image_data = np.random.rand(self.config["image_dimensions"])
        self.speech_data = np.random.rand(self.config["speech_dimensions"])
        self.fractal_data = np.random.rand(self.config["fractal_dimensions"])
        self.radiation_data = np.random.rand(self.config["radiation_dimensions"])
    
    def test_initialization(self):
        """Test system initialization."""
        self.assertEqual(self.fusion.text_dimensions, self.config["text_dimensions"])
        self.assertEqual(self.fusion.image_dimensions, self.config["image_dimensions"])
        self.assertEqual(self.fusion.speech_dimensions, self.config["speech_dimensions"])
        self.assertEqual(self.fusion.fractal_dimensions, self.config["fractal_dimensions"])
        self.assertEqual(self.fusion.radiation_dimensions, self.config["radiation_dimensions"])
        self.assertEqual(self.fusion.fusion_dimensions, self.config["fusion_dimensions"])
        self.assertEqual(self.fusion.attention_heads, self.config["attention_heads"])
        
        # Test state initialization
        self.assertIsNone(self.fusion.state["text_data"])
        self.assertIsNone(self.fusion.state["image_data"])
        self.assertIsNone(self.fusion.state["speech_data"])
        self.assertIsNone(self.fusion.state["fractal_data"])
        self.assertIsNone(self.fusion.state["radiation_data"])
        self.assertIsNone(self.fusion.state["fused_data"])
        self.assertIsNone(self.fusion.state["processing_results"])
        
        # Test metrics initialization
        self.assertEqual(self.fusion.metrics["text_quality"], 0.0)
        self.assertEqual(self.fusion.metrics["image_quality"], 0.0)
        self.assertEqual(self.fusion.metrics["speech_quality"], 0.0)
        self.assertEqual(self.fusion.metrics["fractal_quality"], 0.0)
        self.assertEqual(self.fusion.metrics["radiation_quality"], 0.0)
        self.assertEqual(self.fusion.metrics["fusion_quality"], 0.0)
        self.assertEqual(self.fusion.metrics["processing_time"], 0.0)
    
    def test_data_processing(self):
        """Test multimodal data processing."""
        result = self.fusion.process_multimodal_data(
            self.text_data,
            self.image_data,
            self.speech_data,
            self.fractal_data,
            self.radiation_data
        )
        
        # Test result structure
        self.assertIn("fused_data", result)
        self.assertIn("processing", result)
        self.assertIn("metrics", result)
        self.assertIn("state", result)
        
        # Test fused data properties
        self.assertEqual(len(result["fused_data"]), self.config["fusion_dimensions"])
        self.assertTrue(np.all(-1 <= result["fused_data"]))
        self.assertTrue(np.all(result["fused_data"] <= 1))
    
    def test_processing_results(self):
        """Test processing results."""
        result = self.fusion.process_multimodal_data(
            self.text_data,
            self.image_data,
            self.speech_data,
            self.fractal_data,
            self.radiation_data
        )
        processing = result["processing"]
        
        # Test processing properties
        self.assertIn("text_quality", processing)
        self.assertIn("image_quality", processing)
        self.assertIn("speech_quality", processing)
        self.assertIn("fractal_quality", processing)
        self.assertIn("radiation_quality", processing)
        self.assertIn("fusion_quality", processing)
        
        # Test processing ranges
        self.assertTrue(0 <= processing["text_quality"] <= 1)
        self.assertTrue(0 <= processing["image_quality"] <= 1)
        self.assertTrue(0 <= processing["speech_quality"] <= 1)
        self.assertTrue(0 <= processing["fractal_quality"] <= 1)
        self.assertTrue(0 <= processing["radiation_quality"] <= 1)
        self.assertTrue(0 <= processing["fusion_quality"] <= 1)
    
    def test_metrics_calculation(self):
        """Test metrics calculation."""
        result = self.fusion.process_multimodal_data(
            self.text_data,
            self.image_data,
            self.speech_data,
            self.fractal_data,
            self.radiation_data
        )
        metrics = result["metrics"]
        
        # Test metric properties
        self.assertIn("text_quality", metrics)
        self.assertIn("image_quality", metrics)
        self.assertIn("speech_quality", metrics)
        self.assertIn("fractal_quality", metrics)
        self.assertIn("radiation_quality", metrics)
        self.assertIn("fusion_quality", metrics)
        self.assertIn("processing_time", metrics)
        
        # Test metric ranges
        self.assertTrue(0 <= metrics["text_quality"] <= 1)
        self.assertTrue(0 <= metrics["image_quality"] <= 1)
        self.assertTrue(0 <= metrics["speech_quality"] <= 1)
        self.assertTrue(0 <= metrics["fractal_quality"] <= 1)
        self.assertTrue(0 <= metrics["radiation_quality"] <= 1)
        self.assertTrue(0 <= metrics["fusion_quality"] <= 1)
        self.assertEqual(metrics["processing_time"], 0.47)
    
    def test_state_management(self):
        """Test state management."""
        # Test initial state
        initial_state = self.fusion.get_state()
        self.assertIsNone(initial_state["text_data"])
        self.assertIsNone(initial_state["image_data"])
        self.assertIsNone(initial_state["speech_data"])
        self.assertIsNone(initial_state["fractal_data"])
        self.assertIsNone(initial_state["radiation_data"])
        self.assertIsNone(initial_state["fused_data"])
        self.assertIsNone(initial_state["processing_results"])
        
        # Process data and test updated state
        self.fusion.process_multimodal_data(
            self.text_data,
            self.image_data,
            self.speech_data,
            self.fractal_data,
            self.radiation_data
        )
        updated_state = self.fusion.get_state()
        self.assertIsNotNone(updated_state["text_data"])
        self.assertIsNotNone(updated_state["image_data"])
        self.assertIsNotNone(updated_state["speech_data"])
        self.assertIsNotNone(updated_state["fractal_data"])
        self.assertIsNotNone(updated_state["radiation_data"])
        self.assertIsNotNone(updated_state["fused_data"])
        self.assertIsNotNone(updated_state["processing_results"])
        
        # Test reset
        self.fusion.reset()
        reset_state = self.fusion.get_state()
        self.assertIsNone(reset_state["text_data"])
        self.assertIsNone(reset_state["image_data"])
        self.assertIsNone(reset_state["speech_data"])
        self.assertIsNone(reset_state["fractal_data"])
        self.assertIsNone(reset_state["radiation_data"])
        self.assertIsNone(reset_state["fused_data"])
        self.assertIsNone(reset_state["processing_results"])
    
    def test_error_handling(self):
        """Test error handling."""
        # Test invalid text data
        invalid_text = np.random.rand(self.config["text_dimensions"] + 1)
        with self.assertRaises(ModelError):
            self.fusion.process_multimodal_data(
                invalid_text,
                self.image_data,
                self.speech_data,
                self.fractal_data,
                self.radiation_data
            )
        
        # Test invalid image data
        invalid_image = np.random.rand(self.config["image_dimensions"] + 1)
        with self.assertRaises(ModelError):
            self.fusion.process_multimodal_data(
                self.text_data,
                invalid_image,
                self.speech_data,
                self.fractal_data,
                self.radiation_data
            )
        
        # Test invalid speech data
        invalid_speech = np.random.rand(self.config["speech_dimensions"] + 1)
        with self.assertRaises(ModelError):
            self.fusion.process_multimodal_data(
                self.text_data,
                self.image_data,
                invalid_speech,
                self.fractal_data,
                self.radiation_data
            )
        
        # Test invalid fractal data
        invalid_fractal = np.random.rand(self.config["fractal_dimensions"] + 1)
        with self.assertRaises(ModelError):
            self.fusion.process_multimodal_data(
                self.text_data,
                self.image_data,
                self.speech_data,
                invalid_fractal,
                self.radiation_data
            )
        
        # Test invalid radiation data
        invalid_radiation = np.random.rand(self.config["radiation_dimensions"] + 1)
        with self.assertRaises(ModelError):
            self.fusion.process_multimodal_data(
                self.text_data,
                self.image_data,
                self.speech_data,
                self.fractal_data,
                invalid_radiation
            )
    
    def test_multimodal_fusion(self):
        """Test multimodal fusion quality."""
        # Test with different data combinations
        for _ in range(3):
            # Generate new test data
            text = np.random.rand(self.config["text_dimensions"])
            image = np.random.rand(self.config["image_dimensions"])
            speech = np.random.rand(self.config["speech_dimensions"])
            fractal = np.random.rand(self.config["fractal_dimensions"])
            radiation = np.random.rand(self.config["radiation_dimensions"])
            
            result = self.fusion.process_multimodal_data(
                text, image, speech, fractal, radiation
            )
            
            # Test fused data
            fused_data = result["fused_data"]
            self.assertEqual(len(fused_data), self.config["fusion_dimensions"])
            self.assertTrue(np.all(-1 <= fused_data))
            self.assertTrue(np.all(fused_data <= 1))
            
            # Test processing consistency
            processing = result["processing"]
            self.assertTrue(0 <= processing["text_quality"] <= 1)
            self.assertTrue(0 <= processing["image_quality"] <= 1)
            self.assertTrue(0 <= processing["speech_quality"] <= 1)
            self.assertTrue(0 <= processing["fractal_quality"] <= 1)
            self.assertTrue(0 <= processing["radiation_quality"] <= 1)
            self.assertTrue(0 <= processing["fusion_quality"] <= 1)

if __name__ == '__main__':
    unittest.main() 
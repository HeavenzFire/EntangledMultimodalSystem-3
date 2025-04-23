import unittest
import torch
import numpy as np
from core.neural_integrator import NeuralIntegrator

class TestNeuralIntegrator(unittest.TestCase):
    def setUp(self):
        self.config = {
            'input_channels': 3,
            'hidden_channels': [64, 128, 256],
            'output_channels': 3
        }
        self.model = NeuralIntegrator(self.config)
        
        # Generate test data
        self.test_input = torch.randn(1, 3, 256, 256)
        self.test_modalities = {
            'visual': torch.randn(1, 3, 256, 256),
            'audio': torch.randn(1, 3, 256, 256),
            'tactile': torch.randn(1, 3, 256, 256)
        }
    
    def test_initialization(self):
        """Test model initialization"""
        self.assertIsInstance(self.model, NeuralIntegrator)
        self.assertIsNotNone(self.model.encoder)
        self.assertIsNotNone(self.model.processor)
        self.assertIsNotNone(self.model.decoder)
    
    def test_forward_pass(self):
        """Test forward pass through the network"""
        output = self.model(self.test_input)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (1, 3, 256, 256))
    
    def test_multimodal_processing(self):
        """Test processing of multimodal data"""
        output = self.model.process_multimodal_data(self.test_modalities)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (1, 3, 256, 256))
    
    def test_feature_extraction(self):
        """Test feature extraction from different layers"""
        features = self.model.extract_features(self.test_input)
        self.assertIsInstance(features, dict)
        self.assertIn('encoder_conv1', features)
        self.assertIn('encoder_conv2', features)
        self.assertIn('encoder_pool', features)
        self.assertIn('processor_conv1', features)
        self.assertIn('processor_conv2', features)
        self.assertIn('processor_conv3', features)
    
    def test_loss_computation(self):
        """Test loss computation"""
        output = self.model(self.test_input)
        target = torch.randn_like(output)
        losses = self.model.compute_loss(output, target)
        
        self.assertIsInstance(losses, dict)
        self.assertIn('reconstruction', losses)
        self.assertIn('feature_encoder_conv1', losses)
        self.assertIn('feature_encoder_conv2', losses)
        self.assertIn('feature_encoder_pool', losses)
        self.assertIn('feature_processor_conv1', losses)
        self.assertIn('feature_processor_conv2', losses)
        self.assertIn('feature_processor_conv3', losses)
    
    def test_performance(self):
        """Test processing performance"""
        import time
        start_time = time.time()
        output = self.model.process_multimodal_data(self.test_modalities)
        processing_time = time.time() - start_time
        self.assertLess(processing_time, 1.0)  # Should process within 1 second
    
    def test_gradient_flow(self):
        """Test gradient flow through the network"""
        output = self.model(self.test_input)
        target = torch.randn_like(output)
        losses = self.model.compute_loss(output, target)
        
        # Compute total loss
        total_loss = sum(losses.values())
        
        # Backward pass
        total_loss.backward()
        
        # Check gradients
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                self.assertFalse(torch.isnan(param.grad).any())
                self.assertFalse(torch.isinf(param.grad).any())

if __name__ == '__main__':
    unittest.main() 
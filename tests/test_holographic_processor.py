import unittest
import torch
import numpy as np
from src.core.holographic_processor import HolographicProcessor
from src.utils.errors import ModelError

class TestHolographicProcessor(unittest.TestCase):
    def setUp(self):
        self.config = {
            'wavefront_size': 256,
            'num_channels': 3
        }
        self.processor = HolographicProcessor(self.config)
        
        # Generate test data
        self.test_tensor = torch.randn(256, 256)
        self.test_modalities = {
            'visual': torch.randn(256, 256),
            'audio': torch.randn(256, 256),
            'tactile': torch.randn(256, 256)
        }
    
    def test_initialization(self):
        """Test processor initialization"""
        self.assertIsNone(self.processor.wavefront)
        self.assertIsNone(self.processor.reference_wave)
        
    def test_wavefront_initialization(self):
        """Test wavefront initialization"""
        self.processor.initialize_wavefront((256, 256))
        self.assertIsNotNone(self.processor.wavefront)
        self.assertIsNotNone(self.processor.reference_wave)
        self.assertEqual(self.processor.wavefront.shape, (256, 256))
        self.assertEqual(self.processor.reference_wave.shape, (256, 256))
    
    def test_data_encoding(self):
        """Test data encoding into holographic pattern"""
        hologram = self.processor.encode_data(self.test_tensor)
        self.assertIsInstance(hologram, np.ndarray)
        self.assertEqual(hologram.shape, (256, 256))
        self.assertEqual(hologram.dtype, np.complex128)
    
    def test_data_decoding(self):
        """Test data decoding from holographic pattern"""
        hologram = self.processor.encode_data(self.test_tensor)
        decoded = self.processor.decode_data(hologram)
        self.assertIsInstance(decoded, torch.Tensor)
        self.assertEqual(decoded.shape, (256, 256))
    
    def test_phase_modulation(self):
        """Test phase modulation of holographic pattern"""
        hologram = self.processor.encode_data(self.test_tensor)
        phase = np.random.rand(256, 256) * 2 * np.pi
        modulated = self.processor.apply_phase_modulation(hologram, phase)
        self.assertIsInstance(modulated, np.ndarray)
        self.assertEqual(modulated.shape, (256, 256))
        self.assertEqual(modulated.dtype, np.complex128)
    
    def test_wavefront_reconstruction(self):
        """Test wavefront reconstruction"""
        self.processor.initialize_wavefront((256, 256))
        hologram = self.processor.encode_data(self.test_tensor)
        reconstructed = self.processor.reconstruct_wavefront(hologram)
        self.assertIsInstance(reconstructed, np.ndarray)
        self.assertEqual(reconstructed.shape, (256, 256))
    
    def test_multimodal_processing(self):
        """Test processing of multimodal data"""
        result = self.processor.process_multimodal_data(self.test_modalities)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (256, 256))
    
    def test_performance(self):
        """Test processing performance"""
        import time
        start_time = time.time()
        result = self.processor.process_multimodal_data(self.test_modalities)
        processing_time = time.time() - start_time
        self.assertLess(processing_time, 1.0)  # Should process within 1 second

if __name__ == '__main__':
    unittest.main() 
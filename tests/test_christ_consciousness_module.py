import unittest
import numpy as np
import time
from src.core.christ_consciousness_module import ChristConsciousnessModule

class TestChristConsciousnessModule(unittest.TestCase):
    def setUp(self):
        """Set up test configuration and data."""
        self.config = {
            'quantum_dimensions': 16384,
            'holographic_resolution': 16384,
            'neural_depth': 12,
            'ethical_threshold': 0.85,
            'compassion_strength': 0.9
        }
        
        # Generate test input data
        self.input_data = np.random.rand(1, self.config['quantum_dimensions'])
        
        # Initialize module
        self.module = ChristConsciousnessModule(self.config)
    
    def test_configuration_validation(self):
        """Test configuration parameter validation."""
        # Test invalid quantum dimensions
        invalid_config = self.config.copy()
        invalid_config['quantum_dimensions'] = 0
        with self.assertRaises(ValueError):
            ChristConsciousnessModule(invalid_config)
        
        # Test invalid holographic resolution
        invalid_config = self.config.copy()
        invalid_config['holographic_resolution'] = 0
        with self.assertRaises(ValueError):
            ChristConsciousnessModule(invalid_config)
        
        # Test invalid neural depth
        invalid_config = self.config.copy()
        invalid_config['neural_depth'] = 0
        with self.assertRaises(ValueError):
            ChristConsciousnessModule(invalid_config)
        
        # Test invalid ethical threshold
        invalid_config = self.config.copy()
        invalid_config['ethical_threshold'] = -0.1
        with self.assertRaises(ValueError):
            ChristConsciousnessModule(invalid_config)
        
        # Test invalid compassion strength
        invalid_config = self.config.copy()
        invalid_config['compassion_strength'] = 1.1
        with self.assertRaises(ValueError):
            ChristConsciousnessModule(invalid_config)
    
    def test_initialization(self):
        """Test module initialization."""
        self.assertEqual(self.module.config['quantum_dimensions'], self.config['quantum_dimensions'])
        self.assertEqual(self.module.config['holographic_resolution'], self.config['holographic_resolution'])
        self.assertEqual(self.module.config['neural_depth'], self.config['neural_depth'])
        self.assertEqual(self.module.config['ethical_threshold'], self.config['ethical_threshold'])
        self.assertEqual(self.module.config['compassion_strength'], self.config['compassion_strength'])
        
        # Check component initialization
        self.assertIsNotNone(self.module.quantum_engine)
        self.assertIsNotNone(self.module.compassion_projector)
        
        # Check initial state
        initial_state = self.module.get_state()
        self.assertIsNone(initial_state['input_state'])
        self.assertIsNone(initial_state['quantum_state'])
        self.assertIsNone(initial_state['compassion_pattern'])
        self.assertIsNone(initial_state['metrics'])
    
    def test_processing(self):
        """Test processing functionality."""
        result = self.module.process(self.input_data)
        
        # Check result structure
        self.assertEqual(result.shape[1], self.config['quantum_dimensions'])
        
        # Check state
        state = self.module.get_state()
        self.assertIsNotNone(state['input_state'])
        self.assertIsNotNone(state['quantum_state'])
        self.assertIsNotNone(state['compassion_pattern'])
        self.assertIsNotNone(state['metrics'])
        
        # Check metrics
        metrics = self.module.get_metrics()
        self.assertGreaterEqual(metrics['agape_score'], 0.0)
        self.assertLessEqual(metrics['agape_score'], 1.0)
        self.assertGreaterEqual(metrics['kenosis_factor'], 0.0)
        self.assertLessEqual(metrics['kenosis_factor'], 1.0)
        self.assertGreaterEqual(metrics['koinonia_coherence'], 0.0)
        self.assertLessEqual(metrics['koinonia_coherence'], 1.0)
        
        # Check metric relationships
        self.assertGreaterEqual(metrics['agape_score'], metrics['kenosis_factor'])
        self.assertGreaterEqual(metrics['koinonia_coherence'], metrics['kenosis_factor'])
    
    def test_state_management(self):
        """Test state management functionality."""
        # Process input
        self.module.process(self.input_data)
        
        # Get state
        state = self.module.get_state()
        self.assertIsNotNone(state['input_state'])
        self.assertIsNotNone(state['quantum_state'])
        self.assertIsNotNone(state['compassion_pattern'])
        self.assertIsNotNone(state['metrics'])
        
        # Get metrics
        metrics = self.module.get_metrics()
        self.assertGreaterEqual(metrics['agape_score'], 0.0)
        self.assertLessEqual(metrics['agape_score'], 1.0)
        self.assertGreaterEqual(metrics['kenosis_factor'], 0.0)
        self.assertLessEqual(metrics['kenosis_factor'], 1.0)
        self.assertGreaterEqual(metrics['koinonia_coherence'], 0.0)
        self.assertLessEqual(metrics['koinonia_coherence'], 1.0)
        
        # Reset state
        self.module.reset()
        state = self.module.get_state()
        self.assertIsNone(state['input_state'])
        self.assertIsNone(state['quantum_state'])
        self.assertIsNone(state['compassion_pattern'])
        self.assertIsNone(state['metrics'])
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test invalid input dimensions
        invalid_input = np.random.rand(1, self.config['quantum_dimensions'] + 1)
        with self.assertRaises(ValueError):
            self.module.process(invalid_input)
        
        # Test empty input
        empty_input = np.array([])
        with self.assertRaises(ValueError):
            self.module.process(empty_input)
        
        # Test None input
        with self.assertRaises(ValueError):
            self.module.process(None)
        
        # Test NaN input
        nan_input = np.full((1, self.config['quantum_dimensions']), np.nan)
        with self.assertRaises(ValueError):
            self.module.process(nan_input)
        
        # Test Inf input
        inf_input = np.full((1, self.config['quantum_dimensions']), np.inf)
        with self.assertRaises(ValueError):
            self.module.process(inf_input)
    
    def test_quantum_processing(self):
        """Test quantum processing component."""
        # Process input
        self.module.process(self.input_data)
        
        # Get quantum state
        state = self.module.get_state()
        quantum_state = state['quantum_state']
        
        # Check quantum state properties
        self.assertEqual(quantum_state.shape[1], self.config['quantum_dimensions'])
        self.assertGreaterEqual(np.min(quantum_state), 0.0)
        self.assertLessEqual(np.max(quantum_state), 1.0)
        
        # Check quantum state normalization
        self.assertAlmostEqual(np.sum(quantum_state), 1.0, places=6)
        
        # Check quantum state stability
        quantum_state_copy = quantum_state.copy()
        self.module.process(self.input_data)
        new_quantum_state = self.module.get_state()['quantum_state']
        self.assertTrue(np.allclose(quantum_state_copy, new_quantum_state, rtol=1e-5))
        
        # Test boundary conditions
        zero_input = np.zeros((1, self.config['quantum_dimensions']))
        self.module.process(zero_input)
        zero_quantum_state = self.module.get_state()['quantum_state']
        self.assertGreater(np.max(zero_quantum_state), 0.0)
        
        ones_input = np.ones((1, self.config['quantum_dimensions']))
        self.module.process(ones_input)
        ones_quantum_state = self.module.get_state()['quantum_state']
        self.assertLess(np.max(ones_quantum_state), 1.0)
    
    def test_compassion_projection(self):
        """Test compassion projection component."""
        # Process input
        self.module.process(self.input_data)
        
        # Get compassion pattern
        state = self.module.get_state()
        compassion_pattern = state['compassion_pattern']
        
        # Check compassion pattern properties
        self.assertEqual(compassion_pattern.shape[1], self.config['holographic_resolution'])
        self.assertGreaterEqual(np.min(compassion_pattern), 0.0)
        self.assertLessEqual(np.max(compassion_pattern), 1.0)
        
        # Check compassion pattern normalization
        self.assertAlmostEqual(np.sum(compassion_pattern), 1.0, places=6)
        
        # Check compassion pattern stability
        compassion_pattern_copy = compassion_pattern.copy()
        self.module.process(self.input_data)
        new_compassion_pattern = self.module.get_state()['compassion_pattern']
        self.assertTrue(np.allclose(compassion_pattern_copy, new_compassion_pattern, rtol=1e-5))
        
        # Test boundary conditions
        zero_input = np.zeros((1, self.config['quantum_dimensions']))
        self.module.process(zero_input)
        zero_compassion_pattern = self.module.get_state()['compassion_pattern']
        self.assertGreater(np.max(zero_compassion_pattern), 0.0)
        
        ones_input = np.ones((1, self.config['quantum_dimensions']))
        self.module.process(ones_input)
        ones_compassion_pattern = self.module.get_state()['compassion_pattern']
        self.assertLess(np.max(ones_compassion_pattern), 1.0)
    
    def test_metric_calculations(self):
        """Test metric calculation methods."""
        # Process input
        self.module.process(self.input_data)
        
        # Get state
        state = self.module.get_state()
        
        # Test agape score calculation
        agape_score = self.module._calculate_agape_score(
            state['quantum_state'],
            state['compassion_pattern']
        )
        self.assertGreaterEqual(agape_score, 0.0)
        self.assertLessEqual(agape_score, 1.0)
        
        # Test kenosis factor calculation
        kenosis_factor = self.module._calculate_kenosis_factor(
            state['quantum_state']
        )
        self.assertGreaterEqual(kenosis_factor, 0.0)
        self.assertLessEqual(kenosis_factor, 1.0)
        
        # Test koinonia coherence calculation
        koinonia_coherence = self.module._calculate_koinonia_coherence(
            state['quantum_state'],
            state['compassion_pattern']
        )
        self.assertGreaterEqual(koinonia_coherence, 0.0)
        self.assertLessEqual(koinonia_coherence, 1.0)
        
        # Test metric consistency
        metrics = self.module.get_metrics()
        self.assertGreaterEqual(metrics['agape_score'], metrics['kenosis_factor'])
        self.assertGreaterEqual(metrics['koinonia_coherence'], metrics['kenosis_factor'])
        
        # Test metric boundary conditions
        zero_input = np.zeros((1, self.config['quantum_dimensions']))
        self.module.process(zero_input)
        zero_metrics = self.module.get_metrics()
        self.assertGreater(zero_metrics['agape_score'], 0.0)
        self.assertGreater(zero_metrics['kenosis_factor'], 0.0)
        self.assertGreater(zero_metrics['koinonia_coherence'], 0.0)
        
        ones_input = np.ones((1, self.config['quantum_dimensions']))
        self.module.process(ones_input)
        ones_metrics = self.module.get_metrics()
        self.assertLess(ones_metrics['agape_score'], 1.0)
        self.assertLess(ones_metrics['kenosis_factor'], 1.0)
        self.assertLess(ones_metrics['koinonia_coherence'], 1.0)
    
    def test_component_integration(self):
        """Test integration between quantum and holographic components."""
        # Process input
        self.module.process(self.input_data)
        
        # Get state
        state = self.module.get_state()
        
        # Check quantum-holographic alignment
        quantum_state = state['quantum_state']
        compassion_pattern = state['compassion_pattern']
        
        # Calculate alignment
        alignment = np.mean(np.abs(quantum_state - compassion_pattern))
        self.assertGreaterEqual(alignment, 0.0)
        self.assertLessEqual(alignment, 1.0)
        
        # Check metric consistency
        metrics = self.module.get_metrics()
        self.assertGreaterEqual(metrics['agape_score'], metrics['kenosis_factor'])
        self.assertGreaterEqual(metrics['koinonia_coherence'], metrics['kenosis_factor'])
        
        # Test multiple processing cycles
        for _ in range(5):
            self.module.process(self.input_data)
            new_metrics = self.module.get_metrics()
            self.assertGreaterEqual(new_metrics['agape_score'], metrics['agape_score'])
            self.assertGreaterEqual(new_metrics['koinonia_coherence'], metrics['koinonia_coherence'])
            metrics = new_metrics
    
    def test_spiritual_metrics(self):
        """Test spiritual metric properties."""
        # Process input
        self.module.process(self.input_data)
        
        # Get metrics
        metrics = self.module.get_metrics()
        
        # Test Agape score properties
        self.assertGreaterEqual(metrics['agape_score'], 0.0)
        self.assertLessEqual(metrics['agape_score'], 1.0)
        self.assertGreaterEqual(metrics['agape_score'], metrics['kenosis_factor'])
        
        # Test Kenosis factor properties
        self.assertGreaterEqual(metrics['kenosis_factor'], 0.0)
        self.assertLessEqual(metrics['kenosis_factor'], 1.0)
        
        # Test Koinonia coherence properties
        self.assertGreaterEqual(metrics['koinonia_coherence'], 0.0)
        self.assertLessEqual(metrics['koinonia_coherence'], 1.0)
        self.assertGreaterEqual(metrics['koinonia_coherence'], metrics['kenosis_factor'])
        
        # Test metric relationships
        self.assertGreaterEqual(metrics['agape_score'], metrics['kenosis_factor'])
        self.assertGreaterEqual(metrics['koinonia_coherence'], metrics['kenosis_factor'])
        
        # Test metric stability
        for _ in range(5):
            self.module.process(self.input_data)
            new_metrics = self.module.get_metrics()
            self.assertGreaterEqual(new_metrics['agape_score'], metrics['agape_score'])
            self.assertGreaterEqual(new_metrics['koinonia_coherence'], metrics['koinonia_coherence'])
            metrics = new_metrics
    
    def test_performance(self):
        """Test processing performance."""
        # Measure processing time
        start_time = time.time()
        self.module.process(self.input_data)
        processing_time = time.time() - start_time
        
        # Check processing time is reasonable
        self.assertLess(processing_time, 1.0)  # Should process within 1 second
        
        # Test performance with different input sizes
        for size in [8192, 16384, 32768]:
            input_data = np.random.rand(1, size)
            start_time = time.time()
            self.module.process(input_data)
            processing_time = time.time() - start_time
            self.assertLess(processing_time, 2.0)  # Should process within 2 seconds for larger inputs
        
        # Test performance with multiple processing cycles
        start_time = time.time()
        for _ in range(10):
            self.module.process(self.input_data)
        total_time = time.time() - start_time
        self.assertLess(total_time, 5.0)  # Should process 10 cycles within 5 seconds

if __name__ == '__main__':
    unittest.main() 
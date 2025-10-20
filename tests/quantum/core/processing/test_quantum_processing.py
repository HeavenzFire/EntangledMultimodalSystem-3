import pytest
import numpy as np
from src.quantum.core.processing.quantum_processing import (
    QuantumStateProcessor,
    ProcessingResult
)
from src.quantum.core.quantum_state import QuantumState
from src.quantum.core.nonlinear_processor import NonlinearProcessor

class TestQuantumStateProcessor:
    @pytest.fixture
    def processor(self):
        return QuantumStateProcessor()
    
    @pytest.fixture
    def test_state(self):
        return QuantumState(
            amplitude=1.0,
            phase=np.pi/4,
            error_rate=0.001
        )
    
    def test_state_initialization(self, processor):
        """Test quantum state initialization"""
        state = processor.initialize_state(amplitude=1.0, phase=np.pi/4)
        
        assert isinstance(state, QuantumState)
        assert state.amplitude == 1.0
        assert state.phase == np.pi/4
        assert state.error_rate < 0.1
    
    def test_state_evolution(self, processor, test_state):
        """Test quantum state evolution"""
        result = processor.evolve_state(test_state, time=1.0)
        
        assert isinstance(result, ProcessingResult)
        assert result.success
        assert isinstance(result.final_state, QuantumState)
        assert result.final_state.amplitude != test_state.amplitude
        assert result.final_state.phase != test_state.phase
    
    def test_nonlinear_processing(self, processor, test_state):
        """Test nonlinear processing of quantum state"""
        result = processor.apply_nonlinear_processing(test_state)
        
        assert isinstance(result, ProcessingResult)
        assert result.success
        assert result.final_state.coherence > 0.8
        assert result.final_state.error_rate < 0.1
    
    def test_state_measurement(self, processor, test_state):
        """Test quantum state measurement"""
        result = processor.measure_state(test_state)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert 0 <= result[0] <= 1.0  # Probability
        assert 0 <= result[1] <= 2 * np.pi  # Phase
    
    def test_state_transformation(self, processor, test_state):
        """Test quantum state transformation"""
        result = processor.transform_state(test_state, transformation='rotation')
        
        assert isinstance(result, ProcessingResult)
        assert result.success
        assert result.final_state.amplitude != test_state.amplitude
        assert result.final_state.phase != test_state.phase
    
    def test_error_correction(self, processor, test_state):
        """Test quantum error correction"""
        # Introduce some error
        test_state.error_rate = 0.2
        
        result = processor.correct_errors(test_state)
        
        assert isinstance(result, ProcessingResult)
        assert result.success
        assert result.final_state.error_rate < 0.1
    
    def test_state_verification(self, processor, test_state):
        """Test quantum state verification"""
        result = processor.verify_state(test_state)
        
        assert isinstance(result, ProcessingResult)
        assert result.success
        assert result.verification_passed
        assert result.error_rate < 0.1
    
    def test_processing_robustness(self, processor, test_state):
        """Test processing robustness"""
        results = []
        for _ in range(5):
            result = processor.evolve_state(test_state, time=1.0)
            results.append(result.final_state)
        
        # Check consistency of results
        for i in range(1, len(results)):
            assert np.isclose(results[i].amplitude, results[0].amplitude, atol=1e-6)
            assert np.isclose(results[i].phase, results[0].phase, atol=1e-6)
    
    def test_error_handling(self, processor):
        """Test error handling"""
        # Test with invalid state
        with pytest.raises(Exception):
            processor.evolve_state(None, time=1.0)
        
        # Test with invalid parameters
        with pytest.raises(Exception):
            processor.transform_state(None, transformation=None)
    
    def test_processing_quality(self, processor, test_state):
        """Test processing quality"""
        result = processor.evolve_state(test_state, time=1.0)
        
        assert result.error_rate < 0.1  # Should have low error rate
        assert result.final_state.coherence > 0.8  # Should maintain high coherence
        assert result.verification_passed  # Should pass verification
    
    def test_processing_efficiency(self, processor, test_state):
        """Test processing efficiency"""
        import time
        
        start_time = time.time()
        result = processor.evolve_state(test_state, time=1.0)
        processing_time = time.time() - start_time
        
        assert result.success
        assert processing_time < 1.0  # Should process within reasonable time
        assert result.error_rate < 0.1
    
    def test_state_preservation(self, processor, test_state):
        """Test state preservation during processing"""
        initial_amplitude = test_state.amplitude
        initial_phase = test_state.phase
        
        result = processor.evolve_state(test_state, time=1.0)
        
        assert result.success
        assert not np.isclose(result.final_state.amplitude, initial_amplitude)
        assert not np.isclose(result.final_state.phase, initial_phase)
    
    def test_processing_consistency(self, processor, test_state):
        """Test consistency of processing results"""
        results = []
        for _ in range(10):
            result = processor.evolve_state(test_state, time=1.0)
            results.append(result.final_state)
        
        # Check consistency of results
        for i in range(1, len(results)):
            assert np.isclose(results[i].amplitude, results[0].amplitude, atol=1e-6)
            assert np.isclose(results[i].phase, results[0].phase, atol=1e-6)
    
    def test_nonlinear_operations(self, processor, test_state):
        """Test different nonlinear operations"""
        operations = ['squeezing', 'kerr', 'cross_kerr']
        
        for operation in operations:
            result = processor.apply_nonlinear_processing(test_state, operation=operation)
            assert result.success
            assert result.final_state.coherence > 0.8
            assert result.final_state.error_rate < 0.1
    
    def test_state_quality(self, processor, test_state):
        """Test state quality metrics"""
        result = processor.evolve_state(test_state, time=1.0)
        
        assert result.final_state.coherence > 0.8  # High coherence
        assert result.final_state.error_rate < 0.1  # Low error rate
        assert result.final_state.amplitude > 0  # Valid amplitude
        assert 0 <= result.final_state.phase <= 2 * np.pi  # Valid phase 
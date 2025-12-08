import pytest
import numpy as np
from src.quantum.core.teleportation.quantum_teleportation import (
    QuantumTeleportation,
    TeleportationResult
)
from src.quantum.core.quantum_state import QuantumState
from src.quantum.core.entanglement.multi_qubit_entanglement import (
    MultiQubitEntanglement,
    EntanglementResult
)

class TestQuantumTeleportation:
    @pytest.fixture
    def teleportation(self):
        return QuantumTeleportation()
    
    @pytest.fixture
    def entanglement(self):
        return MultiQubitEntanglement()
    
    @pytest.fixture
    def test_state(self):
        return QuantumState(
            amplitude=1.0,
            phase=np.pi/4,
            error_rate=0.001
        )
    
    def test_teleportation_initialization(self, teleportation):
        """Test teleportation system initialization"""
        assert teleportation.entanglement is not None
        assert teleportation.error_correction is not None
        assert teleportation.verification is not None
    
    def test_state_transfer(self, teleportation, test_state):
        """Test quantum state transfer"""
        result = teleportation.transfer_state(test_state)
        
        assert isinstance(result, TeleportationResult)
        assert result.success
        assert isinstance(result.transferred_state, QuantumState)
        assert result.transferred_state.amplitude != test_state.amplitude
        assert result.transferred_state.phase != test_state.phase
        assert result.error_rate < 0.1
    
    def test_entanglement_creation(self, teleportation):
        """Test entanglement creation for teleportation"""
        result = teleportation.create_entanglement()
        
        assert isinstance(result, EntanglementResult)
        assert result.success
        assert result.entanglement_degree > 0.8
        assert result.error_rate < 0.1
    
    def test_state_verification(self, teleportation, test_state):
        """Test state verification after teleportation"""
        result = teleportation.transfer_state(test_state)
        verification = teleportation.verify_state(result.transferred_state)
        
        assert verification.success
        assert verification.verification_passed
        assert verification.error_rate < 0.1
    
    def test_error_correction(self, teleportation, test_state):
        """Test error correction during teleportation"""
        # Introduce some error
        test_state.error_rate = 0.2
        
        result = teleportation.transfer_state(test_state)
        
        assert result.success
        assert result.error_rate < 0.1
        assert result.correction_applied
    
    def test_teleportation_robustness(self, teleportation, test_state):
        """Test teleportation robustness"""
        results = []
        for _ in range(5):
            result = teleportation.transfer_state(test_state)
            results.append(result.transferred_state)
        
        # Check consistency of results
        for i in range(1, len(results)):
            assert np.isclose(results[i].amplitude, results[0].amplitude, atol=1e-6)
            assert np.isclose(results[i].phase, results[0].phase, atol=1e-6)
    
    def test_error_handling(self, teleportation):
        """Test error handling"""
        # Test with invalid state
        with pytest.raises(Exception):
            teleportation.transfer_state(None)
        
        # Test with invalid parameters
        with pytest.raises(Exception):
            teleportation.verify_state(None)
    
    def test_teleportation_quality(self, teleportation, test_state):
        """Test teleportation quality"""
        result = teleportation.transfer_state(test_state)
        
        assert result.error_rate < 0.1  # Should have low error rate
        assert result.transferred_state.coherence > 0.8  # Should maintain high coherence
        assert result.verification_passed  # Should pass verification
    
    def test_teleportation_efficiency(self, teleportation, test_state):
        """Test teleportation efficiency"""
        import time
        
        start_time = time.time()
        result = teleportation.transfer_state(test_state)
        transfer_time = time.time() - start_time
        
        assert result.success
        assert transfer_time < 1.0  # Should transfer within reasonable time
        assert result.error_rate < 0.1
    
    def test_state_preservation(self, teleportation, test_state):
        """Test state preservation during teleportation"""
        initial_amplitude = test_state.amplitude
        initial_phase = test_state.phase
        
        result = teleportation.transfer_state(test_state)
        
        assert result.success
        assert not np.isclose(result.transferred_state.amplitude, initial_amplitude)
        assert not np.isclose(result.transferred_state.phase, initial_phase)
    
    def test_teleportation_consistency(self, teleportation, test_state):
        """Test consistency of teleportation results"""
        results = []
        for _ in range(10):
            result = teleportation.transfer_state(test_state)
            results.append(result.transferred_state)
        
        # Check consistency of results
        for i in range(1, len(results)):
            assert np.isclose(results[i].amplitude, results[0].amplitude, atol=1e-6)
            assert np.isclose(results[i].phase, results[0].phase, atol=1e-6)
    
    def test_entanglement_quality(self, teleportation):
        """Test quality of entanglement for teleportation"""
        result = teleportation.create_entanglement()
        
        assert result.entanglement_degree > 0.8  # High entanglement
        assert result.error_rate < 0.1  # Low error rate
        assert result.coherence > 0.8  # High coherence
    
    def test_teleportation_verification(self, teleportation, test_state):
        """Test teleportation verification process"""
        result = teleportation.transfer_state(test_state)
        verification = teleportation.verify_state(result.transferred_state)
        
        assert verification.success
        assert verification.verification_passed
        assert verification.error_rate < 0.1
        assert verification.coherence > 0.8
    
    def test_multiple_teleportations(self, teleportation, test_state):
        """Test multiple consecutive teleportations"""
        # Perform multiple teleportations
        current_state = test_state
        for _ in range(3):
            result = teleportation.transfer_state(current_state)
            current_state = result.transferred_state
        
        # Verify final state quality
        assert current_state.coherence > 0.8
        assert current_state.error_rate < 0.1
        assert 0 <= current_state.phase <= 2 * np.pi
        assert current_state.amplitude > 0 
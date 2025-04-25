import pytest
import numpy as np
from src.quantum.core.entanglement.multi_qubit_entanglement import (
    MultiQubitEntanglement,
    EntanglementResult
)
from src.quantum.core.qubit_control import QubitController, QubitState

class TestMultiQubitEntanglement:
    @pytest.fixture
    def entanglement(self):
        controller = QubitController(num_qubits=8)  # Enough qubits for testing
        return MultiQubitEntanglement(controller)
    
    @pytest.fixture
    def test_qubits(self):
        return [0, 1, 2, 3]  # 4 qubits for testing
    
    def test_create_ghz_state(self, entanglement, test_qubits):
        """Test creation of GHZ state"""
        result = entanglement.create_ghz_state(test_qubits)
        
        assert isinstance(result, EntanglementResult)
        assert result.success
        assert result.entanglement_measure > 0.8  # Should have high entanglement
        assert result.verification_passed
    
    def test_create_w_state(self, entanglement, test_qubits):
        """Test creation of W state"""
        result = entanglement.create_w_state(test_qubits)
        
        assert isinstance(result, EntanglementResult)
        assert result.success
        assert result.entanglement_measure > 0.8  # Should have high entanglement
        assert result.verification_passed
    
    def test_create_cluster_state(self, entanglement, test_qubits):
        """Test creation of cluster state"""
        result = entanglement.create_cluster_state(test_qubits)
        
        assert isinstance(result, EntanglementResult)
        assert result.success
        assert result.entanglement_measure > 0.8  # Should have high entanglement
        assert result.verification_passed
    
    def test_measure_entanglement(self, entanglement, test_qubits):
        """Test entanglement measurement"""
        # Create a GHZ state first
        entanglement.create_ghz_state(test_qubits)
        
        measure = entanglement._measure_entanglement(test_qubits)
        assert 0 <= measure <= 1
        assert measure > 0.8  # Should show high entanglement
    
    def test_verify_entanglement(self, entanglement, test_qubits):
        """Test entanglement verification"""
        # Create a GHZ state first
        entanglement.create_ghz_state(test_qubits)
        
        verified = entanglement._verify_entanglement(test_qubits)
        assert verified
    
    def test_apply_entangling_gates(self, entanglement, test_qubits):
        """Test application of entangling gates"""
        success = entanglement._apply_entangling_gates(test_qubits, 'ghz')
        assert success
        
        # Verify entanglement was created
        measure = entanglement._measure_entanglement(test_qubits)
        assert measure > 0.8
    
    def test_entanglement_robustness(self, entanglement, test_qubits):
        """Test entanglement robustness"""
        # Test multiple entanglement creations
        for state_type in ['ghz', 'w', 'cluster']:
            result = getattr(entanglement, f'create_{state_type}_state')(test_qubits)
            assert result.success
            assert result.entanglement_measure > 0.8
            assert result.verification_passed
    
    def test_error_handling(self, entanglement):
        """Test error handling"""
        # Test with invalid qubit indices
        with pytest.raises(Exception):
            entanglement.create_ghz_state([-1, 0, 1])
        
        with pytest.raises(Exception):
            entanglement.create_ghz_state([0, 100, 1])
        
        # Test with insufficient qubits
        with pytest.raises(Exception):
            entanglement.create_ghz_state([0])
    
    def test_entanglement_consistency(self, entanglement, test_qubits):
        """Test consistency of entanglement results"""
        results = []
        for _ in range(5):
            result = entanglement.create_ghz_state(test_qubits)
            results.append(result.entanglement_measure)
        
        # Check consistency of results
        assert np.std(results) < 0.1  # Should be relatively consistent
    
    def test_state_preservation(self, entanglement, test_qubits):
        """Test state preservation during entanglement"""
        # Create initial state
        entanglement.controller.apply_gate(entanglement.entanglement_gates['H'], test_qubits[0])
        
        # Create entangled state
        result = entanglement.create_ghz_state(test_qubits)
        
        assert result.success
        assert result.entanglement_measure > 0.8
        assert result.verification_passed
    
    def test_entanglement_gate_operations(self, entanglement, test_qubits):
        """Test individual entanglement gate operations"""
        # Test CNOT gate
        entanglement.controller.apply_gate(entanglement.entanglement_gates['CNOT'], test_qubits[0], test_qubits[1])
        measure = entanglement._measure_entanglement(test_qubits[:2])
        assert measure > 0.8
        
        # Test CZ gate
        entanglement.controller.apply_gate(entanglement.entanglement_gates['CZ'], test_qubits[2], test_qubits[3])
        measure = entanglement._measure_entanglement(test_qubits[2:])
        assert measure > 0.8
    
    def test_entanglement_quality(self, entanglement, test_qubits):
        """Test quality of entanglement"""
        result = entanglement.create_ghz_state(test_qubits)
        
        assert result.entanglement_measure > 0.8  # Should have high entanglement
        assert result.verification_passed
        assert result.error_rate < 0.1  # Should have low error rate
        assert result.coherence > 0.8  # Should have high coherence 
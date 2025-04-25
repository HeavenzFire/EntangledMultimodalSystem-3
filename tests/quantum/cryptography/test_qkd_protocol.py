import pytest
import numpy as np
from src.quantum.cryptography.qkd_protocol import QKDProtocol, QKDResult
from src.quantum.core.qubit_control import QubitController, QubitState

class TestQKDProtocol:
    @pytest.fixture
    def qkd_protocol(self):
        controller = QubitController(num_qubits=512)  # Extra qubits for testing
        return QKDProtocol(controller)
    
    def test_bb84_protocol_basic(self, qkd_protocol):
        """Test basic BB84 protocol execution"""
        result = qkd_protocol.bb84_protocol(num_qubits=256)
        
        assert isinstance(result, QKDResult)
        assert len(result.shared_key) > 0
        assert 0 <= result.error_rate <= 1
        assert 'error_rate' in result.security_metrics
        assert 'key_length' in result.security_metrics
        assert 'eavesdropping_probability' in result.security_metrics
    
    def test_bb84_protocol_error_rate(self, qkd_protocol):
        """Test BB84 protocol error rate calculation"""
        # Simulate perfect channel
        result = qkd_protocol.bb84_protocol(num_qubits=256)
        assert result.error_rate < 0.1  # Should be very low in perfect conditions
    
    def test_e91_protocol_basic(self, qkd_protocol):
        """Test basic E91 protocol execution"""
        result = qkd_protocol.e91_protocol(num_qubits=128)
        
        assert isinstance(result, QKDResult)
        assert len(result.shared_key) > 0
        assert 0 <= result.error_rate <= 1
        assert 'entanglement_quality' in result.security_metrics
    
    def test_e91_protocol_entanglement(self, qkd_protocol):
        """Test E91 protocol entanglement quality"""
        result = qkd_protocol.e91_protocol(num_qubits=128)
        assert result.security_metrics['entanglement_quality'] > 0.8  # Should be high for good entanglement
    
    def test_key_length_consistency(self, qkd_protocol):
        """Test that key length is consistent between protocols"""
        bb84_result = qkd_protocol.bb84_protocol(num_qubits=256)
        e91_result = qkd_protocol.e91_protocol(num_qubits=128)
        
        assert len(bb84_result.shared_key) > 0
        assert len(e91_result.shared_key) > 0
        assert bb84_result.security_metrics['key_length'] == len(bb84_result.shared_key)
        assert e91_result.security_metrics['key_length'] == len(e91_result.shared_key)
    
    def test_security_metrics(self, qkd_protocol):
        """Test security metrics calculation"""
        result = qkd_protocol.bb84_protocol(num_qubits=256)
        
        metrics = result.security_metrics
        assert 'error_rate' in metrics
        assert 'key_length' in metrics
        assert 'eavesdropping_probability' in metrics
        assert 0 <= metrics['error_rate'] <= 1
        assert metrics['key_length'] > 0
        assert 0 <= metrics['eavesdropping_probability'] <= 1
    
    def test_protocol_robustness(self, qkd_protocol):
        """Test protocol robustness with different qubit counts"""
        for num_qubits in [64, 128, 256, 512]:
            result = qkd_protocol.bb84_protocol(num_qubits=num_qubits)
            assert result.success
            assert len(result.shared_key) > 0
    
    def test_entanglement_creation(self, qkd_protocol):
        """Test entanglement creation in E91 protocol"""
        result = qkd_protocol.e91_protocol(num_qubits=128)
        assert result.security_metrics['entanglement_quality'] > 0.8
    
    def test_error_handling(self, qkd_protocol):
        """Test error handling in protocols"""
        # Test with invalid qubit count
        with pytest.raises(Exception):
            qkd_protocol.bb84_protocol(num_qubits=0)
        
        # Test with too many qubits
        with pytest.raises(Exception):
            qkd_protocol.bb84_protocol(num_qubits=1000)  # Assuming 512 is max
    
    def test_protocol_comparison(self, qkd_protocol):
        """Compare BB84 and E91 protocols"""
        bb84_result = qkd_protocol.bb84_protocol(num_qubits=256)
        e91_result = qkd_protocol.e91_protocol(num_qubits=128)
        
        # Both protocols should produce valid keys
        assert len(bb84_result.shared_key) > 0
        assert len(e91_result.shared_key) > 0
        
        # E91 should have better entanglement quality
        assert 'entanglement_quality' in e91_result.security_metrics
        assert e91_result.security_metrics['entanglement_quality'] > 0.8 
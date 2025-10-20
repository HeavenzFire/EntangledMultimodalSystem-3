import pytest
import numpy as np
from datetime import datetime
from src.quantum.cryptography.quantum_crypto import (
    QuantumCryptographicSystem,
    QuantumKey
)
from src.quantum.core.quantum_state import QuantumState
from src.quantum.core.nonlinear_processor import NonlinearState

class TestQuantumCryptographicSystem:
    @pytest.fixture
    def crypto_system(self):
        return QuantumCryptographicSystem()
    
    @pytest.fixture
    def test_key(self):
        return QuantumKey(
            amplitude=1.0,
            phase=np.pi/4,
            coherence=0.95,
            error_rate=0.001,
            timestamp=datetime.now(),
            security_level=0.99,
            entanglement_degree=0.8
        )
    
    def test_key_generation(self, crypto_system):
        """Test quantum key generation"""
        key = crypto_system.generate_quantum_key()
        
        assert isinstance(key, QuantumKey)
        assert key.amplitude > 0
        assert 0 <= key.phase <= 2 * np.pi
        assert key.coherence > 0.8
        assert key.error_rate < 0.1
        assert isinstance(key.timestamp, datetime)
        assert key.security_level > 0.8
        assert key.entanglement_degree > 0.6
    
    def test_encryption_decryption(self, crypto_system, test_key):
        """Test encryption and decryption of data"""
        test_data = b"Test quantum encryption"
        
        # Encrypt data
        encrypted_data, metrics = crypto_system.encrypt_data(test_data, test_key)
        
        assert isinstance(encrypted_data, bytes)
        assert encrypted_data != test_data
        assert isinstance(metrics, dict)
        assert 'security_level' in metrics
        assert 'entanglement_degree' in metrics
        assert 'error_rate' in metrics
        assert 'timestamp' in metrics
    
    def test_security_metrics(self, crypto_system, test_key):
        """Test security metrics calculation"""
        metrics = crypto_system.get_security_metrics()
        
        assert isinstance(metrics, dict)
        assert 'current_security_level' in metrics
        assert 'entanglement_degree' in metrics
        assert 'error_rate' in metrics
        assert 'timestamp' in metrics
    
    def test_key_quality(self, crypto_system):
        """Test quality of generated keys"""
        keys = []
        for _ in range(5):
            key = crypto_system.generate_quantum_key()
            keys.append(key)
        
        # Check key quality metrics
        for key in keys:
            assert key.coherence > 0.8
            assert key.error_rate < 0.1
            assert key.security_level > 0.8
            assert key.entanglement_degree > 0.6
    
    def test_error_handling(self, crypto_system):
        """Test error handling"""
        # Test with invalid data
        with pytest.raises(Exception):
            crypto_system.encrypt_data(None, None)
        
        # Test with invalid key
        with pytest.raises(Exception):
            crypto_system.encrypt_data(b"test", None)
    
    def test_key_history(self, crypto_system):
        """Test key history tracking"""
        # Generate multiple keys
        for _ in range(3):
            crypto_system.generate_quantum_key()
        
        assert len(crypto_system.key_history) == 3
        assert all(isinstance(key, QuantumKey) for key in crypto_system.key_history)
    
    def test_security_level_calculation(self, crypto_system, test_key):
        """Test security level calculation"""
        security_level = crypto_system._calculate_security_level(test_key)
        
        assert isinstance(security_level, float)
        assert 0 <= security_level <= 1.0
        assert security_level > 0.8  # Should maintain high security
    
    def test_entanglement_degree_calculation(self, crypto_system, test_key):
        """Test entanglement degree calculation"""
        entanglement_degree = crypto_system._calculate_entanglement_degree(test_key)
        
        assert isinstance(entanglement_degree, float)
        assert 0 <= entanglement_degree <= 1.0
        assert entanglement_degree > 0.6  # Should maintain good entanglement
    
    def test_quantum_state_conversion(self, crypto_system):
        """Test conversion between classical data and quantum states"""
        test_data = b"Test conversion"
        
        # Convert to quantum state
        quantum_state = crypto_system._data_to_quantum_state(np.frombuffer(test_data, dtype=np.uint8))
        
        assert isinstance(quantum_state, QuantumState)
        assert quantum_state.amplitude > 0
        assert 0 <= quantum_state.phase <= 2 * np.pi
        
        # Convert back to classical data
        classical_data = crypto_system._quantum_state_to_data(quantum_state)
        
        assert isinstance(classical_data, bytes)
        assert len(classical_data) > 0
    
    def test_quantum_encryption(self, crypto_system, test_key):
        """Test quantum encryption process"""
        test_state = QuantumState(
            amplitude=1.0,
            phase=np.pi/4,
            error_rate=0.001
        )
        
        encrypted_state = crypto_system._apply_quantum_encryption(test_state, test_key)
        
        assert isinstance(encrypted_state, QuantumState)
        assert encrypted_state.amplitude != test_state.amplitude
        assert encrypted_state.phase != test_state.phase
        assert encrypted_state.error_rate < 0.1
    
    def test_encryption_robustness(self, crypto_system):
        """Test encryption robustness"""
        test_data = b"Test robustness"
        results = []
        
        for _ in range(5):
            key = crypto_system.generate_quantum_key()
            encrypted_data, _ = crypto_system.encrypt_data(test_data, key)
            results.append(encrypted_data)
        
        # Check that all encryptions are different
        for i in range(1, len(results)):
            assert results[i] != results[0]
    
    def test_security_metrics_history(self, crypto_system):
        """Test security metrics history"""
        # Perform multiple encryptions
        for _ in range(3):
            key = crypto_system.generate_quantum_key()
            crypto_system.encrypt_data(b"test", key)
        
        assert len(crypto_system.security_metrics) == 3
        assert all(isinstance(metrics, dict) for metrics in crypto_system.security_metrics)
    
    def test_key_validation(self, crypto_system):
        """Test key validation"""
        # Generate a valid key
        valid_key = crypto_system.generate_quantum_key()
        assert crypto_system._validate_key(valid_key)
        
        # Test with invalid key
        invalid_key = QuantumKey(
            amplitude=0.0,  # Invalid amplitude
            phase=np.pi/4,
            coherence=0.95,
            error_rate=0.001,
            timestamp=datetime.now(),
            security_level=0.99,
            entanglement_degree=0.8
        )
        assert not crypto_system._validate_key(invalid_key) 
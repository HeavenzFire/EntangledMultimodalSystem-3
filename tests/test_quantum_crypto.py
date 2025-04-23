import pytest
import numpy as np
from datetime import datetime
from src.quantum.cryptography.quantum_crypto import (
    QuantumCryptographicSystem,
    QuantumKey
)

@pytest.fixture
def quantum_crypto():
    return QuantumCryptographicSystem()

def test_quantum_key_generation(quantum_crypto):
    # Generate quantum key
    key = quantum_crypto.generate_quantum_key()
    
    # Verify key properties
    assert isinstance(key, QuantumKey)
    assert 0 <= key.amplitude <= 2.0
    assert 0 <= key.phase <= 2 * np.pi
    assert 0 <= key.coherence <= 1.0
    assert 0 <= key.error_rate <= 1.0
    assert isinstance(key.timestamp, datetime)
    assert 0 <= key.security_level <= 1.0
    assert 0 <= key.entanglement_degree <= 1.0
    
    # Verify key history
    assert len(quantum_crypto.key_history) == 1

def test_data_encryption(quantum_crypto):
    # Generate key
    key = quantum_crypto.generate_quantum_key()
    
    # Test data
    test_data = b"Hello, Quantum World!"
    
    # Encrypt data
    encrypted_data, metrics = quantum_crypto.encrypt_data(test_data, key)
    
    # Verify encryption results
    assert isinstance(encrypted_data, bytes)
    assert len(encrypted_data) > 0
    assert isinstance(metrics, dict)
    assert 'security_level' in metrics
    assert 'entanglement_degree' in metrics
    assert 'error_rate' in metrics
    assert isinstance(metrics['timestamp'], datetime)
    
    # Verify security metrics
    assert len(quantum_crypto.security_metrics) == 1

def test_security_metrics(quantum_crypto):
    # Generate key and encrypt data
    key = quantum_crypto.generate_quantum_key()
    test_data = b"Test data"
    quantum_crypto.encrypt_data(test_data, key)
    
    # Get security metrics
    metrics = quantum_crypto.get_security_metrics()
    
    # Verify metrics
    assert isinstance(metrics, dict)
    assert 'current_security_level' in metrics
    assert 'entanglement_degree' in metrics
    assert 'error_rate' in metrics
    assert isinstance(metrics['timestamp'], datetime)

def test_quantum_state_conversion(quantum_crypto):
    # Test data
    test_data = b"Test"
    
    # Convert to quantum state
    quantum_state = quantum_crypto._data_to_quantum_state(np.frombuffer(test_data, dtype=np.uint8))
    
    # Verify quantum state
    assert isinstance(quantum_state.amplitude, float)
    assert isinstance(quantum_state.phase, float)
    assert isinstance(quantum_state.error_rate, float)
    
    # Convert back to data
    converted_data = quantum_crypto._quantum_state_to_data(quantum_state)
    
    # Verify conversion
    assert isinstance(converted_data, bytes)
    assert len(converted_data) == 2  # Amplitude and phase data

def test_quantum_encryption(quantum_crypto):
    # Create test state and key
    test_state = quantum_crypto._data_to_quantum_state(np.array([1, 2, 3], dtype=np.uint8))
    key = quantum_crypto.generate_quantum_key()
    
    # Apply quantum encryption
    encrypted_state = quantum_crypto._apply_quantum_encryption(test_state, key)
    
    # Verify encrypted state
    assert isinstance(encrypted_state.amplitude, float)
    assert isinstance(encrypted_state.phase, float)
    assert isinstance(encrypted_state.error_rate, float)
    assert 0 <= encrypted_state.error_rate <= 0.1 
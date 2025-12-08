import pytest
import numpy as np
from src.quantum.security import (
    QuantumSecurityFramework,
    SecurityConfig,
    SecurityLevel
)

def test_security_framework_initialization():
    """Test initialization of security framework"""
    security = QuantumSecurityFramework()
    assert security.config.security_level == SecurityLevel.HYPER_QUANTUM
    assert len(security.quantum_key) == 512 // 8
    assert security.merkaba_field.shape == (64, 64)

def test_quantum_key_generation():
    """Test quantum-safe key generation"""
    security = QuantumSecurityFramework()
    key1 = security._generate_quantum_key()
    key2 = security._generate_quantum_key()
    
    # Keys should be different due to entropy
    assert key1 != key2
    assert len(key1) == 512 // 8

def test_quantum_transformation():
    """Test quantum-safe transformation"""
    security = QuantumSecurityFramework()
    test_data = b"test data"
    transformed = security._apply_quantum_transformation(test_data)
    
    assert len(transformed) == len(test_data)
    assert transformed != test_data

def test_merkaba_field_generation():
    """Test Merkaba field generation"""
    security = QuantumSecurityFramework()
    field = security._generate_merkaba_field()
    
    assert field.shape == (64, 64)
    assert np.all(np.abs(field) <= 1.0)
    assert np.any(field != 0)

def test_encryption_decryption():
    """Test quantum-safe encryption and decryption"""
    security = QuantumSecurityFramework()
    test_data = b"secret quantum data"
    
    # Encrypt data
    encrypted_data, tag = security.encrypt_data(test_data)
    assert encrypted_data != test_data
    
    # Decrypt data
    decrypted_data = security.decrypt_data(encrypted_data, tag, security.salt[:16])
    assert decrypted_data == test_data

def test_quantum_integrity():
    """Test quantum integrity verification"""
    security = QuantumSecurityFramework()
    test_data = b"data to verify"
    
    # Verify integrity
    assert security.verify_quantum_integrity(test_data)
    
    # Tamper with data
    tampered_data = test_data + b"tampered"
    assert not security.verify_quantum_integrity(tampered_data)

def test_security_metrics():
    """Test security metrics generation"""
    security = QuantumSecurityFramework()
    metrics = security.get_security_metrics()
    
    assert metrics["security_level"] == "hyper_quantum"
    assert metrics["key_strength"] == 512
    assert metrics["merkaba_field_size"] == (64, 64)
    assert isinstance(metrics["quantum_integrity"], bool)

def test_custom_security_config():
    """Test custom security configuration"""
    config = SecurityConfig(
        security_level=SecurityLevel.POST_QUANTUM,
        key_length=256,
        salt_length=16,
        iterations=50000
    )
    security = QuantumSecurityFramework(config)
    
    assert security.config.security_level == SecurityLevel.POST_QUANTUM
    assert len(security.quantum_key) == 256 // 8
    assert len(security.salt) == 16

def test_quantum_hash_calculation():
    """Test quantum-resistant hash calculation"""
    security = QuantumSecurityFramework()
    test_data = b"data to hash"
    hash_result = security._calculate_quantum_hash(test_data)
    
    assert isinstance(hash_result, np.ndarray)
    assert len(hash_result) > 0

def test_merkaba_alignment():
    """Test Merkaba field alignment verification"""
    security = QuantumSecurityFramework()
    test_data = b"data to align"
    hash_data = security._calculate_quantum_hash(test_data)
    
    # Verify alignment
    assert security._verify_merkaba_alignment(hash_data)
    
    # Test with random data
    random_data = np.random.randint(0, 256, size=hash_data.shape, dtype=np.uint8)
    assert not security._verify_merkaba_alignment(random_data)

def test_security_levels():
    """Test different security levels"""
    for level in SecurityLevel:
        config = SecurityConfig(security_level=level)
        security = QuantumSecurityFramework(config)
        
        assert security.config.security_level == level
        metrics = security.get_security_metrics()
        assert metrics["security_level"] == level.value 
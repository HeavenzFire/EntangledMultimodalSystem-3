import pytest
import numpy as np
from src.quantum.validation.validator import (
    QuantumValidator,
    ValidationStatus,
    ValidationResult
)

def test_validator_initialization():
    """Test validator initialization"""
    validator = QuantumValidator()
    assert len(validator.validation_history) == 0
    assert 'security' in validator.thresholds
    assert 'synthesis' in validator.thresholds
    assert 'torus' in validator.thresholds

def test_security_validation():
    """Test security validation"""
    validator = QuantumValidator()
    
    # Test valid metrics
    valid_metrics = {
        'key_strength': 512,
        'entropy': 0.9,
        'coherence': 0.95
    }
    result = validator.validate_security(valid_metrics)
    assert result.status == ValidationStatus.VALID
    
    # Test warning metrics
    warning_metrics = {
        'key_strength': 512,
        'entropy': 0.7,  # Below threshold
        'coherence': 0.95
    }
    result = validator.validate_security(warning_metrics)
    assert result.status == ValidationStatus.WARNING
    
    # Test error metrics
    error_metrics = {
        'key_strength': 128,  # Below threshold
        'entropy': 0.9,
        'coherence': 0.95
    }
    result = validator.validate_security(error_metrics)
    assert result.status == ValidationStatus.ERROR
    
    # Test critical metrics
    critical_metrics = {
        'key_strength': 512,
        'entropy': 0.9,
        'coherence': 0.7  # Below threshold
    }
    result = validator.validate_security(critical_metrics)
    assert result.status == ValidationStatus.CRITICAL

def test_synthesis_validation():
    """Test synthesis validation"""
    validator = QuantumValidator()
    
    # Test valid metrics
    valid_metrics = {
        'state_fidelity': 0.98,
        'resonance': 0.9,
        'harmony': 0.95
    }
    result = validator.validate_synthesis(valid_metrics)
    assert result.status == ValidationStatus.VALID
    
    # Test warning metrics
    warning_metrics = {
        'state_fidelity': 0.98,
        'resonance': 0.8,  # Below threshold
        'harmony': 0.95
    }
    result = validator.validate_synthesis(warning_metrics)
    assert result.status == ValidationStatus.WARNING
    
    # Test error metrics
    error_metrics = {
        'state_fidelity': 0.9,  # Below threshold
        'resonance': 0.9,
        'harmony': 0.95
    }
    result = validator.validate_synthesis(error_metrics)
    assert result.status == ValidationStatus.ERROR
    
    # Test critical metrics
    critical_metrics = {
        'state_fidelity': 0.98,
        'resonance': 0.9,
        'harmony': 0.8  # Below threshold
    }
    result = validator.validate_synthesis(critical_metrics)
    assert result.status == ValidationStatus.CRITICAL

def test_torus_validation():
    """Test torus validation"""
    validator = QuantumValidator()
    
    # Test valid metrics
    valid_metrics = {
        'field_stability': 0.9,
        'alignment': 0.9,
        'coherence': 0.95
    }
    result = validator.validate_torus(valid_metrics)
    assert result.status == ValidationStatus.VALID
    
    # Test warning metrics
    warning_metrics = {
        'field_stability': 0.9,
        'alignment': 0.8,  # Below threshold
        'coherence': 0.95
    }
    result = validator.validate_torus(warning_metrics)
    assert result.status == ValidationStatus.WARNING
    
    # Test error metrics
    error_metrics = {
        'field_stability': 0.7,  # Below threshold
        'alignment': 0.9,
        'coherence': 0.95
    }
    result = validator.validate_torus(error_metrics)
    assert result.status == ValidationStatus.ERROR
    
    # Test critical metrics
    critical_metrics = {
        'field_stability': 0.9,
        'alignment': 0.9,
        'coherence': 0.8  # Below threshold
    }
    result = validator.validate_torus(critical_metrics)
    assert result.status == ValidationStatus.CRITICAL

def test_system_validation():
    """Test system-wide validation"""
    validator = QuantumValidator()
    
    # Test valid system
    valid_metrics = {
        'security': {
            'key_strength': 512,
            'entropy': 0.9,
            'coherence': 0.95
        },
        'synthesis': {
            'state_fidelity': 0.98,
            'resonance': 0.9,
            'harmony': 0.95
        },
        'torus': {
            'field_stability': 0.9,
            'alignment': 0.9,
            'coherence': 0.95
        }
    }
    results = validator.validate_system(valid_metrics)
    assert all(r.status == ValidationStatus.VALID for r in results.values())
    
    # Test system with warnings
    warning_metrics = {
        'security': {
            'key_strength': 512,
            'entropy': 0.7,  # Below threshold
            'coherence': 0.95
        },
        'synthesis': {
            'state_fidelity': 0.98,
            'resonance': 0.8,  # Below threshold
            'harmony': 0.95
        },
        'torus': {
            'field_stability': 0.9,
            'alignment': 0.8,  # Below threshold
            'coherence': 0.95
        }
    }
    results = validator.validate_system(warning_metrics)
    assert all(r.status == ValidationStatus.WARNING for r in results.values())

def test_validation_history():
    """Test validation history management"""
    validator = QuantumValidator()
    
    # Add some validations
    metrics = {
        'security': {'key_strength': 512},
        'synthesis': {'state_fidelity': 0.98},
        'torus': {'field_stability': 0.9}
    }
    
    for _ in range(5):
        validator.validate_system(metrics)
    
    assert len(validator.validation_history) == 15  # 5 validations * 3 components
    
    # Test history limit
    for _ in range(1000):
        validator.validate_system(metrics)
    
    assert len(validator.validation_history) == 1000

def test_system_status():
    """Test system status determination"""
    validator = QuantumValidator()
    
    # Test empty history
    assert validator.get_system_status() == ValidationStatus.VALID
    
    # Add some validations
    metrics = {
        'security': {'key_strength': 512, 'entropy': 0.9, 'coherence': 0.95},
        'synthesis': {'state_fidelity': 0.98, 'resonance': 0.9, 'harmony': 0.95},
        'torus': {'field_stability': 0.9, 'alignment': 0.9, 'coherence': 0.95}
    }
    
    validator.validate_system(metrics)
    assert validator.get_system_status() == ValidationStatus.VALID
    
    # Add critical validation
    critical_metrics = {
        'security': {'key_strength': 512, 'entropy': 0.9, 'coherence': 0.7},
        'synthesis': {'state_fidelity': 0.98, 'resonance': 0.9, 'harmony': 0.95},
        'torus': {'field_stability': 0.9, 'alignment': 0.9, 'coherence': 0.95}
    }
    
    validator.validate_system(critical_metrics)
    assert validator.get_system_status() == ValidationStatus.CRITICAL 
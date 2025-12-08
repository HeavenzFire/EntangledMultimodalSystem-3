import pytest
import numpy as np
from src.quantum.validation.quantum_classical_validation import (
    QuantumClassicalValidator,
    ValidationResult
)
from qiskit import AerSimulator
from src.quantum.security_middleware import SecurityMiddleware
from src.quantum.error_correction import QuantumErrorCorrection
from src.quantum.quantum_threading import QuantumThreadManager

def test_quantum_processing_validation():
    validator = QuantumClassicalValidator()
    result = validator.validate_quantum_processing()
    
    assert isinstance(result, ValidationResult)
    assert result.component == "quantum_processing"
    assert result.timestamp != ""
    assert result.success
    assert result.metrics["fidelity"] > 0.9
    assert result.metrics["coherence_time"] > 0
    assert result.metrics["circuit_depth"] > 0
    assert result.metrics["qubit_count"] == 2

def test_classical_processing_validation():
    validator = QuantumClassicalValidator()
    result = validator.validate_classical_processing()
    
    assert isinstance(result, ValidationResult)
    assert result.component == "classical_processing"
    assert result.timestamp != ""
    assert result.success
    assert result.metrics["accuracy"] > 0.95
    assert result.metrics["execution_time"] < 1.0
    assert result.metrics["matrix_size"] == 8192
    assert isinstance(result.metrics["hardware_accelerated"], bool)

def test_ethical_framework_validation():
    validator = QuantumClassicalValidator()
    result = validator.validate_ethical_framework()
    
    assert isinstance(result, ValidationResult)
    assert result.component == "ethical_framework"
    assert result.timestamp != ""
    assert result.success
    assert result.metrics["compliance_score"] > 0.8
    assert result.metrics["transparency_score"] > 0.7
    assert isinstance(result.metrics["optimization_converged"], bool)
    assert isinstance(result.metrics["optimal_value"], float)

def test_radiation_detection_validation():
    validator = QuantumClassicalValidator()
    result = validator.validate_radiation_detection()
    
    assert isinstance(result, ValidationResult)
    assert result.component == "radiation_detection"
    assert result.timestamp != ""
    assert result.success
    assert result.metrics["sensitivity"] > 0.9
    assert result.metrics["specificity"] > 0.9
    assert result.metrics["num_samples"] == 1000

def test_integrated_validation():
    validator = QuantumClassicalValidator()
    result = validator.validate_integrated_system()
    
    assert isinstance(result, ValidationResult)
    assert result.component == "integrated_system"
    assert result.timestamp != ""
    assert result.success
    assert all(score > 0.8 for score in result.metrics.values() if isinstance(score, float) and score <= 1.0)
    assert "system_uptime" in result.metrics

def test_security_validation():
    validator = QuantumClassicalValidator()
    result = validator.validate_security()
    
    assert isinstance(result, ValidationResult)
    assert result.component == "security"
    assert result.timestamp != ""
    assert result.success
    assert result.metrics["encryption_strength"] > 0.95
    assert result.metrics["qkd_success_rate"] > 0.9
    assert result.metrics["intrusion_detection_rate"] > 0.95

def test_error_correction_validation():
    validator = QuantumClassicalValidator()
    result = validator.validate_error_correction()
    
    assert isinstance(result, ValidationResult)
    assert result.component == "error_correction"
    assert result.timestamp != ""
    assert result.success
    assert result.metrics["correction_rate"] > 0.95
    assert result.metrics["logical_error_rate"] < 0.01
    assert result.metrics["error_rate"] == 0.1

def test_quantum_threading_validation():
    validator = QuantumClassicalValidator()
    result = validator.validate_quantum_threading()
    
    assert isinstance(result, ValidationResult)
    assert result.component == "quantum_threading"
    assert result.timestamp != ""
    assert result.success
    assert result.metrics["thread_success_rate"] > 0.95
    assert result.metrics["resource_efficiency"] > 0.9
    assert result.metrics["sync_success_rate"] > 0.95
    assert result.metrics["thread_count"] == 4

def test_validation_error_handling():
    validator = QuantumClassicalValidator()
    
    # Test quantum processing with invalid circuit
    validator.simulator = None  # Force error
    result = validator.validate_quantum_processing()
    assert not result.success
    assert "Quantum validation failed" in result.message
    
    # Test classical processing with invalid matrix
    validator.simulator = AerSimulator(method='statevector')  # Restore simulator
    validator.rng = None  # Force error
    result = validator.validate_classical_processing()
    assert not result.success
    assert "Classical validation failed" in result.message
    
    # Test security with invalid middleware
    validator.security_middleware = None  # Force error
    result = validator.validate_security()
    assert not result.success
    assert "Security validation failed" in result.message
    
    # Test error correction with invalid circuit
    validator.security_middleware = SecurityMiddleware()  # Restore middleware
    validator.error_correction = None  # Force error
    result = validator.validate_error_correction()
    assert not result.success
    assert "Error correction validation failed" in result.message
    
    # Test quantum threading with invalid manager
    validator.error_correction = QuantumErrorCorrection()  # Restore error correction
    validator.thread_manager = None  # Force error
    result = validator.validate_quantum_threading()
    assert not result.success
    assert "Quantum threading validation failed" in result.message 
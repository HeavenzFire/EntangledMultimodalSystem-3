import pytest
import numpy as np
from src.quantum.validation.security_validator import QuantumSecurityValidator, SecurityMetrics
from src.quantum.validation.error_correction import SurfaceCodeValidator, ErrorCorrectionMetrics
from src.quantum.validation.thread_manager import HybridThreadController, ThreadMetrics
import time

def test_security_validation():
    validator = QuantumSecurityValidator()
    metrics = validator.run_security_sweep()
    
    assert isinstance(metrics, SecurityMetrics)
    assert metrics.qkd_success_rate > 0.95
    assert metrics.encryption_strength >= 256
    assert isinstance(metrics.intrusion_attempts, int)
    assert metrics.timestamp != ""

def test_error_correction_validation():
    validator = SurfaceCodeValidator(distance=7)
    metrics = validator.inject_errors(cycles=1000)
    
    assert isinstance(metrics, ErrorCorrectionMetrics)
    assert metrics.physical_error_rate == 1e-3
    assert metrics.logical_error_rate < 1e-5
    assert metrics.correction_rate > 0.95
    assert metrics.timestamp != ""
    
    # Test Bell state preservation
    assert validator.validate_bell_state()

def test_thread_management_validation():
    controller = HybridThreadController(max_workers=4)
    
    # Test thread management
    success_rate = controller.test_thread_management(num_threads=4)
    assert success_rate > 0.95
    
    # Test resource efficiency
    efficiency = controller.measure_resource_efficiency()
    assert 0 <= efficiency <= 1
    
    # Test synchronization
    sync_rate = controller.test_synchronization()
    assert sync_rate == 1.0

def test_parallel_execution():
    controller = HybridThreadController(max_workers=4)
    
    # Create test tasks
    tasks = [
        lambda: time.sleep(0.1),
        lambda: np.random.random(),
        lambda: "test",
        lambda: {"key": "value"}
    ]
    
    # Execute tasks
    results = controller.execute_parallel(tasks)
    assert len(results) == 4
    assert all(r is not None for r in results)

def test_system_recovery():
    # Test security recovery
    security_validator = QuantumSecurityValidator()
    security_validator.simulator = None  # Force error
    metrics = security_validator.run_security_sweep()
    assert metrics.qkd_success_rate == 0.0
    assert metrics.encryption_strength == 0
    
    # Test error correction recovery
    error_validator = SurfaceCodeValidator()
    error_validator.simulator = None  # Force error
    metrics = error_validator.inject_errors(cycles=100)
    assert metrics.correction_rate == 0.0
    
    # Test thread management recovery
    controller = HybridThreadController()
    controller.thread_pool = None  # Force error
    success_rate = controller.test_thread_management(num_threads=4)
    assert success_rate == 0.0

def test_performance_metrics():
    controller = HybridThreadController()
    status = controller.get_thread_status()
    
    assert isinstance(status, dict)
    assert "qubit_utilization" in status
    assert "classical_throughput" in status
    assert "synchronization_latency" in status
    assert "timestamp" in status
    
    # Verify performance thresholds
    assert status["synchronization_latency"] < 0.47  # 470ms threshold

def test_integrated_validation():
    # Initialize all validators
    security_validator = QuantumSecurityValidator()
    error_validator = SurfaceCodeValidator()
    thread_controller = HybridThreadController()
    
    # Run security validation
    security_metrics = security_validator.run_security_sweep()
    assert security_metrics.qkd_success_rate > 0.95
    
    # Run error correction validation
    error_metrics = error_validator.inject_errors(cycles=1000)
    assert error_metrics.correction_rate > 0.95
    
    # Run thread management validation
    thread_success = thread_controller.test_thread_management(num_threads=4)
    assert thread_success > 0.95
    
    # Verify system-wide performance
    thread_status = thread_controller.get_thread_status()
    assert thread_status["synchronization_latency"] < 0.47

def test_resource_balancing():
    controller = HybridThreadController()
    
    # Test quantum-classical resource balance
    metrics = controller.thread_pool.get_metrics()
    quantum_ratio = metrics.qubit_utilization / (metrics.qubit_utilization + metrics.classical_throughput)
    assert 0.4 <= quantum_ratio <= 0.6  # 40-60% quantum resource allocation 
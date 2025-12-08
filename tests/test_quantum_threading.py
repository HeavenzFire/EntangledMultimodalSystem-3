import pytest
import numpy as np
from src.quantum.quantum_threading import QuantumThread, QuantumThreadingBridge
from src.quantum.error_correction import QuantumErrorCorrection
import time
import logging

@pytest.fixture
def quantum_thread():
    return QuantumThread("test_thread", capacity=4)

@pytest.fixture
def quantum_bridge():
    return QuantumThreadingBridge(num_threads=4, thread_capacity=4)

def test_thread_initialization(quantum_thread):
    """Test quantum thread initialization."""
    state = quantum_thread.get_state()
    assert np.allclose(np.abs(state), 1/np.sqrt(4))
    assert quantum_thread.error_correction is True
    
def test_gate_application(quantum_thread):
    """Test quantum gate application."""
    # Create Hadamard gate
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    
    # Apply gate
    quantum_thread.apply_gate(H)
    
    # Check state evolution
    state = quantum_thread.get_state()
    assert not np.allclose(state, np.ones(4)/2)
    
def test_measurement(quantum_thread):
    """Test quantum measurement."""
    result = quantum_thread.measure()
    assert 0 <= result < 4
    
    # Check state collapse
    state = quantum_thread.get_state()
    assert np.sum(np.abs(state)) == 1
    
def test_error_correction(quantum_thread):
    """Test error correction capabilities."""
    error_info = quantum_thread.get_error_info()
    assert error_info is not None
    assert "code_type" in error_info
    assert "distance" in error_info
    assert "threshold" in error_info
    
def test_thread_coupling(quantum_bridge):
    """Test quantum thread coupling."""
    thread1 = quantum_bridge.create_thread("thread1")
    thread2 = quantum_bridge.create_thread("thread2")
    
    quantum_bridge.apply_coupling(thread1.thread_id, thread2.thread_id)
    
    # Check coupling effect
    state1 = thread1.get_state()
    state2 = thread2.get_state()
    assert not np.allclose(state1, state2)
    
def test_interference(quantum_bridge):
    """Test quantum interference."""
    threads = [quantum_bridge.create_thread(f"thread_{i}") for i in range(3)]
    thread_ids = [t.thread_id for t in threads]
    
    quantum_bridge.create_interference(thread_ids)
    
    # Check interference pattern
    states = [t.get_state() for t in threads]
    for i in range(len(states)-1):
        assert not np.allclose(states[i], states[i+1])
        
@pytest.mark.benchmark
def test_performance_benchmark(quantum_bridge):
    """Benchmark quantum threading performance."""
    # Create test threads
    num_threads = 100
    threads = [quantum_bridge.create_thread(f"benchmark_{i}") for i in range(num_threads)]
    
    # Measure initialization time
    start_time = time.time()
    for thread in threads:
        thread.initialize_thread()
    init_time = time.time() - start_time
    
    # Measure gate application time
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    start_time = time.time()
    for thread in threads:
        thread.apply_gate(H)
    gate_time = time.time() - start_time
    
    # Measure coupling time
    start_time = time.time()
    for i in range(0, num_threads-1, 2):
        quantum_bridge.apply_coupling(
            threads[i].thread_id,
            threads[i+1].thread_id
        )
    coupling_time = time.time() - start_time
    
    # Measure interference time
    start_time = time.time()
    quantum_bridge.create_interference([t.thread_id for t in threads])
    interference_time = time.time() - start_time
    
    # Log benchmark results
    logging.info(f"Performance Benchmark Results:")
    logging.info(f"Thread Initialization: {init_time:.4f}s")
    logging.info(f"Gate Application: {gate_time:.4f}s")
    logging.info(f"Thread Coupling: {coupling_time:.4f}s")
    logging.info(f"Interference Creation: {interference_time:.4f}s")
    
    # Assert performance requirements
    assert init_time < 1.0  # Initialization should be fast
    assert gate_time < 0.5  # Gate operations should be efficient
    assert coupling_time < 0.3  # Coupling should be quick
    assert interference_time < 0.2  # Interference should be fast
    
@pytest.mark.benchmark
def test_error_correction_benchmark(quantum_bridge):
    """Benchmark error correction performance."""
    # Create test thread with error correction
    thread = quantum_bridge.create_thread("error_correction_test")
    
    # Measure error detection time
    start_time = time.time()
    error_info = thread.get_error_info()
    detection_time = time.time() - start_time
    
    # Measure error correction time
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    start_time = time.time()
    thread.apply_gate(H)
    correction_time = time.time() - start_time
    
    # Log benchmark results
    logging.info(f"Error Correction Benchmark Results:")
    logging.info(f"Error Detection: {detection_time:.4f}s")
    logging.info(f"Error Correction: {correction_time:.4f}s")
    
    # Assert performance requirements
    assert detection_time < 0.1  # Error detection should be fast
    assert correction_time < 0.2  # Error correction should be efficient
    
def test_quantum_pipeline(quantum_bridge):
    """Test quantum operation pipeline."""
    # Define quantum operations
    def hadamard_transform(state):
        H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        return np.kron(H, H)
        
    def phase_shift(state, angle=np.pi/4):
        P = np.array([[1, 0], [0, np.exp(1j * angle)]])
        return np.kron(P, P)
        
    # Create thread and execute pipeline
    thread = quantum_bridge.create_thread("pipeline_test")
    pipeline = [
        hadamard_transform,
        lambda s: phase_shift(s, np.pi/4),
        hadamard_transform
    ]
    
    # Execute pipeline
    result = quantum_bridge.execute_pipeline(pipeline, thread.thread_id)
    assert result is not None
    
    # Check final state
    final_state = thread.get_state()
    assert not np.allclose(final_state, np.ones(4)/2) 
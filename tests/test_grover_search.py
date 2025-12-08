import pytest
import numpy as np
from core.quantum.algorithms.quantum_algorithms import GroverSearch
from core.quantum.qubit_control import QubitController

def test_grover_search_initialization():
    """Test initialization of Grover's search algorithm"""
    num_qubits = 3
    oracle = lambda n: np.eye(2**n)  # Identity oracle for testing
    
    grover = GroverSearch(num_qubits, oracle)
    assert grover.num_qubits == num_qubits
    assert grover.iterations == int(np.pi/4 * np.sqrt(2**num_qubits))

def test_grover_search_simple_case():
    """Test Grover's search with a simple oracle"""
    num_qubits = 2
    target_state = 3  # Binary: 11
    
    def simple_oracle(n):
        # Create oracle that marks state |11>
        oracle_matrix = np.eye(2**n)
        oracle_matrix[target_state, target_state] = -1
        return oracle_matrix
    
    grover = GroverSearch(num_qubits, simple_oracle)
    grover.run()
    
    # The algorithm should find the target state with high probability
    assert grover.result == target_state

def test_grover_search_error_handling():
    """Test error handling in Grover's search"""
    num_qubits = 2
    
    def invalid_oracle(n):
        # Return invalid oracle matrix
        return np.ones((2**n, 2**n))
    
    grover = GroverSearch(num_qubits, invalid_oracle)
    
    with pytest.raises(Exception):
        grover.run()

def test_grover_search_large_case():
    """Test Grover's search with larger number of qubits"""
    num_qubits = 4
    target_state = 10  # Binary: 1010
    
    def large_oracle(n):
        oracle_matrix = np.eye(2**n)
        oracle_matrix[target_state, target_state] = -1
        return oracle_matrix
    
    grover = GroverSearch(num_qubits, large_oracle)
    grover.run()
    
    # The algorithm should find the target state with high probability
    assert grover.result == target_state

def test_grover_search_multiple_runs():
    """Test consistency of Grover's search across multiple runs"""
    num_qubits = 3
    target_state = 5  # Binary: 101
    
    def consistent_oracle(n):
        oracle_matrix = np.eye(2**n)
        oracle_matrix[target_state, target_state] = -1
        return oracle_matrix
    
    results = []
    for _ in range(10):
        grover = GroverSearch(num_qubits, consistent_oracle)
        grover.run()
        results.append(grover.result)
    
    # The algorithm should consistently find the target state
    assert all(result == target_state for result in results) 
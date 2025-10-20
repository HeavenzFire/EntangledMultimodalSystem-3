import pytest
import numpy as np
from src.quantum.algorithms.quantum_algorithms import (
    QuantumAlgorithm,
    AlgorithmResult
)
from src.quantum.core.quantum_circuit import QuantumCircuit
from src.quantum.core.qubit_control import QubitController

class TestQuantumAlgorithms:
    @pytest.fixture
    def algorithm(self):
        return QuantumAlgorithm()
    
    @pytest.fixture
    def test_circuit(self):
        circuit = QuantumCircuit(num_qubits=4)
        return circuit
    
    def test_quantum_fourier_transform(self, algorithm, test_circuit):
        """Test quantum Fourier transform"""
        result = algorithm.quantum_fourier_transform(test_circuit)
        
        assert isinstance(result, AlgorithmResult)
        assert result.success
        assert isinstance(result.final_state, np.ndarray)
        assert result.final_state.shape == (2**4,)  # 4 qubits
        assert np.isclose(np.sum(np.abs(result.final_state)**2), 1.0)
    
    def test_phase_estimation(self, algorithm, test_circuit):
        """Test quantum phase estimation"""
        # Create a unitary operator (example: rotation)
        unitary = np.array([[np.cos(np.pi/4), -np.sin(np.pi/4)],
                          [np.sin(np.pi/4), np.cos(np.pi/4)]])
        
        result = algorithm.phase_estimation(test_circuit, unitary)
        
        assert isinstance(result, AlgorithmResult)
        assert result.success
        assert isinstance(result.estimated_phase, float)
        assert 0 <= result.estimated_phase <= 2 * np.pi
        assert result.error_rate < 0.1
    
    def test_grover_search(self, algorithm, test_circuit):
        """Test Grover's search algorithm"""
        # Define a simple oracle (example: searching for |11⟩)
        oracle = lambda x: 1 if x == 3 else 0  # |11⟩ is state 3
        
        result = algorithm.grover_search(test_circuit, oracle)
        
        assert isinstance(result, AlgorithmResult)
        assert result.success
        assert isinstance(result.solution, int)
        assert 0 <= result.solution < 2**4
        assert result.iterations > 0
    
    def test_shor_factorization(self, algorithm, test_circuit):
        """Test Shor's factorization algorithm"""
        N = 15  # Example number to factor
        
        result = algorithm.shor_factorization(N)
        
        assert isinstance(result, AlgorithmResult)
        assert result.success
        assert isinstance(result.factors, tuple)
        assert len(result.factors) == 2
        assert result.factors[0] * result.factors[1] == N
    
    def test_quantum_amplitude_amplification(self, algorithm, test_circuit):
        """Test quantum amplitude amplification"""
        # Define a simple marking function
        marking_function = lambda x: 1 if x == 0 else 0  # Mark |00⟩ state
        
        result = algorithm.amplitude_amplification(test_circuit, marking_function)
        
        assert isinstance(result, AlgorithmResult)
        assert result.success
        assert result.amplification_factor > 1.0
        assert result.error_rate < 0.1
    
    def test_quantum_walk(self, algorithm, test_circuit):
        """Test quantum walk algorithm"""
        # Define a simple graph adjacency matrix
        adjacency_matrix = np.array([[0, 1, 1, 0],
                                   [1, 0, 1, 1],
                                   [1, 1, 0, 1],
                                   [0, 1, 1, 0]])
        
        result = algorithm.quantum_walk(test_circuit, adjacency_matrix)
        
        assert isinstance(result, AlgorithmResult)
        assert result.success
        assert isinstance(result.probability_distribution, np.ndarray)
        assert np.isclose(np.sum(result.probability_distribution), 1.0)
    
    def test_algorithm_robustness(self, algorithm, test_circuit):
        """Test algorithm robustness"""
        # Test multiple algorithm runs
        results = []
        for _ in range(5):
            result = algorithm.quantum_fourier_transform(test_circuit)
            results.append(result.final_state)
        
        # Check consistency of results
        for i in range(1, len(results)):
            assert np.allclose(results[i], results[0], atol=1e-6)
    
    def test_error_handling(self, algorithm):
        """Test error handling"""
        # Test with invalid input
        with pytest.raises(Exception):
            algorithm.quantum_fourier_transform(None)
        
        # Test with invalid parameters
        with pytest.raises(Exception):
            algorithm.phase_estimation(None, None)
    
    def test_algorithm_quality(self, algorithm, test_circuit):
        """Test algorithm quality"""
        result = algorithm.quantum_fourier_transform(test_circuit)
        
        assert result.error_rate < 0.1  # Should have low error rate
        assert result.coherence > 0.8  # Should maintain high coherence
        assert result.verification_passed  # Should pass verification
    
    def test_algorithm_efficiency(self, algorithm, test_circuit):
        """Test algorithm efficiency"""
        import time
        
        start_time = time.time()
        result = algorithm.quantum_fourier_transform(test_circuit)
        execution_time = time.time() - start_time
        
        assert result.success
        assert execution_time < 1.0  # Should execute within reasonable time
        assert result.error_rate < 0.1
    
    def test_algorithm_consistency(self, algorithm, test_circuit):
        """Test consistency of algorithm results"""
        results = []
        for _ in range(10):
            result = algorithm.quantum_fourier_transform(test_circuit)
            results.append(result.final_state)
        
        # Check consistency of results
        for i in range(1, len(results)):
            assert np.allclose(results[i], results[0], atol=1e-6)
    
    def test_algorithm_scaling(self, algorithm):
        """Test algorithm scaling with different qubit counts"""
        for num_qubits in [2, 4, 6, 8]:
            circuit = QuantumCircuit(num_qubits=num_qubits)
            result = algorithm.quantum_fourier_transform(circuit)
            
            assert result.success
            assert result.final_state.shape == (2**num_qubits,)
            assert result.error_rate < 0.1
    
    def test_algorithm_verification(self, algorithm, test_circuit):
        """Test algorithm verification"""
        result = algorithm.quantum_fourier_transform(test_circuit)
        
        # Verify the result is a valid quantum state
        assert np.isclose(np.sum(np.abs(result.final_state)**2), 1.0)
        assert result.verification_passed
        assert result.error_rate < 0.1 
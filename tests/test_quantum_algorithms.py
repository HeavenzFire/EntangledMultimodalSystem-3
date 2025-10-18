import unittest
import numpy as np
from qiskit import QuantumCircuit, Aer
from core.quantum.quantum_algorithms import QuantumAlgorithms
from core.quantum.quantum_benchmarking import QuantumBenchmarking

class TestQuantumAlgorithms(unittest.TestCase):
    def setUp(self):
        self.num_qubits = 4
        self.algorithms = QuantumAlgorithms(self.num_qubits)
        self.backend = Aer.get_backend('qasm_simulator')
        
    def test_grovers_search(self):
        """Test Grover's search algorithm"""
        # Test with a simple oracle
        oracle = QuantumCircuit(self.num_qubits)
        oracle.x(0)  # Mark |0001> as solution
        oracle.cz(0, 1)
        oracle.x(0)
        
        circuit = self.algorithms.grovers_search(oracle, iterations=1)
        self.assertIsInstance(circuit, QuantumCircuit)
        self.assertEqual(circuit.num_qubits, self.num_qubits)
        
    def test_shors_factoring(self):
        """Test Shor's factoring algorithm"""
        # Test with a small number
        N = 15
        circuit = self.algorithms.shors_factoring(N)
        self.assertIsInstance(circuit, QuantumCircuit)
        self.assertTrue(circuit.num_qubits > 0)
        
    def test_quantum_fourier_transform(self):
        """Test Quantum Fourier Transform"""
        circuit = self.algorithms.quantum_fourier_transform()
        self.assertIsInstance(circuit, QuantumCircuit)
        self.assertEqual(circuit.num_qubits, self.num_qubits)
        
    def test_error_correction(self):
        """Test quantum error correction"""
        # Create a simple circuit
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        
        # Apply error correction
        corrected_circuit = self.algorithms.error_correction(circuit, 'surface')
        self.assertIsInstance(corrected_circuit, QuantumCircuit)
        self.assertTrue(corrected_circuit.num_qubits > circuit.num_qubits)
        
    def test_error_mitigation(self):
        """Test error mitigation techniques"""
        # Test with sample counts
        counts = {'00': 900, '01': 50, '10': 40, '11': 10}
        
        # Test ZNE
        mitigated_zne = self.algorithms.error_mitigation(counts, 'zne')
        self.assertIsInstance(mitigated_zne, dict)
        self.assertEqual(set(mitigated_zne.keys()), set(counts.keys()))
        
        # Test readout error mitigation
        mitigated_readout = self.algorithms.error_mitigation(counts, 'readout')
        self.assertIsInstance(mitigated_readout, dict)
        self.assertEqual(set(mitigated_readout.keys()), set(counts.keys()))

class TestQuantumBenchmarking(unittest.TestCase):
    def setUp(self):
        self.benchmarking = QuantumBenchmarking()
        
    def test_benchmark_algorithm(self):
        """Test algorithm benchmarking"""
        # Test Grover's search
        params = {
            'oracle': QuantumCircuit(4),
            'iterations': 1
        }
        results = self.benchmarking.benchmark_algorithm('grover', params)
        
        self.assertIn('algorithm', results)
        self.assertIn('execution_time', results)
        self.assertIn('counts', results)
        self.assertIn('circuit_depth', results)
        self.assertIn('gate_count', results)
        
    def test_benchmark_error_correction(self):
        """Test error correction benchmarking"""
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        
        results = self.benchmarking.benchmark_error_correction(circuit, 'surface')
        
        self.assertIn('code', results)
        self.assertIn('original_error_rate', results)
        self.assertIn('corrected_error_rate', results)
        self.assertIn('overhead', results)
        
    def test_benchmark_error_mitigation(self):
        """Test error mitigation benchmarking"""
        counts = {'00': 900, '01': 50, '10': 40, '11': 10}
        
        results = self.benchmarking.benchmark_error_mitigation(counts, 'zne')
        
        self.assertIn('method', results)
        self.assertIn('original_counts', results)
        self.assertIn('mitigated_counts', results)
        self.assertIn('improvement', results)
        
    def test_comprehensive_benchmark(self):
        """Test comprehensive benchmarking suite"""
        algorithms = ['grover', 'shor', 'qft']
        params = [
            {'oracle': QuantumCircuit(4), 'iterations': 1},
            {'N': 15},
            {}
        ]
        
        results = self.benchmarking.run_comprehensive_benchmark(algorithms, params)
        
        # Check algorithm benchmarks
        for algo in algorithms:
            self.assertIn(algo, results)
            
        # Check error correction benchmarks
        for code in ['surface', 'stabilizer']:
            self.assertIn(f'error_correction_{code}', results)
            
        # Check error mitigation benchmarks
        for method in ['zne', 'readout']:
            self.assertIn(f'error_mitigation_{method}', results)

if __name__ == '__main__':
    unittest.main() 
import unittest
import numpy as np
from qiskit import QuantumCircuit, transpile, Aer, execute
from qiskit.quantum_info import Operator, average_gate_fidelity, process_fidelity
from qiskit.providers.aer.noise import NoiseModel, depolarizing_error
from qiskit.ignis.verification.tomography import process_tomography_circuits, ProcessTomographyFitter

class TestGateFidelity(unittest.TestCase):
    def setUp(self):
        # Prepare the simulator and create a simple noise model
        self.backend = Aer.get_backend('qasm_simulator')
        self.noise_model = NoiseModel()
        
        # Create depolarizing errors for single and two-qubit gates
        self.single_qubit_error = depolarizing_error(0.01, 1)
        self.two_qubit_error = depolarizing_error(0.05, 2)
        
        # Add errors to the noise model
        self.noise_model.add_all_qubit_quantum_error(self.single_qubit_error, ['x', 'h'])
        self.noise_model.add_all_qubit_quantum_error(self.two_qubit_error, ['cx'])

    def test_x_gate_state_fidelity(self):
        """Test the fidelity of a noisy X gate by comparing the measurement counts."""
        circuit = QuantumCircuit(1, 1)
        circuit.x(0)
        circuit.measure(0, 0)
        
        transpiled_circuit = transpile(circuit, self.backend)
        shots = 10000
        noisy_result = execute(transpiled_circuit,
                             self.backend,
                             noise_model=self.noise_model,
                             shots=shots).result()
        counts = noisy_result.get_counts()
        
        prob_one = counts.get('1', 0) / shots
        self.assertGreaterEqual(prob_one, 0.98,
                              f"X gate fidelity is too low: {prob_one:.4f} (expected >= 0.98)")

    def test_hadamard_gate_state_fidelity(self):
        """Test the fidelity of a noisy Hadamard gate."""
        circuit = QuantumCircuit(1, 1)
        circuit.h(0)
        circuit.measure(0, 0)
        
        transpiled_circuit = transpile(circuit, self.backend)
        shots = 10000
        noisy_result = execute(transpiled_circuit,
                             self.backend,
                             noise_model=self.noise_model,
                             shots=shots).result()
        counts = noisy_result.get_counts()
        
        # For Hadamard, we expect approximately equal probabilities
        prob_zero = counts.get('0', 0) / shots
        self.assertAlmostEqual(prob_zero, 0.5, delta=0.02,
                             msg=f"Hadamard gate fidelity is too low: {prob_zero:.4f}")

    def test_cnot_gate_state_fidelity(self):
        """Test the fidelity of a noisy CNOT gate."""
        circuit = QuantumCircuit(2, 2)
        circuit.h(0)  # Create superposition
        circuit.cx(0, 1)  # Apply CNOT
        circuit.measure([0, 1], [0, 1])
        
        transpiled_circuit = transpile(circuit, self.backend)
        shots = 10000
        noisy_result = execute(transpiled_circuit,
                             self.backend,
                             noise_model=self.noise_model,
                             shots=shots).result()
        counts = noisy_result.get_counts()
        
        # For ideal CNOT, we expect equal probabilities for |00⟩ and |11⟩
        prob_00 = counts.get('00', 0) / shots
        prob_11 = counts.get('11', 0) / shots
        self.assertAlmostEqual(prob_00, 0.5, delta=0.05,
                             msg=f"CNOT gate fidelity is too low: {prob_00:.4f}")
        self.assertAlmostEqual(prob_11, 0.5, delta=0.05,
                             msg=f"CNOT gate fidelity is too low: {prob_11:.4f}")

    def test_process_fidelity_x(self):
        """Test process fidelity of X gate using process tomography."""
        # Create ideal X gate operator
        ideal_operator = Operator.from_label('X')
        
        # Create process tomography circuits
        qc = QuantumCircuit(1)
        qc.x(0)
        tomo_circuits = process_tomography_circuits(qc, [0])
        
        # Execute tomography circuits
        shots = 1000
        result = execute(tomo_circuits,
                        self.backend,
                        noise_model=self.noise_model,
                        shots=shots).result()
        
        # Fit tomography data
        tomo_fitter = ProcessTomographyFitter(result, tomo_circuits)
        process_matrix = tomo_fitter.fit()
        
        # Calculate process fidelity
        fidelity = process_fidelity(process_matrix, ideal_operator)
        self.assertGreaterEqual(fidelity, 0.95,
                              f"Process fidelity is too low: {fidelity:.4f}")

    def test_average_gate_fidelity(self):
        """Test average gate fidelity for various gates."""
        # Test X gate
        ideal_x = Operator.from_label('X')
        p_x = 0.01
        d_x = 2
        expected_fidelity_x = 1 - p_x + p_x/d_x
        self.assertAlmostEqual(expected_fidelity_x, 0.995, places=3)
        
        # Test CNOT gate
        ideal_cnot = Operator.from_label('CX')
        p_cnot = 0.05
        d_cnot = 4
        expected_fidelity_cnot = 1 - p_cnot + p_cnot/d_cnot
        self.assertAlmostEqual(expected_fidelity_cnot, 0.9625, places=3)

    def test_randomized_benchmarking(self):
        """Test gate fidelity using randomized benchmarking."""
        # Create a sequence of random gates
        circuit = QuantumCircuit(1, 1)
        for _ in range(10):  # 10 random gates
            circuit.h(0)
            circuit.x(0)
        circuit.measure(0, 0)
        
        # Execute the circuit
        shots = 10000
        result = execute(circuit,
                        self.backend,
                        noise_model=self.noise_model,
                        shots=shots).result()
        counts = result.get_counts()
        
        # Calculate survival probability
        total_counts = sum(counts.values())
        survival_prob = counts.get('0', 0) / total_counts
        
        # For a sequence of 10 gates with 1% error rate, we expect:
        expected_survival = (1 - 0.01) ** 10
        self.assertAlmostEqual(survival_prob, expected_survival, delta=0.05,
                             msg=f"Randomized benchmarking fidelity is too low: {survival_prob:.4f}")

if __name__ == '__main__':
    unittest.main() 
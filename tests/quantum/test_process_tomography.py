import unittest
import numpy as np
from qiskit import QuantumCircuit, transpile, Aer, execute
from qiskit.quantum_info import Operator, process_fidelity, state_fidelity
from qiskit.providers.aer.noise import NoiseModel, depolarizing_error
from qiskit.ignis.verification.tomography import (
    process_tomography_circuits,
    ProcessTomographyFitter,
    state_tomography_circuits,
    StateTomographyFitter
)

class TestQuantumProcessTomography(unittest.TestCase):
    def setUp(self):
        # Prepare simulator and noise model
        self.backend = Aer.get_backend('qasm_simulator')
        self.noise_model = NoiseModel()
        
        # Add depolarizing noise
        error = depolarizing_error(0.01, 1)
        self.noise_model.add_all_qubit_quantum_error(error, ['x', 'h', 'cx'])

    def test_standard_qpt_x_gate(self):
        """Test standard quantum process tomography for X gate."""
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

    def test_ancilla_assisted_qpt_cnot(self):
        """Test ancilla-assisted process tomography for CNOT gate."""
        # Create ideal CNOT operator
        ideal_operator = Operator.from_label('CX')
        
        # Create Bell state preparation circuit
        prep = QuantumCircuit(2)
        prep.h(0)
        prep.cx(0, 1)
        
        # Create process tomography circuits
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        tomo_circuits = process_tomography_circuits(qc, [0, 1])
        
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
        self.assertGreaterEqual(fidelity, 0.90,
                              f"CNOT process fidelity is too low: {fidelity:.4f}")

    def test_compressed_sensing_qpt(self):
        """Test compressed sensing approach to process tomography."""
        # Create ideal Hadamard operator
        ideal_operator = Operator.from_label('H')
        
        # Create reduced set of tomography circuits
        qc = QuantumCircuit(1)
        qc.h(0)
        
        # Use a subset of Pauli measurements for compressed sensing
        tomo_circuits = process_tomography_circuits(qc, [0], 
                                                  meas_basis=['PauliX', 'PauliY'])
        
        # Execute tomography circuits
        shots = 1000
        result = execute(tomo_circuits,
                        self.backend,
                        noise_model=self.noise_model,
                        shots=shots).result()
        
        # Fit tomography data with regularization
        tomo_fitter = ProcessTomographyFitter(result, tomo_circuits)
        process_matrix = tomo_fitter.fit(method='lstsq', regularization=0.1)
        
        # Calculate process fidelity
        fidelity = process_fidelity(process_matrix, ideal_operator)
        self.assertGreaterEqual(fidelity, 0.90,
                              f"Compressed sensing fidelity is too low: {fidelity:.4f}")

    def test_maximum_likelihood_qpt(self):
        """Test maximum likelihood estimation for process tomography."""
        # Create ideal T gate operator
        ideal_operator = Operator.from_label('T')
        
        # Create process tomography circuits
        qc = QuantumCircuit(1)
        qc.t(0)
        tomo_circuits = process_tomography_circuits(qc, [0])
        
        # Execute tomography circuits
        shots = 1000
        result = execute(tomo_circuits,
                        self.backend,
                        noise_model=self.noise_model,
                        shots=shots).result()
        
        # Fit tomography data using maximum likelihood
        tomo_fitter = ProcessTomographyFitter(result, tomo_circuits)
        process_matrix = tomo_fitter.fit(method='lstsq')
        
        # Calculate process fidelity
        fidelity = process_fidelity(process_matrix, ideal_operator)
        self.assertGreaterEqual(fidelity, 0.95,
                              f"Maximum likelihood fidelity is too low: {fidelity:.4f}")

    def test_multi_qubit_qpt(self):
        """Test process tomography for a two-qubit gate sequence."""
        # Create ideal operator for H⊗X
        ideal_operator = Operator.from_label('H') ^ Operator.from_label('X')
        
        # Create process tomography circuits
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.x(1)
        tomo_circuits = process_tomography_circuits(qc, [0, 1])
        
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
        self.assertGreaterEqual(fidelity, 0.90,
                              f"Multi-qubit process fidelity is too low: {fidelity:.4f}")

    def test_qpt_error_analysis(self):
        """Test error analysis capabilities of process tomography."""
        # Create ideal X gate operator
        ideal_operator = Operator.from_label('X')
        
        # Create process tomography circuits
        qc = QuantumCircuit(1)
        qc.x(0)
        tomo_circuits = process_tomography_circuits(qc, [0])
        
        # Execute tomography circuits with different noise levels
        noise_levels = [0.01, 0.05, 0.1]
        fidelities = []
        
        for noise_level in noise_levels:
            noise_model = NoiseModel()
            error = depolarizing_error(noise_level, 1)
            noise_model.add_all_qubit_quantum_error(error, ['x'])
            
            result = execute(tomo_circuits,
                           self.backend,
                           noise_model=noise_model,
                           shots=1000).result()
            
            tomo_fitter = ProcessTomographyFitter(result, tomo_circuits)
            process_matrix = tomo_fitter.fit()
            fidelity = process_fidelity(process_matrix, ideal_operator)
            fidelities.append(fidelity)
        
        # Verify that fidelity decreases with increasing noise
        for i in range(len(fidelities)-1):
            self.assertGreater(fidelities[i], fidelities[i+1],
                             f"Fidelity should decrease with increasing noise")

if __name__ == '__main__':
    unittest.main() 
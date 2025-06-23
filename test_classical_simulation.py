import numpy as np
from quantum_avatar_agent import QuantumAvatarAgent
import unittest
from unittest.mock import patch, MagicMock
from scipy.linalg import expm
import time

class ClassicalSimulation:
    """Enhanced classical simulation of quantum avatar agent"""
    def __init__(self, num_qubits=7):
        self.num_qubits = num_qubits
        self.state = np.zeros(2**num_qubits, dtype=np.complex128)
        self.state[0] = 1.0  # Initialize in |0⟩ state
        self.gate_history = []
        self.measurement_history = []
        
    def apply_hadamard(self, qubit):
        """Apply Hadamard gate classically"""
        h_matrix = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
        self._apply_single_qubit_gate(qubit, h_matrix)
        self.gate_history.append(('h', qubit))
        
    def apply_cnot(self, control, target):
        """Apply CNOT gate classically"""
        cnot_matrix = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 0, 1],
                              [0, 0, 1, 0]], dtype=np.complex128)
        self._apply_two_qubit_gate(control, target, cnot_matrix)
        self.gate_history.append(('cx', control, target))
        
    def apply_rotation(self, qubit, theta, phi):
        """Apply rotation gate classically"""
        r_matrix = np.array([[np.cos(theta/2), -np.exp(1j*phi)*np.sin(theta/2)],
                           [np.exp(-1j*phi)*np.sin(theta/2), np.cos(theta/2)]], dtype=np.complex128)
        self._apply_single_qubit_gate(qubit, r_matrix)
        self.gate_history.append(('r', qubit, theta, phi))
        
    def apply_phase(self, qubit, phi):
        """Apply phase gate classically"""
        p_matrix = np.array([[1, 0], [0, np.exp(1j*phi)]], dtype=np.complex128)
        self._apply_single_qubit_gate(qubit, p_matrix)
        self.gate_history.append(('p', qubit, phi))
        
    def _apply_single_qubit_gate(self, qubit, gate):
        """Apply single-qubit gate classically"""
        state = self.state.reshape([2] * self.num_qubits)
        state = np.tensordot(gate, state, axes=([1], [qubit]))
        state = np.moveaxis(state, 0, qubit)
        self.state = state.reshape(2**self.num_qubits)
        
    def _apply_two_qubit_gate(self, control, target, gate):
        """Apply two-qubit gate classically"""
        state = self.state.reshape([2] * self.num_qubits)
        gate = gate.reshape([2, 2, 2, 2])
        state = np.tensordot(gate, state, axes=([2, 3], [control, target]))
        state = np.moveaxis(state, [0, 1], [control, target])
        self.state = state.reshape(2**self.num_qubits)
        
    def measure(self, shots=1):
        """Measure the state classically"""
        probabilities = np.abs(self.state)**2
        results = np.random.choice(len(probabilities), size=shots, p=probabilities)
        self.measurement_history.extend(results)
        return results
        
    def get_state_vector(self):
        """Get the current state vector"""
        return self.state
        
    def get_density_matrix(self):
        """Get the density matrix"""
        return np.outer(self.state, self.state.conj())
        
    def get_entanglement_entropy(self, subsystem):
        """Calculate entanglement entropy for a subsystem"""
        density_matrix = self.get_density_matrix()
        reduced_density = np.trace(density_matrix.reshape([2] * (2*self.num_qubits)), 
                                 axis1=subsystem, axis2=subsystem+self.num_qubits)
        eigenvalues = np.linalg.eigvalsh(reduced_density)
        return -np.sum(eigenvalues * np.log2(eigenvalues + 1e-10))
        
    def get_state_fidelity(self, target_state):
        """Calculate state fidelity with target state"""
        return np.abs(np.vdot(self.state, target_state))**2
        
    def get_purity(self):
        """Calculate state purity"""
        density_matrix = self.get_density_matrix()
        return np.trace(density_matrix @ density_matrix).real

class TestClassicalSimulation(unittest.TestCase):
    def setUp(self):
        self.quantum_agent = QuantumAvatarAgent(name="Quantum", num_qubits=7, depth=3, shots=1024)
        self.classical_sim = ClassicalSimulation(num_qubits=7)
        
    def test_initialization(self):
        """Test initialization equivalence"""
        quantum_state = self.quantum_agent.quantum_state
        classical_state = self.classical_sim.state
        
        self.assertAlmostEqual(np.abs(quantum_state[0])**2, np.abs(classical_state[0])**2, places=5)
        
    def test_hadamard_gate(self):
        """Test Hadamard gate equivalence"""
        self.quantum_agent.consciousness_circuit.h(0)
        self.classical_sim.apply_hadamard(0)
        
        quantum_result = self.quantum_agent.simulator.run(self.quantum_agent.consciousness_circuit).result()
        quantum_counts = quantum_result.get_counts()
        
        classical_state = self.classical_sim.state
        self.assertAlmostEqual(np.abs(classical_state[0])**2, 0.5, places=5)
        self.assertAlmostEqual(np.abs(classical_state[1])**2, 0.5, places=5)
        
    def test_entanglement(self):
        """Test entanglement equivalence"""
        self.quantum_agent.consciousness_circuit.h(0)
        self.quantum_agent.consciousness_circuit.cx(0, 1)
        
        self.classical_sim.apply_hadamard(0)
        self.classical_sim.apply_cnot(0, 1)
        
        quantum_result = self.quantum_agent.simulator.run(self.quantum_agent.consciousness_circuit).result()
        quantum_counts = quantum_result.get_counts()
        
        classical_state = self.classical_sim.state
        self.assertAlmostEqual(np.abs(classical_state[0])**2, 0.5, places=5)
        self.assertAlmostEqual(np.abs(classical_state[3])**2, 0.5, places=5)
        
    def test_measurement(self):
        """Test measurement equivalence"""
        self.quantum_agent.consciousness_circuit.h(0)
        self.classical_sim.apply_hadamard(0)
        
        quantum_results = []
        classical_results = []
        
        for _ in range(1000):
            quantum_result = self.quantum_agent.simulator.run(self.quantum_agent.consciousness_circuit).result()
            quantum_counts = quantum_result.get_counts()
            quantum_results.append(list(quantum_counts.keys())[0])
            
            classical_results.append(self.classical_sim.measure())
            
        quantum_0_count = sum(1 for r in quantum_results if r.endswith('0'))
        classical_0_count = sum(1 for r in classical_results if r == 0)
        
        self.assertAlmostEqual(quantum_0_count/1000, 0.5, places=1)
        self.assertAlmostEqual(classical_0_count/1000, 0.5, places=1)
        
    def test_rotation_gate(self):
        """Test rotation gate equivalence"""
        theta = np.pi/4
        phi = np.pi/2
        
        self.quantum_agent.consciousness_circuit.rx(theta, 0)
        self.quantum_agent.consciousness_circuit.rz(phi, 0)
        self.classical_sim.apply_rotation(0, theta, phi)
        
        quantum_result = self.quantum_agent.simulator.run(self.quantum_agent.consciousness_circuit).result()
        quantum_counts = quantum_result.get_counts()
        
        classical_state = self.classical_sim.state
        self.assertAlmostEqual(np.abs(classical_state[0])**2, np.cos(theta/2)**2, places=5)
        
    def test_phase_gate(self):
        """Test phase gate equivalence"""
        phi = np.pi/2
        
        self.quantum_agent.consciousness_circuit.p(phi, 0)
        self.classical_sim.apply_phase(0, phi)
        
        quantum_result = self.quantum_agent.simulator.run(self.quantum_agent.consciousness_circuit).result()
        quantum_counts = quantum_result.get_counts()
        
        classical_state = self.classical_sim.state
        self.assertAlmostEqual(np.angle(classical_state[1]), phi, places=5)
        
    def test_entanglement_entropy(self):
        """Test entanglement entropy calculation"""
        self.classical_sim.apply_hadamard(0)
        self.classical_sim.apply_cnot(0, 1)
        
        entropy = self.classical_sim.get_entanglement_entropy([0])
        self.assertAlmostEqual(entropy, 1.0, places=5)  # Maximum entanglement
        
    def test_state_fidelity(self):
        """Test state fidelity calculation"""
        target_state = np.zeros(2**7, dtype=np.complex128)
        target_state[0] = 1.0
        
        fidelity = self.classical_sim.get_state_fidelity(target_state)
        self.assertAlmostEqual(fidelity, 1.0, places=5)  # Perfect fidelity
        
    def test_purity(self):
        """Test purity calculation"""
        purity = self.classical_sim.get_purity()
        self.assertAlmostEqual(purity, 1.0, places=5)  # Pure state
        
    def test_gate_history(self):
        """Test gate history tracking"""
        self.classical_sim.apply_hadamard(0)
        self.classical_sim.apply_cnot(0, 1)
        self.classical_sim.apply_rotation(2, np.pi/4, np.pi/2)
        
        self.assertEqual(len(self.classical_sim.gate_history), 3)
        self.assertEqual(self.classical_sim.gate_history[0][0], 'h')
        self.assertEqual(self.classical_sim.gate_history[1][0], 'cx')
        self.assertEqual(self.classical_sim.gate_history[2][0], 'r')
        
    def test_measurement_history(self):
        """Test measurement history tracking"""
        self.classical_sim.apply_hadamard(0)
        results = self.classical_sim.measure(shots=100)
        
        self.assertEqual(len(self.classical_sim.measurement_history), 100)
        self.assertEqual(len(results), 100)

if __name__ == '__main__':
    unittest.main() 
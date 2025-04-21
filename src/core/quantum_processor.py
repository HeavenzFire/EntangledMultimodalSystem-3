import numpy as np
from scipy.linalg import expm
from src.utils.logger import logger
from src.utils.errors import ModelError
import matplotlib.pyplot as plt

class QuantumProcessor:
    def __init__(self):
        """Initialize the quantum processor."""
        try:
            # Pauli matrices
            self.X = np.array([[0, 1], [1, 0]])
            self.Y = np.array([[0, -1j], [1j, 0]])
            self.Z = np.array([[1, 0], [0, -1]])
            
            # Hadamard gate
            self.H = 1/np.sqrt(2) * np.array([[1, 1], [1, -1]])
            
            # Quantum state
            self.quantum_state = None
            self.entanglement_matrix = None
            
            logger.info("QuantumProcessor initialized")
        except Exception as e:
            logger.error(f"Failed to initialize QuantumProcessor: {str(e)}")
            raise ModelError(f"Quantum processor initialization failed: {str(e)}")

    def initialize_state(self, n_qubits):
        """Initialize a quantum state with n qubits."""
        try:
            state = np.zeros(2**n_qubits, dtype=np.complex128)
            state[0] = 1
            self.quantum_state = state
            return state
        except Exception as e:
            logger.error(f"State initialization failed: {str(e)}")
            raise ModelError(f"State initialization failed: {str(e)}")

    def apply_gate(self, gate, qubit):
        """Apply a quantum gate to a specific qubit."""
        try:
            n_qubits = int(np.log2(len(self.quantum_state)))
            
            # Create full gate matrix
            full_gate = np.eye(2**n_qubits, dtype=np.complex128)
            for i in range(2**n_qubits):
                for j in range(2**n_qubits):
                    if (i >> qubit) & 1 == 0 and (j >> qubit) & 1 == 0:
                        full_gate[i, j] = gate[0, 0]
                    elif (i >> qubit) & 1 == 0 and (j >> qubit) & 1 == 1:
                        full_gate[i, j] = gate[0, 1]
                    elif (i >> qubit) & 1 == 1 and (j >> qubit) & 1 == 0:
                        full_gate[i, j] = gate[1, 0]
                    elif (i >> qubit) & 1 == 1 and (j >> qubit) & 1 == 1:
                        full_gate[i, j] = gate[1, 1]
            
            # Apply gate
            self.quantum_state = np.dot(full_gate, self.quantum_state)
            return self.quantum_state
        except Exception as e:
            logger.error(f"Gate application failed: {str(e)}")
            raise ModelError(f"Gate application failed: {str(e)}")

    def create_entanglement(self, qubit1, qubit2):
        """Create entanglement between two qubits."""
        try:
            # Apply Hadamard to first qubit
            self.apply_gate(self.H, qubit1)
            
            # Apply CNOT gate
            n_qubits = int(np.log2(len(self.quantum_state)))
            cnot = np.eye(2**n_qubits, dtype=np.complex128)
            for i in range(2**n_qubits):
                for j in range(2**n_qubits):
                    if (i >> qubit1) & 1 == 1 and (j >> qubit1) & 1 == 1:
                        if (i >> qubit2) & 1 == 0 and (j >> qubit2) & 1 == 1:
                            cnot[i, j] = 1
                        elif (i >> qubit2) & 1 == 1 and (j >> qubit2) & 1 == 0:
                            cnot[i, j] = 1
                        else:
                            cnot[i, j] = 0
            
            # Apply CNOT
            self.quantum_state = np.dot(cnot, self.quantum_state)
            
            # Update entanglement matrix
            self.entanglement_matrix = np.outer(
                self.quantum_state,
                self.quantum_state.conj()
            )
            
            return {
                "state": self.quantum_state.tolist(),
                "entanglement_matrix": self.entanglement_matrix.tolist()
            }
        except Exception as e:
            logger.error(f"Entanglement creation failed: {str(e)}")
            raise ModelError(f"Entanglement creation failed: {str(e)}")

    def measure(self, qubit):
        """Measure a specific qubit."""
        try:
            n_qubits = int(np.log2(len(self.quantum_state)))
            
            # Calculate probabilities
            prob_0 = 0
            prob_1 = 0
            for i in range(2**n_qubits):
                if (i >> qubit) & 1 == 0:
                    prob_0 += abs(self.quantum_state[i])**2
                else:
                    prob_1 += abs(self.quantum_state[i])**2
            
            # Perform measurement
            result = np.random.choice([0, 1], p=[prob_0, prob_1])
            
            # Collapse state
            new_state = np.zeros_like(self.quantum_state)
            for i in range(2**n_qubits):
                if (i >> qubit) & 1 == result:
                    new_state[i] = self.quantum_state[i] / np.sqrt(prob_0 if result == 0 else prob_1)
            
            self.quantum_state = new_state
            return result
        except Exception as e:
            logger.error(f"Measurement failed: {str(e)}")
            raise ModelError(f"Measurement failed: {str(e)}")

    def quantum_fourier_transform(self):
        """Apply quantum Fourier transform to the state."""
        try:
            n_qubits = int(np.log2(len(self.quantum_state)))
            
            # Apply Hadamard to all qubits
            for qubit in range(n_qubits):
                self.apply_gate(self.H, qubit)
            
            # Apply controlled phase gates
            for i in range(n_qubits):
                for j in range(i+1, n_qubits):
                    phase = 2 * np.pi / (2**(j-i+1))
                    phase_gate = np.array([[1, 0], [0, np.exp(1j * phase)]])
                    self.apply_gate(phase_gate, j)
            
            return self.quantum_state
        except Exception as e:
            logger.error(f"Quantum Fourier transform failed: {str(e)}")
            raise ModelError(f"Quantum Fourier transform failed: {str(e)}")

    def visualize_state(self, save_path=None):
        """Visualize the quantum state."""
        try:
            plt.figure(figsize=(12, 6))
            
            # Plot real and imaginary parts
            plt.subplot(1, 2, 1)
            plt.bar(range(len(self.quantum_state)), np.real(self.quantum_state))
            plt.title('Real Part')
            
            plt.subplot(1, 2, 2)
            plt.bar(range(len(self.quantum_state)), np.imag(self.quantum_state))
            plt.title('Imaginary Part')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
            else:
                plt.show()
            
            plt.close()
        except Exception as e:
            logger.error(f"State visualization failed: {str(e)}")
            raise ModelError(f"State visualization failed: {str(e)}")

    def get_state_info(self):
        """Get information about the current quantum state."""
        try:
            return {
                "state_vector": self.quantum_state.tolist(),
                "entanglement_matrix": self.entanglement_matrix.tolist() if self.entanglement_matrix is not None else None,
                "n_qubits": int(np.log2(len(self.quantum_state))),
                "state_norm": np.linalg.norm(self.quantum_state)
            }
        except Exception as e:
            logger.error(f"State info retrieval failed: {str(e)}")
            raise ModelError(f"State info retrieval failed: {str(e)}") 
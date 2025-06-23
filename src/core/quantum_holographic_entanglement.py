import numpy as np
from src.core.quantum_processor import QuantumProcessor
from src.core.holographic_processor import HolographicProcessor
from src.utils.logger import logger
from src.utils.errors import ModelError
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class QuantumHolographicEntanglement:
    def __init__(self):
        """Initialize the quantum-holographic entanglement processor."""
        try:
            self.quantum_processor = QuantumProcessor()
            self.holographic_processor = HolographicProcessor()
            self.entanglement_matrix = None
            self.holographic_state = None
            
            logger.info("QuantumHolographicEntanglement initialized")
        except Exception as e:
            logger.error(f"Failed to initialize QuantumHolographicEntanglement: {str(e)}")
            raise ModelError(f"Quantum-holographic entanglement initialization failed: {str(e)}")

    def create_entangled_state(self, n_qubits, hologram_size):
        """Create an entangled state between quantum and holographic systems."""
        try:
            # Initialize quantum state
            self.quantum_processor.initialize_state(n_qubits)
            
            # Create quantum entanglement
            for i in range(0, n_qubits-1, 2):
                self.quantum_processor.create_entanglement(i, i+1)
            
            # Create holographic state
            point_source = self.holographic_processor.create_point_source(
                position=(hologram_size/2, hologram_size/2),
                size=hologram_size
            )
            
            reference_wave = self.holographic_processor.create_plane_wave(
                angle=np.pi/4,
                size=hologram_size
            )
            
            hologram = self.holographic_processor.create_hologram(
                point_source,
                reference_wave
            )
            
            # Create entanglement matrix
            quantum_state = self.quantum_processor.get_state_info()['state_vector']
            self.entanglement_matrix = np.outer(
                quantum_state,
                hologram.flatten()
            )
            
            self.holographic_state = hologram
            
            return {
                "quantum_state": quantum_state,
                "holographic_state": hologram,
                "entanglement_matrix": self.entanglement_matrix
            }
        except Exception as e:
            logger.error(f"Entangled state creation failed: {str(e)}")
            raise ModelError(f"Entangled state creation failed: {str(e)}")

    def propagate_entanglement(self, distance):
        """Propagate the entangled state through space."""
        try:
            if self.holographic_state is None:
                raise ValueError("No holographic state available for propagation")
            
            # Propagate holographic state
            propagated_state = self.holographic_processor.propagate_wave(
                self.holographic_state,
                distance
            )
            
            # Update entanglement matrix
            quantum_state = self.quantum_processor.get_state_info()['state_vector']
            self.entanglement_matrix = np.outer(
                quantum_state,
                propagated_state.flatten()
            )
            
            self.holographic_state = propagated_state
            
            return {
                "propagated_state": propagated_state,
                "entanglement_matrix": self.entanglement_matrix
            }
        except Exception as e:
            logger.error(f"Entanglement propagation failed: {str(e)}")
            raise ModelError(f"Entanglement propagation failed: {str(e)}")

    def measure_entanglement(self):
        """Measure the entanglement between quantum and holographic states."""
        try:
            if self.entanglement_matrix is None:
                raise ValueError("No entanglement matrix available for measurement")
            
            # Calculate entanglement entropy
            singular_values = np.linalg.svd(self.entanglement_matrix, compute_uv=False)
            entropy = -np.sum(singular_values**2 * np.log2(singular_values**2))
            
            # Calculate correlation strength
            correlation = np.abs(np.corrcoef(
                self.quantum_processor.get_state_info()['state_vector'],
                self.holographic_state.flatten()
            )[0, 1])
            
            return {
                "entanglement_entropy": entropy,
                "correlation_strength": correlation,
                "singular_values": singular_values
            }
        except Exception as e:
            logger.error(f"Entanglement measurement failed: {str(e)}")
            raise ModelError(f"Entanglement measurement failed: {str(e)}")

    def visualize_entanglement(self, save_path=None):
        """Visualize the quantum-holographic entanglement."""
        try:
            fig = plt.figure(figsize=(15, 5))
            
            # Quantum state visualization
            ax1 = fig.add_subplot(131)
            quantum_state = self.quantum_processor.get_state_info()['state_vector']
            ax1.imshow(np.abs(quantum_state.reshape(-1, 1)), cmap='viridis')
            ax1.set_title('Quantum State')
            
            # Holographic state visualization
            ax2 = fig.add_subplot(132)
            ax2.imshow(np.abs(self.holographic_state), cmap='gray')
            ax2.set_title('Holographic State')
            
            # Entanglement matrix visualization
            ax3 = fig.add_subplot(133)
            ax3.imshow(np.abs(self.entanglement_matrix), cmap='plasma')
            ax3.set_title('Entanglement Matrix')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
            else:
                plt.show()
            
            plt.close()
        except Exception as e:
            logger.error(f"Entanglement visualization failed: {str(e)}")
            raise ModelError(f"Entanglement visualization failed: {str(e)}")

    def get_entanglement_info(self):
        """Get information about the quantum-holographic entanglement."""
        try:
            measurements = self.measure_entanglement()
            
            return {
                "quantum_info": self.quantum_processor.get_state_info(),
                "holographic_info": self.holographic_processor.get_hologram_info(),
                "entanglement_measurements": measurements,
                "entanglement_matrix_shape": self.entanglement_matrix.shape if self.entanglement_matrix is not None else None
            }
        except Exception as e:
            logger.error(f"Entanglement info retrieval failed: {str(e)}")
            raise ModelError(f"Entanglement info retrieval failed: {str(e)}") 
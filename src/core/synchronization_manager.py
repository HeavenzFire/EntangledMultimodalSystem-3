import numpy as np
from src.core.quantum_processor import QuantumProcessor
from src.core.holographic_processor import HolographicProcessor
from src.core.neural_interface import NeuralInterface
from src.core.quantum_holographic_entanglement import QuantumHolographicEntanglement
from src.utils.logger import logger
from src.utils.errors import ModelError
import matplotlib.pyplot as plt
from datetime import datetime

class SynchronizationManager:
    def __init__(self):
        """Initialize the synchronization manager."""
        try:
            self.quantum_processor = QuantumProcessor()
            self.holographic_processor = HolographicProcessor()
            self.neural_interface = NeuralInterface()
            self.entanglement_processor = QuantumHolographicEntanglement()
            
            self.sync_state = {
                "quantum": None,
                "holographic": None,
                "neural": None,
                "entanglement": None,
                "last_sync": None
            }
            
            logger.info("SynchronizationManager initialized")
        except Exception as e:
            logger.error(f"Failed to initialize SynchronizationManager: {str(e)}")
            raise ModelError(f"Synchronization manager initialization failed: {str(e)}")

    def synchronize_systems(self, n_qubits, hologram_size):
        """Synchronize all system components."""
        try:
            # Create entangled state
            entanglement_state = self.entanglement_processor.create_entangled_state(
                n_qubits,
                hologram_size
            )
            
            # Process through neural interface
            neural_output = self.neural_interface.integrate_systems(
                n_qubits,
                hologram_size
            )
            
            # Update sync state
            self.sync_state.update({
                "quantum": self.quantum_processor.get_state_info(),
                "holographic": self.holographic_processor.get_hologram_info(),
                "neural": neural_output,
                "entanglement": entanglement_state,
                "last_sync": datetime.now().isoformat()
            })
            
            return self.sync_state
        except Exception as e:
            logger.error(f"System synchronization failed: {str(e)}")
            raise ModelError(f"System synchronization failed: {str(e)}")

    def propagate_synchronized_state(self, distance):
        """Propagate the synchronized state through space."""
        try:
            # Propagate entanglement
            propagated_state = self.entanglement_processor.propagate_entanglement(distance)
            
            # Update neural processing
            neural_output = self.neural_interface.neural_process(
                np.concatenate([
                    propagated_state['propagated_state'].flatten(),
                    self.quantum_processor.get_state_info()['state_vector'].flatten()
                ])
            )
            
            # Update sync state
            self.sync_state.update({
                "quantum": self.quantum_processor.get_state_info(),
                "holographic": self.holographic_processor.get_hologram_info(),
                "neural": neural_output,
                "entanglement": propagated_state,
                "last_sync": datetime.now().isoformat()
            })
            
            return self.sync_state
        except Exception as e:
            logger.error(f"State propagation failed: {str(e)}")
            raise ModelError(f"State propagation failed: {str(e)}")

    def measure_synchronization(self):
        """Measure the synchronization quality between components."""
        try:
            # Get component states
            quantum_state = self.quantum_processor.get_state_info()['state_vector']
            holographic_state = self.holographic_processor.hologram
            neural_output = self.neural_interface.get_system_info()
            entanglement_measurements = self.entanglement_processor.measure_entanglement()
            
            # Calculate synchronization metrics
            quantum_holographic_correlation = np.abs(np.corrcoef(
                quantum_state.flatten(),
                holographic_state.flatten()
            )[0, 1])
            
            neural_entanglement_correlation = np.abs(np.corrcoef(
                neural_output['neural_output'].flatten(),
                entanglement_measurements['singular_values']
            )[0, 1])
            
            # Calculate overall synchronization score
            sync_score = (quantum_holographic_correlation + neural_entanglement_correlation) / 2
            
            return {
                "quantum_holographic_correlation": quantum_holographic_correlation,
                "neural_entanglement_correlation": neural_entanglement_correlation,
                "synchronization_score": sync_score,
                "entanglement_measurements": entanglement_measurements,
                "neural_metrics": neural_output['neural_network_info']
            }
        except Exception as e:
            logger.error(f"Synchronization measurement failed: {str(e)}")
            raise ModelError(f"Synchronization measurement failed: {str(e)}")

    def visualize_synchronization(self, save_path=None):
        """Visualize the synchronized system state."""
        try:
            fig = plt.figure(figsize=(20, 5))
            
            # Quantum state visualization
            ax1 = fig.add_subplot(141)
            quantum_state = self.quantum_processor.get_state_info()['state_vector']
            ax1.imshow(np.abs(quantum_state.reshape(-1, 1)), cmap='viridis')
            ax1.set_title('Quantum State')
            
            # Holographic state visualization
            ax2 = fig.add_subplot(142)
            ax2.imshow(np.abs(self.holographic_processor.hologram), cmap='gray')
            ax2.set_title('Holographic State')
            
            # Entanglement visualization
            ax3 = fig.add_subplot(143)
            entanglement_matrix = self.entanglement_processor.entanglement_matrix
            ax3.imshow(np.abs(entanglement_matrix), cmap='plasma')
            ax3.set_title('Entanglement Matrix')
            
            # Neural output visualization
            ax4 = fig.add_subplot(144)
            neural_output = self.neural_interface.get_system_info()['neural_output']
            ax4.imshow(np.abs(neural_output), cmap='hot')
            ax4.set_title('Neural Output')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
            else:
                plt.show()
            
            plt.close()
        except Exception as e:
            logger.error(f"Synchronization visualization failed: {str(e)}")
            raise ModelError(f"Synchronization visualization failed: {str(e)}")

    def get_system_status(self):
        """Get the current status of all system components."""
        try:
            measurements = self.measure_synchronization()
            
            return {
                "synchronization_state": self.sync_state,
                "measurements": measurements,
                "quantum_info": self.quantum_processor.get_state_info(),
                "holographic_info": self.holographic_processor.get_hologram_info(),
                "neural_info": self.neural_interface.get_system_info(),
                "entanglement_info": self.entanglement_processor.get_entanglement_info()
            }
        except Exception as e:
            logger.error(f"System status retrieval failed: {str(e)}")
            raise ModelError(f"System status retrieval failed: {str(e)}")

    def reset_synchronization(self):
        """Reset all components to their initial states."""
        try:
            # Reinitialize components
            self.quantum_processor = QuantumProcessor()
            self.holographic_processor = HolographicProcessor()
            self.neural_interface = NeuralInterface()
            self.entanglement_processor = QuantumHolographicEntanglement()
            
            # Reset sync state
            self.sync_state = {
                "quantum": None,
                "holographic": None,
                "neural": None,
                "entanglement": None,
                "last_sync": None
            }
            
            logger.info("System synchronization reset")
        except Exception as e:
            logger.error(f"Synchronization reset failed: {str(e)}")
            raise ModelError(f"Synchronization reset failed: {str(e)}") 
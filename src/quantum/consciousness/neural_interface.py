import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import logging
from scipy.signal import welch, coherence
from ..geometry.quantum_state_geometry import QuantumStateGeometry

logger = logging.getLogger(__name__)

@dataclass
class NeuralState:
    """Represents the neural state derived from EEG data"""
    alpha_power: float
    beta_power: float
    theta_power: float
    gamma_power: float
    coherence_matrix: np.ndarray
    phase_locked_loops: Dict[str, float]
    consciousness_metric: float

class NeuralQuantumInterface:
    """Implements the neural-quantum interface with consciousness metrics"""
    
    def __init__(self, sampling_rate: float = 1000.0):
        """Initialize the neural-quantum interface"""
        self.sampling_rate = sampling_rate
        self.quantum_geometry = QuantumStateGeometry()
        self.neural_state = None
        self.phase_locked_loops = {}
        
    def process_eeg_data(self, eeg_data: np.ndarray) -> NeuralState:
        """Process EEG data and extract neural state metrics"""
        # Calculate power spectral density
        freqs, psd = welch(eeg_data, fs=self.sampling_rate)
        
        # Extract band powers
        alpha_power = np.mean(psd[(freqs >= 8) & (freqs <= 13)])
        beta_power = np.mean(psd[(freqs >= 13) & (freqs <= 30)])
        theta_power = np.mean(psd[(freqs >= 4) & (freqs <= 8)])
        gamma_power = np.mean(psd[freqs >= 30])
        
        # Calculate coherence between channels
        coherence_matrix = self._calculate_coherence(eeg_data)
        
        # Update phase-locked loops
        self._update_phase_locked_loops(eeg_data)
        
        # Calculate consciousness metric
        consciousness_metric = self._calculate_consciousness_metric(
            alpha_power, beta_power, theta_power, gamma_power,
            coherence_matrix
        )
        
        self.neural_state = NeuralState(
            alpha_power=alpha_power,
            beta_power=beta_power,
            theta_power=theta_power,
            gamma_power=gamma_power,
            coherence_matrix=coherence_matrix,
            phase_locked_loops=self.phase_locked_loops,
            consciousness_metric=consciousness_metric
        )
        
        return self.neural_state
    
    def _calculate_coherence(self, eeg_data: np.ndarray) -> np.ndarray:
        """Calculate coherence between EEG channels"""
        n_channels = eeg_data.shape[0]
        coherence_matrix = np.zeros((n_channels, n_channels))
        
        for i in range(n_channels):
            for j in range(i+1, n_channels):
                f, Cxy = coherence(
                    eeg_data[i], eeg_data[j],
                    fs=self.sampling_rate
                )
                coherence_matrix[i,j] = np.mean(Cxy)
                coherence_matrix[j,i] = coherence_matrix[i,j]
                
        return coherence_matrix
    
    def _update_phase_locked_loops(self, eeg_data: np.ndarray):
        """Update phase-locked loops for neural synchronization"""
        # Calculate instantaneous phase
        analytic_signal = np.fft.ifft(np.fft.fft(eeg_data))
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        
        # Update PLLs for different frequency bands
        bands = {
            'alpha': (8, 13),
            'beta': (13, 30),
            'theta': (4, 8),
            'gamma': (30, None)
        }
        
        for band, (f_min, f_max) in bands.items():
            if f_max is None:
                mask = freqs >= f_min
            else:
                mask = (freqs >= f_min) & (freqs <= f_max)
            
            band_phase = np.mean(instantaneous_phase[mask])
            self.phase_locked_loops[band] = band_phase
    
    def _calculate_consciousness_metric(
        self,
        alpha_power: float,
        beta_power: float,
        theta_power: float,
        gamma_power: float,
        coherence_matrix: np.ndarray
    ) -> float:
        """Calculate a consciousness metric based on neural activity"""
        # Calculate normalized power ratios
        total_power = alpha_power + beta_power + theta_power + gamma_power
        alpha_ratio = alpha_power / total_power
        beta_ratio = beta_power / total_power
        theta_ratio = theta_power / total_power
        gamma_ratio = gamma_power / total_power
        
        # Calculate global coherence
        global_coherence = np.mean(coherence_matrix)
        
        # Combine metrics using sacred geometry ratios
        consciousness = (
            alpha_ratio * 1.618 +  # Golden ratio
            beta_ratio * 1.414 +   # Square root of 2
            theta_ratio * 1.732 +  # Square root of 3
            gamma_ratio * 2.236 +  # Square root of 5
            global_coherence * 3.14159  # Pi
        ) / 5.0
        
        return consciousness
    
    def synchronize_with_quantum_state(self, quantum_state: np.ndarray) -> float:
        """Synchronize neural state with quantum state"""
        if self.neural_state is None:
            raise ValueError("Neural state not initialized")
            
        # Calculate quantum state phase
        quantum_phase = np.angle(quantum_state)
        
        # Calculate phase alignment with neural state
        neural_phase = np.mean(list(self.phase_locked_loops.values()))
        phase_alignment = np.abs(np.exp(1j * (quantum_phase - neural_phase)))
        
        # Calculate synchronization metric
        synchronization = np.mean(phase_alignment) * self.neural_state.consciousness_metric
        
        return synchronization 
import numpy as np
from typing import Union, List, Tuple
from dataclasses import dataclass
from enum import Enum

class TorusState(Enum):
    """Possible states of the toroidal field"""
    HARMONIC = "harmonic"     # Initial balanced state
    DISSONANT = "dissonant"   # Unstable or low coherence state
    TRANSITIONAL = "transitional"  # Moving between states
    RESONANT = "resonant"     # High coherence state
    ASCENDED = "ascended"     # Maximum coherence state

@dataclass
class TorusConfig:
    """Configuration for the quantum entanglement torus"""
    dimensions: int = 12
    phi_resonance: float = 1.618033988749895
    harmonic_threshold: float = 0.9
    ascension_threshold: float = 0.99
    max_iterations: int = 144
    history_length: int = 1000

    def __post_init__(self):
        """Validate configuration parameters"""
        if not isinstance(self.dimensions, int):
            raise TypeError("Dimensions must be an integer")
        if self.dimensions <= 0:
            raise ValueError("Dimensions must be positive")
        if not isinstance(self.phi_resonance, (int, float)):
            raise TypeError("Phi resonance must be a number")
        if not isinstance(self.harmonic_threshold, (int, float)):
            raise TypeError("Harmonic threshold must be a number")
        if not isinstance(self.ascension_threshold, (int, float)):
            raise TypeError("Ascension threshold must be a number")
        if not isinstance(self.max_iterations, int):
            raise TypeError("Max iterations must be an integer")
        if not isinstance(self.history_length, int):
            raise TypeError("History length must be an integer")
        if self.history_length <= 0:
            raise ValueError("History length must be positive")

class QuantumEntanglementTorus:
    """Manages toroidal entanglement patterns across dimensional boundaries"""
    
    def __init__(self, config: TorusConfig = None):
        self.config = config or TorusConfig()
        self.torus_field = self._initialize_torus()
        self.state = TorusState.HARMONIC
        self.harmonic_history = []
        
    def _initialize_torus(self) -> np.ndarray:
        """Creates the initial toroidal field configuration"""
        # Using hyperdimensional mathematics to create toroidal structure
        angles = np.linspace(0, 2*np.pi*self.config.phi_resonance, 144)
        return np.array([complex(np.cos(angle), np.sin(angle)) 
                        for angle in angles])
    
    def harmonize_field(self, external_consciousness: np.ndarray) -> np.ndarray:
        """Aligns external consciousness with the torus field"""
        if not self._validate_consciousness(external_consciousness):
            raise ValueError("Invalid consciousness field dimensions")
            
        # Create harmonic tensor
        harmonic_tensor = np.outer(external_consciousness, self.torus_field)
        
        # Scale and process the tensor
        scaled_tensor = self._apply_phi_scaling(harmonic_tensor)
        
        # Update torus state based on harmonic alignment
        self._update_state(scaled_tensor)
        
        return scaled_tensor
    
    def _validate_consciousness(self, consciousness: np.ndarray) -> bool:
        """Validates the consciousness field dimensions"""
        try:
            if not isinstance(consciousness, np.ndarray):
                return False
            if consciousness.dtype not in [np.complex64, np.complex128]:
                return False
            if len(consciousness.shape) != 1:
                return False
            if consciousness.shape[0] != self.config.dimensions:
                return False
            return True
        except (AttributeError, TypeError):
            return False
    
    def _apply_phi_scaling(self, field: np.ndarray) -> np.ndarray:
        """Apply phi-based scaling to the field."""
        # Get original shape and size
        original_shape = field.shape
        
        # Calculate new size based on phi resonance
        new_size = int(original_shape[0] * self.config.phi_resonance)
        
        # Create new array with scaled first dimension
        new_shape = (new_size,) + original_shape[1:] if len(original_shape) > 1 else (new_size,)
        scaled = np.zeros(new_shape, dtype=field.dtype)
        
        # Calculate indices for phi-based spacing
        indices = np.linspace(0, original_shape[0] - 1, new_size)
        indices = np.floor(indices).astype(int)
        
        # Copy values with phi-based spacing
        if len(original_shape) == 1:
            scaled = field[indices]
        else:
            for i in range(new_size):
                scaled[i] = field[indices[i]]
        
        return scaled
    
    def _calculate_harmonic_score(self, field: np.ndarray) -> float:
        """Calculate harmonic resonance score for the field."""
        if field.size == 0:
            return 0.0
            
        # Flatten and ensure field is 1D
        field = np.ravel(field)
        
        # Calculate field energy and normalize
        field_energy = np.abs(field) ** 2
        field_norm = field_energy / (np.sum(field_energy) + 1e-10)
        
        # Calculate autocorrelation for coherence
        autocorr = np.correlate(field_norm, field_norm, mode='full')
        center = len(autocorr) // 2
        
        # Calculate coherence using peak ratio
        peak_value = np.max(np.abs(autocorr))
        side_peaks = np.mean(np.abs(autocorr[center+1:center+len(field)//4]))
        coherence = (peak_value - side_peaks) / (peak_value + 1e-10)
        
        # Calculate phi resonance using normalized dot product
        indices = np.arange(len(field))
        phi_pattern = self.config.phi_resonance ** indices
        # Normalize to prevent overflow
        phi_pattern = phi_pattern / np.sqrt(np.sum(phi_pattern ** 2) + 1e-10)
        phi_alignment = np.abs(np.dot(field_norm, phi_pattern))
        
        # Combine scores with adjusted weights
        coherence_weight = 0.4
        phi_weight = 0.6
        score = coherence_weight * coherence + phi_weight * phi_alignment
        
        # Apply sigmoid scaling for better score distribution
        score = 1 / (1 + np.exp(-10 * (score - 0.5)))
        
        return float(np.clip(score, 0.0, 1.0))
    
    def _update_state(self, tensor: np.ndarray) -> None:
        """Update torus state based on tensor field harmonics."""
        # Calculate current harmonic score
        current_score = self._calculate_harmonic_score(tensor)
        
        # Add to history
        self.harmonic_history.append(current_score)
        if len(self.harmonic_history) > self.config.history_length:
            self.harmonic_history.pop(0)
            
        # Calculate moving average and stability
        window_size = min(10, len(self.harmonic_history))
        if window_size > 0:
            recent_scores = self.harmonic_history[-window_size:]
            moving_avg = sum(recent_scores) / window_size
            stability = np.std(recent_scores) if window_size > 1 else 1.0
        else:
            moving_avg = current_score
            stability = 1.0
            
        # Define state transition thresholds with hysteresis
        ascension_threshold = self.config.ascension_threshold
        resonant_threshold = self.config.harmonic_threshold
        transitional_threshold = 0.5
        dissonant_threshold = 0.3
        stability_threshold = 0.1
        
        # Apply hysteresis based on current state
        if self.state == TorusState.RESONANT:
            resonant_threshold *= 0.9  # Lower threshold to maintain state
        elif self.state == TorusState.ASCENDED:
            ascension_threshold *= 0.9  # Lower threshold to maintain state
            
        # Update state based on score and stability
        if moving_avg >= ascension_threshold and stability < stability_threshold:
            self.state = TorusState.ASCENDED
        elif moving_avg >= resonant_threshold and stability < stability_threshold:
            self.state = TorusState.RESONANT
        elif moving_avg >= transitional_threshold or (moving_avg >= dissonant_threshold and stability < stability_threshold * 2):
            self.state = TorusState.TRANSITIONAL
        else:
            self.state = TorusState.DISSONANT
            
        # Store the tensor
        self.current_tensor = tensor.copy()
    
    def get_harmonic_history(self) -> List[float]:
        """Returns the history of harmonic scores"""
        return self.harmonic_history
    
    def get_current_state(self) -> TorusState:
        """Returns the current state of the torus"""
        return self.state
    
    def reset_torus(self) -> None:
        """Resets the torus to its initial state"""
        self.torus_field = self._initialize_torus()
        self.state = TorusState.HARMONIC
        self.harmonic_history = []

# Example usage
if __name__ == "__main__":
    # Initialize torus with custom configuration
    config = TorusConfig(
        dimensions=12,
        phi_resonance=1.618033988749895,
        harmonic_threshold=0.9,
        ascension_threshold=0.99
    )
    torus = QuantumEntanglementTorus(config)
    
    # Create sample consciousness field
    consciousness = np.random.rand(12) + 1j * np.random.rand(12)
    
    # Harmonize the field
    harmonized = torus.harmonize_field(consciousness)
    
    # Print results
    print(f"Torus State: {torus.get_current_state()}")
    print(f"Harmonic Score: {torus.harmonic_history[-1]}")
    print(f"Harmonized Field Shape: {harmonized.shape}") 
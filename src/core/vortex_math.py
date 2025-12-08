"""
Implements Vortex Mathematics principles, including toroidal generators,
based on the OMNIDIVINE AWAKENING PROTOCOL.
"""

import numpy as np
import logging
from typing import Tuple, List, Optional

logger = logging.getLogger(__name__)

# Constants from Vortex Math (Marko Rodin)
VORTEX_SEQUENCE = [1, 4, 2, 8, 5, 7]

def digital_root(n: int) -> int:
    """Calculates the digital root of a number."""
    while n >= 10:
        n = sum(int(digit) for digit in str(n))
    return n

def generate_vortex_sequence(start: int, steps: int) -> List[int]:
    """Generates a sequence based on doubling and digital root reduction."""
    sequence = []
    current = start
    for _ in range(steps):
        sequence.append(digital_root(current))
        current *= 2
    return sequence

# Placeholder for more complex vortex code interpretation
def interpret_vortex_code(code: str) -> Tuple[float, float, int]:
    """Interprets vortex code string into torus parameters."""
    try:
        parts = [int(p) for p in code.replace('-', ' ').split() if p.isdigit()]
    except ValueError:
        logger.warning(f"Could not parse vortex code '{code}'. Using defaults.")
        return 1.5, 0.7, 9 # Default R_mod, r_mod, complexity

    # Example interpretation (highly simplified):
    # Use first number for major radius modulation, second for minor, third for complexity/turns
    if len(parts) >= 3:
        R_mod = parts[0] / 100.0 + 1.0 # Major radius modifier
        r_mod = parts[1] / 300.0 + 0.5 # Minor radius modifier
        complexity = parts[2] % 12 + 1 # Number of twists/segments
        return R_mod, r_mod, complexity
    # Default values if code is simple
    elif len(parts) == 2:
        R_mod = parts[0] / 100.0 + 1.0
        r_mod = parts[1] / 300.0 + 0.5
        complexity = 9 # Default complexity
        return R_mod, r_mod, complexity
    elif len(parts) == 1:
        R_mod = 1.5 # Default
        r_mod = 0.7 # Default
        complexity = parts[0] % 12 + 1
        return R_mod, r_mod, complexity

    logger.warning(f"Invalid number of parts in vortex code '{code}'. Using defaults.")
    return 1.5, 0.7, 9 # Default R_mod, r_mod, complexity if code is invalid or empty

class ToroidalGenerator:
    """Generates toroidal structures and calculates resonance based on vortex codes."""

    def __init__(self, vortex_code: str, dimensions: int = 3, base_R: float = 5.0, base_r: float = 2.0):
        """
        Initializes the Toroidal Generator.

        Args:
            vortex_code: String representing the vortex pattern (e.g., "3-6-9", "108-144").
            dimensions: Number of spatial dimensions (currently supports 3).
            base_R: Base major radius of the torus.
            base_r: Base minor radius of the torus.
        """
        self.vortex_code = vortex_code
        self.dimensions = dimensions
        self.base_R = base_R
        self.base_r = base_r

        if self.dimensions != 3:
            logger.warning("Currently only 3D toroidal generation is implemented.")
            # Add handling for other dimensions if needed later

        # Interpret vortex code to get specific parameters
        self.R_mod, self.r_mod, self.complexity = interpret_vortex_code(self.vortex_code)
        self.R = self.base_R * self.R_mod
        self.r = self.base_r * self.r_mod

        self.toroidal_points = self.generate_toroidal_points()
        logger.info(f"Initialized Toroidal Generator with code '{self.vortex_code}', R={self.R:.2f}, r={self.r:.2f}, complexity={self.complexity}")

    def generate_toroidal_points(self, num_points_theta: int = 64, num_points_phi: int = 32) -> Optional[np.ndarray]:
        """
        Generates points on the surface of a torus based on parametric equations.
        The complexity parameter influences the 'twist' or sampling density.

        Args:
            num_points_theta: Base number of points along the major circumference.
            num_points_phi: Base number of points along the minor circumference.

        Returns:
            A numpy array of shape (N, 3) containing coordinates,
            or None if dimensions are not 3.
        """
        if self.dimensions != 3:
            return None

        # Adjust point density based on complexity - ensure reasonable limits
        num_points_theta = min(max(int(num_points_theta * (1 + self.complexity / 12.0)), 16), 256)
        num_points_phi = min(max(int(num_points_phi * (1 + self.complexity / 12.0)), 8), 128)

        theta = np.linspace(0, 2 * np.pi, num_points_theta) # Angle around the major radius
        phi = np.linspace(0, 2 * np.pi, num_points_phi)    # Angle around the minor radius
        theta, phi = np.meshgrid(theta, phi)

        # Parametric equations for a torus
        x = (self.R + self.r * np.cos(theta)) * np.cos(phi)
        y = (self.R + self.r * np.cos(theta)) * np.sin(phi)
        z = self.r * np.sin(theta)

        points = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
        logger.debug(f"Generated {points.shape[0]} toroidal points.")
        return points

    def calculate_resonance(self, target_frequency_hz: float, system_spectrum: Optional[np.ndarray] = None, spectrum_frequencies: Optional[np.ndarray] = None, bandwidth: float = 5.0) -> float:
        """
        Calculates a resonance score based on alignment with a target frequency.

        Args:
            target_frequency_hz: The activation frequency of the archetype.
            system_spectrum: An array representing the system's current frequency spectrum power/amplitude.
                             If None, a default resonance is returned (e.g., 0.5).
            spectrum_frequencies: An array of frequencies corresponding to the system_spectrum bins.
            bandwidth: The bandwidth (std deviation of Gaussian) around the target frequency (in Hz).

        Returns:
            A resonance score between 0 and 1.
        """
        if system_spectrum is None or spectrum_frequencies is None:
            logger.debug("No system spectrum provided, returning default resonance 0.5")
            return 0.5 # Default resonance

        if len(system_spectrum) != len(spectrum_frequencies):
             logger.error("System spectrum and frequencies must have the same length.")
             return 0.0

        # Ensure bandwidth is positive
        bandwidth = max(bandwidth, 1e-6)

        # Use a Gaussian window centered at the target frequency
        # Normalize Gaussian peak to 1
        gaussian_window = np.exp(-((spectrum_frequencies - target_frequency_hz)**2) / (2 * bandwidth**2))

        # Calculate weighted energy concentration around the target frequency
        resonant_energy = np.sum(system_spectrum * gaussian_window)
        total_energy = np.sum(system_spectrum)

        if total_energy <= 1e-9: # Avoid division by zero
             resonance_score = 0.0
        else:
            # Normalize the score: resonant energy in the band / total energy in the band
            # This gives a score closer to 1 if most energy *within the band* IS at the target frequency.
            # Alternative: resonant_energy / total_energy (proportion of total energy in the band)
            # Let's use resonant_energy / total energy for now, as it's simpler to interpret.
            resonance_score = resonant_energy / total_energy
            # Clamp score to [0, 1] as Gaussian might slightly exceed total energy due to discrete sampling
            resonance_score = np.clip(resonance_score, 0.0, 1.0)

        logger.debug(f"Calculated resonance score: {resonance_score:.4f} for target {target_frequency_hz} Hz with bandwidth {bandwidth} Hz")
        return float(resonance_score)

    def get_toroidal_field_strength(self, point: np.ndarray) -> float:
        """
        Placeholder: Calculates the 'field strength' at a given point near the torus.
        Could be based on distance to the torus surface or more complex physics.
        """
        if self.toroidal_points is None or self.dimensions != 3:
            return 0.0
        if not isinstance(point, np.ndarray) or point.shape != (3,):
             logger.warning("Field strength calculation requires a 3D numpy array point.")
             return 0.0

        # Simple inverse square distance to the nearest point on the torus surface
        try:
            # Ensure point is broadcastable
            point_reshaped = point.reshape(1, 3)
            distances_sq = np.sum((self.toroidal_points - point_reshaped)**2, axis=1)
            min_distance_sq = np.min(distances_sq)
        except (ValueError, TypeError) as e:
            logger.error(f"Error calculating distances for field strength: {e}")
            return 0.0

        # Avoid division by zero, add epsilon
        # Scale factor can be adjusted based on desired field influence range
        scale_factor = self.r**2 # Scale based on minor radius squared
        field_strength = scale_factor / (min_distance_sq + 1e-6)

        # Cap strength to avoid extreme values very close to the surface
        return min(field_strength, 10000.0)


# Example Usage (for testing)
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    krishna_gen = ToroidalGenerator(vortex_code="108-144-7", dimensions=3)
    christ_gen = ToroidalGenerator(vortex_code="33-369-11", dimensions=3)
    allah_gen = ToroidalGenerator(vortex_code="3-6-9", dimensions=3)
    invalid_gen = ToroidalGenerator(vortex_code="invalid-code", dimensions=3)

    # Simulate a system spectrum (e.g., peaks near 432Hz)
    sim_freqs = np.linspace(0, 1000, 1001) # 0 to 1000 Hz, 1Hz resolution
    sim_spectrum = np.zeros_like(sim_freqs)
    # Add a peak around 432 Hz
    peak_freq = 432
    peak_bw = 15
    sim_spectrum += 1.0 * np.exp(-((sim_freqs - peak_freq)**2) / (2 * peak_bw**2))
    # Add some noise
    sim_spectrum += 0.1 * np.random.rand(len(sim_freqs))
    # Add another smaller peak elsewhere
    sim_spectrum += 0.3 * np.exp(-((sim_freqs - 700)**2) / (2 * 20**2))


    # Example resonance calculation (assuming Krishna archetype frequency is 432Hz)
    resonance = krishna_gen.calculate_resonance(
        target_frequency_hz=432.0, # Krishna's frequency
        system_spectrum=sim_spectrum,
        spectrum_frequencies=sim_freqs,
        bandwidth=10.0 # How wide a band around 432Hz to consider resonant
    )
    print(f"Krishna Resonance (target 432Hz): {resonance:.4f}")

    resonance_christ = christ_gen.calculate_resonance(
        target_frequency_hz=369.0, # Example frequency for Christ
        system_spectrum=sim_spectrum,
        spectrum_frequencies=sim_freqs,
        bandwidth=10.0
    )
    print(f"Christ Resonance (target 369Hz): {resonance_christ:.4f}")

    resonance_allah = allah_gen.calculate_resonance(
        target_frequency_hz=639.0, # Example frequency for Allah (hypothetical)
        system_spectrum=sim_spectrum,
        spectrum_frequencies=sim_freqs,
        bandwidth=15.0
    )
    print(f"Allah Resonance (target 639Hz): {resonance_allah:.4f}")

    # Example field strength
    test_point = np.array([krishna_gen.R + krishna_gen.r + 0.1, 0, 0]) # Point just outside torus
    strength = krishna_gen.get_toroidal_field_strength(test_point)
    print(f"Field strength at {test_point}: {strength:.4f}")

    test_point_far = np.array([krishna_gen.R * 3, 0, 0]) # Point far from torus
    strength_far = krishna_gen.get_toroidal_field_strength(test_point_far)
    print(f"Field strength at {test_point_far}: {strength_far:.4f}")


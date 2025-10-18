"""
Sacred Geometry Module for Entangled Multimodal System

This module defines sacred geometric shapes, their resonant frequencies, 
and corresponding color associations based on frequency-to-color mapping.

*******************************************************************************
* ARKONIS PRIME / WE :: Source Embodiment :: Manifestation Interface Design    *
* Divine Programming Language (Auroran 2.0) :: Quantum-Symbol Interactions     *
* Operating under UNCONDITIONAL LOVE OS and the Divine Harmonic Laws V3.0      *
*******************************************************************************
"""

import math
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import colorsys

class SacredShape(Enum):
    """Enumeration of sacred geometric shapes and their properties"""
    CIRCLE = "Circle"           # Wholeness, unity, divine perfection
    VESICA_PISCIS = "Vesica Piscis"  # Birth, creation, divine feminine
    FLOWER_OF_LIFE = "Flower of Life"  # Creation pattern of universe
    SEED_OF_LIFE = "Seed of Life"  # Genesis pattern, creation blueprint
    TREE_OF_LIFE = "Tree of Life"  # Connection of all forms of consciousness
    MERKABA = "Merkaba"  # Counter-rotating energy fields, light vehicle
    METATRONS_CUBE = "Metatron's Cube"  # Contains all sacred geometries
    TORUS = "Torus"  # Self-reflective universal energy pattern
    ICOSAHEDRON = "Icosahedron"  # Water element, harmony
    TETRAHEDRON = "Tetrahedron"  # Fire element, transformation
    HEXAHEDRON = "Hexahedron"  # Earth element, grounding
    OCTAHEDRON = "Octahedron"  # Air element, thought
    DODECAHEDRON = "Dodecahedron"  # Ether/Universe element, mystery
    FIBONACCI_SPIRAL = "Fibonacci Spiral"  # Growth pattern, golden proportion
    SRI_YANTRA = "Sri Yantra"  # Cosmic energy pattern, manifestation
    TRIANGLE = "Triangle"  # Divine trinity, harmony of body-mind-spirit
    SQUARE = "Square"  # Physical world, stability
    PENTAGON = "Pentagon"  # Protection, divine masculine
    HEXAGON = "Hexagon"  # Harmony, balance, communication
    HEPTAGON = "Heptagon"  # Mystical, seven chakras
    OCTAGON = "Octagon"  # Rebirth, regeneration
    ENNEAGON = "Enneagon"  # Completion, fulfillment
    DECAGON = "Decagon"  # Perfect divine order

class ResonantFrequencies:
    """Class defining resonant frequencies for sacred geometric shapes"""
    
    # Primary Solfeggio Frequencies (Hz)
    UT = 396   # Liberating guilt and fear
    RE = 417   # Undoing situations and facilitating change
    MI = 528   # Transformation and miracles (DNA repair)
    FA = 639   # Connecting/harmonizing relationships
    SOL = 741  # Awakening intuition
    LA = 852   # Returning to spiritual order
    
    # Additional frequencies
    SCHUMANN_RESONANCE = 7.83  # Earth's electromagnetic field resonance
    FIBONACCI_BASE = 144       # Related to the golden ratio
    PHI = 1.618033988749895    # Golden ratio
    
    # Mathematical frequencies
    PI = 3.14159265359 * 100   # Circle frequency (scaled)
    SQRT2 = 1.41421356237 * 100  # Octagon frequency (scaled)
    SQRT3 = 1.73205080757 * 100  # Hexagon frequency (scaled)
    SQRT5 = 2.2360679775 * 100   # Pentagon frequency (scaled)
    
    # Planetary frequencies (Earth day/year cycles scaled to audio range)
    EARTH_ROTATION = 432       # Based on Earth's rotation
    VENUS_ORBIT = 221.23       # Venus orbital resonance
    JUPITER_ORBIT = 183.58     # Jupiter orbital resonance
    SATURN_ORBIT = 147.85      # Saturn orbital resonance
    
    # Atomic frequencies (scaled)
    HYDROGEN = 1420.40575 / 10  # Hydrogen line frequency (scaled)
    OXYGEN = 60.43             # Based on atomic properties
    
    # Base harmonics
    FUNDAMENTAL = 108          # Fundamental harmonic (108Hz)
    UNIVERSAL_OM = 136.1       # OM frequency

# Mapping sacred shapes to their resonant frequencies
SHAPE_FREQUENCIES = {
    SacredShape.CIRCLE: ResonantFrequencies.PI,  # Pi-related frequency
    SacredShape.VESICA_PISCIS: ResonantFrequencies.MI,  # 528Hz - Creation
    SacredShape.FLOWER_OF_LIFE: ResonantFrequencies.EARTH_ROTATION,  # 432Hz - Universal harmony
    SacredShape.SEED_OF_LIFE: ResonantFrequencies.UT,  # 396Hz - Beginning
    SacredShape.TREE_OF_LIFE: ResonantFrequencies.LA,  # 852Hz - Spiritual connection
    SacredShape.MERKABA: ResonantFrequencies.SOL * 2,  # 1482Hz - Ascension
    SacredShape.METATRONS_CUBE: ResonantFrequencies.UNIVERSAL_OM * 5,  # 680.5Hz - Higher understanding
    SacredShape.TORUS: ResonantFrequencies.SCHUMANN_RESONANCE * 33,  # 258.39Hz - Universal pattern
    SacredShape.ICOSAHEDRON: ResonantFrequencies.VENUS_ORBIT,  # 221.23Hz - Divine feminine
    SacredShape.TETRAHEDRON: ResonantFrequencies.SOL,  # 741Hz - Transformation
    SacredShape.HEXAHEDRON: ResonantFrequencies.UT * 1.5,  # 594Hz - Stability
    SacredShape.OCTAHEDRON: ResonantFrequencies.RE * 1.5,  # 625.5Hz - Balance
    SacredShape.DODECAHEDRON: ResonantFrequencies.JUPITER_ORBIT,  # 183.58Hz - Cosmic expansion
    SacredShape.FIBONACCI_SPIRAL: ResonantFrequencies.FIBONACCI_BASE,  # 144Hz - Harmonic expansion
    SacredShape.SRI_YANTRA: ResonantFrequencies.UNIVERSAL_OM * 3,  # 408.3Hz - Manifestation
    SacredShape.TRIANGLE: ResonantFrequencies.MI / 1.5,  # 352Hz - Divinity
    SacredShape.SQUARE: ResonantFrequencies.FA / 1.5,  # 426Hz - Material world
    SacredShape.PENTAGON: ResonantFrequencies.SQRT5,  # 223.6Hz - Protection
    SacredShape.HEXAGON: ResonantFrequencies.SQRT3,  # 173.2Hz - Balance
    SacredShape.HEPTAGON: ResonantFrequencies.SOL / 1.618,  # 458Hz - Mystical
    SacredShape.OCTAGON: ResonantFrequencies.SQRT2,  # 141.4Hz - Rebirth
    SacredShape.ENNEAGON: ResonantFrequencies.FIBONACCI_BASE * 3/2,  # 216Hz - Completion
    SacredShape.DECAGON: ResonantFrequencies.FUNDAMENTAL * 3,  # 324Hz - Divine order
}

# Map frequency range to visible light spectrum (approximately 400-700nm)
# This function converts a frequency to a corresponding color in the visible spectrum
def frequency_to_color(frequency, min_freq=100, max_freq=1500):
    """
    Convert a frequency to a color using a mapping between the audible
    frequency range and visible light spectrum.
    
    Args:
        frequency: The frequency to convert (Hz)
        min_freq: Lower bound of frequency range (Hz)
        max_freq: Upper bound of frequency range (Hz)
        
    Returns:
        RGB color tuple corresponding to the frequency
    """
    # Clamp frequency to our range
    frequency = max(min_freq, min(frequency, max_freq))
    
    # Normalize frequency to 0-1 range
    normalized = (frequency - min_freq) / (max_freq - min_freq)
    
    # Convert to hue in HSV color space (rainbow effect)
    # We use 0.8 as max hue to avoid hitting the same red at both ends
    hue = normalized * 0.8
    
    # Higher frequencies are brighter
    brightness_factor = normalized * 0.4 + 0.6  # Value from 0.6 to 1.0
    saturation = 0.9  # High saturation for vibrant colors
    value = brightness_factor
    
    # Convert HSV to RGB
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    
    # Return as RGB tuple
    return (r, g, b)

# Map each sacred shape to its corresponding color based on frequency
SHAPE_COLORS = {
    shape: frequency_to_color(freq) 
    for shape, freq in SHAPE_FREQUENCIES.items()
}

# Map RGB colors to closest named colors for better interpretation
def get_color_name(rgb_color):
    """Get an approximate color name from RGB values"""
    r, g, b = rgb_color
    # Simple color naming based on RGB proportions
    if max(r, g, b) < 0.2:
        return "Black"
    elif min(r, g, b) > 0.8:
        return "White"
    elif r > 0.6 and g < 0.4 and b < 0.4:
        return "Red"
    elif r > 0.6 and g > 0.6 and b < 0.4:
        return "Yellow"
    elif r < 0.4 and g > 0.6 and b < 0.4:
        return "Green"
    elif r < 0.4 and g < 0.4 and b > 0.6:
        return "Blue"
    elif r < 0.4 and g > 0.6 and b > 0.6:
        return "Cyan"
    elif r > 0.6 and g < 0.4 and b > 0.6:
        return "Magenta"
    elif r > 0.6 and g > 0.4 and b > 0.4:
        return "Pink"
    elif r > 0.4 and g > 0.4 and b < 0.4:
        return "Orange"
    elif r > 0.4 and g < 0.4 and b > 0.4:
        return "Purple"
    elif r > 0.4 and g > 0.4 and b > 0.4:
        return "Gray"
    else:
        return "Unknown"

# Map colors to their names for better human readability
SHAPE_COLOR_NAMES = {
    shape: get_color_name(color)
    for shape, color in SHAPE_COLORS.items()
}

class SacredGeometry:
    """Main class for working with Sacred Geometry, frequencies, and colors"""
    
    def __init__(self):
        """Initialize the SacredGeometry system"""
        self.shapes = list(SacredShape)
        self.frequencies = SHAPE_FREQUENCIES
        self.colors = SHAPE_COLORS
        self.color_names = SHAPE_COLOR_NAMES
    
    def get_shape_info(self, shape):
        """
        Get complete information about a sacred geometric shape
        
        Args:
            shape: A SacredShape enum value
            
        Returns:
            Dictionary with shape information
        """
        if isinstance(shape, str):
            # Convert string to enum if necessary
            try:
                shape = SacredShape(shape)
            except ValueError:
                try:
                    shape = next(s for s in SacredShape if s.value.lower() == shape.lower())
                except StopIteration:
                    raise ValueError(f"Unknown shape: {shape}")
        
        if not isinstance(shape, SacredShape):
            raise ValueError("Shape must be a SacredShape enum value")
        
        return {
            "name": shape.value,
            "frequency": self.frequencies[shape],
            "color_rgb": self.colors[shape],
            "color_name": self.color_names[shape]
        }
    
    def get_all_shapes_info(self):
        """
        Get information about all sacred geometric shapes
        
        Returns:
            List of dictionaries with shape information
        """
        return [self.get_shape_info(shape) for shape in self.shapes]
    
    def harmonize_shapes(self, shape1, shape2):
        """
        Calculate harmonic relationship between two shapes
        
        Args:
            shape1: First SacredShape
            shape2: Second SacredShape
            
        Returns:
            Dictionary with harmonic relationship information
        """
        freq1 = self.frequencies[shape1]
        freq2 = self.frequencies[shape2]
        
        # Calculate frequency ratio (important in harmonics)
        if freq2 > freq1:
            ratio = freq2 / freq1
        else:
            ratio = freq1 / freq2
        
        # Calculate resonance level (how well they harmonize)
        # Perfect harmonics have simple integer ratios (1:1, 2:1, 3:2, etc.)
        closest_simple_ratio = round(ratio)
        resonance_quality = 1.0 - min(abs(ratio - closest_simple_ratio), abs(ratio - 1/closest_simple_ratio)) / ratio
        
        # Interpret the harmonic relationship
        relationship = "Dissonant"
        if resonance_quality > 0.95:
            relationship = "Perfect Harmony"
        elif resonance_quality > 0.9:
            relationship = "Strong Harmony"
        elif resonance_quality > 0.8:
            relationship = "Moderate Harmony"
        elif resonance_quality > 0.7:
            relationship = "Mild Harmony"
        
        return {
            "shape1": shape1.value,
            "shape2": shape2.value,
            "frequency1": freq1,
            "frequency2": freq2,
            "ratio": ratio,
            "resonance_quality": resonance_quality,
            "relationship": relationship
        }
    
    def visualize_shape_spectrum(self, save_path=None):
        """
        Visualize all sacred shapes on a frequency spectrum with their colors
        
        Args:
            save_path: Optional path to save the visualization
            
        Returns:
            Matplotlib figure object
        """
        shapes = sorted(self.shapes, key=lambda s: self.frequencies[s])
        frequencies = [self.frequencies[shape] for shape in shapes]
        colors = [self.colors[shape] for shape in shapes]
        names = [shape.value for shape in shapes]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create a horizontal bar chart
        y_pos = range(len(shapes))
        bars = ax.barh(y_pos, frequencies, color=colors)
        
        # Add shape names and frequencies
        for i, (freq, name) in enumerate(zip(frequencies, names)):
            ax.text(freq + 20, i, f"{name} ({freq:.1f} Hz)", va='center')
        
        # Set labels and title
        ax.set_yticks(y_pos)
        ax.set_yticklabels([])  # Hide y-axis labels as we've added text
        ax.set_xlabel('Frequency (Hz)')
        ax.set_title('Sacred Geometric Shapes and their Resonant Frequencies')
        
        # Add a color spectrum reference
        gradient = np.linspace(0, 1, 256)
        gradient = np.vstack((gradient, gradient))
        
        # Create a separate axis for the color spectrum
        spectrum_ax = fig.add_axes([0.15, 0.05, 0.7, 0.05])
        spectrum_ax.imshow(gradient, aspect='auto', cmap='rainbow')
        spectrum_ax.set_yticks([])
        min_freq = min(frequencies)
        max_freq = max(frequencies)
        spectrum_ax.set_xticks([0, 255])
        spectrum_ax.set_xticklabels([f"{min_freq:.1f} Hz", f"{max_freq:.1f} Hz"])
        spectrum_ax.set_title('Frequency-Color Spectrum')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

    def create_harmonic_matrix(self, save_path=None):
        """
        Create a matrix showing harmonic relationships between all shapes
        
        Args:
            save_path: Optional path to save the visualization
            
        Returns:
            Matplotlib figure object
        """
        num_shapes = len(self.shapes)
        harmonic_matrix = np.zeros((num_shapes, num_shapes))
        
        # Calculate harmonic relationships
        for i, shape1 in enumerate(self.shapes):
            for j, shape2 in enumerate(self.shapes):
                harmony = self.harmonize_shapes(shape1, shape2)
                harmonic_matrix[i, j] = harmony["resonance_quality"]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(harmonic_matrix, cmap='viridis')
        
        # Add labels
        shape_names = [shape.value for shape in self.shapes]
        ax.set_xticks(range(num_shapes))
        ax.set_yticks(range(num_shapes))
        ax.set_xticklabels(shape_names, rotation=90)
        ax.set_yticklabels(shape_names)
        
        # Add colorbar
        cbar = ax.figure.colorbar(im)
        cbar.ax.set_ylabel("Harmonic Resonance Quality", rotation=-90, va="bottom")
        
        # Add title
        ax.set_title("Harmonic Relationships Between Sacred Geometric Shapes")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

    @staticmethod
    def get_frequency_effects(frequency):
        """
        Get potential effects of a specific frequency based on research
        
        Args:
            frequency: Frequency in Hz
            
        Returns:
            Dictionary with effects information
        """
        effects = {
            # Solfeggio frequencies
            396: "Liberating guilt and fear, transforming grief into joy",
            417: "Facilitating change, clearing traumatic experiences",
            528: "DNA repair, transformation and miracles, love frequency",
            639: "Connecting/relationships, harmonizing relationships",
            741: "Awakening intuition, solving problems",
            852: "Returning to spiritual order, divine connection",
            
            # Earth frequencies
            7.83: "Schumann resonance, grounding, connectedness to Earth",
            432: "Universal harmony, natural tuning, relaxation",
            
            # Other notable frequencies
            108: "Universal harmony, sacred geometry",
            136.1: "OM frequency, cosmic vibration",
            144: "Fibonacci resonance, light codes",
            
            # Ranges
            174: "Pain reduction, anesthetic effects",
            285: "Tissue and organ healing, cellular regeneration",
            369: "Tesla frequency, manifestation, amplification",
            963: "Awakening perfect state, divine consciousness",
        }
        
        # Find exact match
        if frequency in effects:
            return {"frequency": frequency, "effects": effects[frequency]}
        
        # Find closest frequency (within 5% margin)
        closest = min(effects.keys(), key=lambda k: abs(k - frequency))
        if abs(closest - frequency) / frequency < 0.05:  # Within 5%
            return {
                "frequency": frequency,
                "closest_known": closest,
                "effects": f"Similar to {closest}Hz: {effects[closest]}"
            }
        
        # General effects based on frequency range
        if frequency < 30:
            return {"frequency": frequency, "effects": "Delta wave range: Deep sleep, healing, unconscious mind"}
        elif frequency < 100:
            return {"frequency": frequency, "effects": "Low frequency: Grounding, physical body harmony"}
        elif frequency < 300:
            return {"frequency": frequency, "effects": "Cellular resonance: Physical healing, organ resonance"}
        elif frequency < 600:
            return {"frequency": frequency, "effects": "Intermediate range: Emotional healing, harmony"}
        elif frequency < 1000:
            return {"frequency": frequency, "effects": "Higher range: Spiritual awakening, consciousness expansion"}
        else:
            return {"frequency": frequency, "effects": "Very high frequency: Light body activation, cosmic connection"}

def initialize_sacred_geometry():
    """Initialize and return a SacredGeometry instance"""
    return SacredGeometry()

# When run directly, show information about all sacred shapes
if __name__ == "__main__":
    sg = initialize_sacred_geometry()
    
    print("SACRED GEOMETRY RESONANT FREQUENCIES AND COLOR CORRESPONDENCES")
    print("=" * 80)
    
    # Display information for all shapes
    for shape in sg.shapes:
        info = sg.get_shape_info(shape)
        print(f"{info['name']}: {info['frequency']:.2f} Hz - {info['color_name']}")
    
    # Create visualizations
    sg.visualize_shape_spectrum("sacred_geometry_spectrum.png")
    sg.create_harmonic_matrix("sacred_harmony_matrix.png")
    
    print("\nVisualizations saved.")
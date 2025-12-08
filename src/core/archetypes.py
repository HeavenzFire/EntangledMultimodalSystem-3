"""
Defines the sacred archetypes based on the OMNIDIVINE AWAKENING PROTOCOL.
"""

from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class Archetype:
    """Represents a Divine Archetype with associated properties."""
    name: str
    sacred_geometry: str
    vortex_code: Tuple[int, int] # Seed -> Target/Parameter (e.g., (108, 144))
    quantum_state_symbol: str # e.g., ψ\u2096
    activation_frequency_hz: float
    chakra_association: Optional[str] = None
    mantra: Optional[str] = None # Added for potential use

# Divine Archetype Matrix Data
# Source: OMNIDIVINE AWAKENING PROTOCOL Document
ARCHETYPE_MATRIX = {
    "krishna": Archetype(
        name="Krishna",
        sacred_geometry="Sri Yantra",
        vortex_code=(108, 144),
        quantum_state_symbol="ψ\u2096", # ψ subscript k
        activation_frequency_hz=432.0,
        chakra_association="Heart Chakra",
        mantra="ॐ नमो भगवते वासुदेवाय"
    ),
    "christ": Archetype(
        name="Christ",
        sacred_geometry="Crucifixion Torus",
        vortex_code=(33, 369),
        quantum_state_symbol="ψ\u1D9C", # ψ superscript c
        activation_frequency_hz=528.0,
        chakra_association="DNA Repair", # Note: Protocol lists this, not a standard chakra
        mantra=None # Example: Could add "Kyrie Eleison" or similar
    ),
    "allah": Archetype(
        name="Allah",
        sacred_geometry="99-Fold Fractal",
        vortex_code=(99, 144),
        quantum_state_symbol="ψ\u1D43", # ψ superscript A
        activation_frequency_hz=963.0,
        chakra_association="Crown Chakra",
        mantra=None # Example: Could add "La ilaha illallah"
    ),
    "buddha": Archetype(
        name="Buddha",
        sacred_geometry="Dharma Wheel (8D)",
        vortex_code=(8, float('inf')), # Representing 8 -> ∞
        quantum_state_symbol="ψ\u1D47", # ψ superscript B
        activation_frequency_hz=174.0,
        chakra_association="Root Stability", # Note: Protocol lists this
        mantra=None # Example: Could add "Om Mani Padme Hum"
    ),
    "divine_feminine": Archetype(
        name="Divine Feminine",
        sacred_geometry="Vesica Piscis",
        vortex_code=(3, 9), # Representing 3 -> 6 -> 9 (using start/end)
        quantum_state_symbol="ψ\u1D60\u1D52", # ψ Sophia-Logos (approximation)
        activation_frequency_hz=639.0,
        chakra_association="Interconnection", # Note: Protocol lists this
        mantra=None # Example: Could add "Om Shakti Om"
    ),
}

def get_archetype(name: str) -> Optional[Archetype]:
    """Retrieves an archetype by name from the matrix."""
    return ARCHETYPE_MATRIX.get(name.lower())


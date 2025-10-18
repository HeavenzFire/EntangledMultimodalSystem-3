import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional

class FrequencyType(Enum):
    SACRED = "sacred"
    HARMONIC = "harmonic"
    QUANTUM = "quantum"
    DIVINE = "divine"

@dataclass
class ResonancePattern:
    frequencies: List[float]
    amplitudes: List[float]
    phases: List[float]
    pattern_type: FrequencyType
    energy_level: float

class QuantumResonance:
    def __init__(self):
        self.patterns: Dict[str, ResonancePattern] = {}
        self.active_pattern: Optional[ResonancePattern] = None
        self.base_frequency = 432.0  # Hz
        self.energy_threshold = 0.5
        self.resonance_factor = 1.0

    def generate_pattern(self, pattern_type: FrequencyType) -> ResonancePattern:
        """Generate a resonance pattern based on the specified type."""
        if pattern_type == FrequencyType.SACRED:
            return self._generate_sacred_pattern()
        elif pattern_type == FrequencyType.HARMONIC:
            return self._generate_harmonic_pattern()
        elif pattern_type == FrequencyType.QUANTUM:
            return self._generate_quantum_pattern()
        elif pattern_type == FrequencyType.DIVINE:
            return self._generate_divine_pattern()
        else:
            raise ValueError(f"Unknown pattern type: {pattern_type}")

    def _generate_sacred_pattern(self) -> ResonancePattern:
        """Generate sacred frequency pattern."""
        frequencies = [
            432.0,  # Earth resonance
            528.0,  # DNA repair
            639.0,  # Heart connection
            741.0,  # Awakening intuition
            852.0   # Spiritual order
        ]
        amplitudes = [1.0, 0.8, 0.6, 0.4, 0.2]
        phases = [0.0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
        
        return ResonancePattern(
            frequencies=frequencies,
            amplitudes=amplitudes,
            phases=phases,
            pattern_type=FrequencyType.SACRED,
            energy_level=1.0
        )

    def _generate_harmonic_pattern(self) -> ResonancePattern:
        """Generate harmonic frequency pattern."""
        base = self.base_frequency
        frequencies = [base * (i + 1) for i in range(8)]
        amplitudes = [1.0 / (i + 1) for i in range(8)]
        phases = [i * np.pi/4 for i in range(8)]
        
        return ResonancePattern(
            frequencies=frequencies,
            amplitudes=amplitudes,
            phases=phases,
            pattern_type=FrequencyType.HARMONIC,
            energy_level=0.8
        )

    def _generate_quantum_pattern(self) -> ResonancePattern:
        """Generate quantum frequency pattern."""
        frequencies = [
            963.0,  # Pineal activation
            852.0,  # Third eye
            741.0,  # Throat chakra
            639.0,  # Heart chakra
            528.0,  # Solar plexus
            417.0,  # Sacral chakra
            396.0   # Root chakra
        ]
        amplitudes = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
        phases = [i * np.pi/3 for i in range(7)]
        
        return ResonancePattern(
            frequencies=frequencies,
            amplitudes=amplitudes,
            phases=phases,
            pattern_type=FrequencyType.QUANTUM,
            energy_level=0.9
        )

    def _generate_divine_pattern(self) -> ResonancePattern:
        """Generate divine frequency pattern."""
        frequencies = [
            111.0,  # Angelic connection
            222.0,  # Spiritual awakening
            333.0,  # Ascended masters
            444.0,  # Archangels
            555.0,  # Major life changes
            666.0,  # Material world
            777.0,  # Divine perfection
            888.0,  # Abundance
            999.0   # Completion
        ]
        amplitudes = [1.0] * 9
        phases = [i * np.pi/4 for i in range(9)]
        
        return ResonancePattern(
            frequencies=frequencies,
            amplitudes=amplitudes,
            phases=phases,
            pattern_type=FrequencyType.DIVINE,
            energy_level=1.0
        )

    def transform_pattern(self, pattern: ResonancePattern,
                         frequency_scale: float = 1.0,
                         amplitude_scale: float = 1.0,
                         phase_shift: float = 0.0) -> ResonancePattern:
        """Apply transformations to a resonance pattern."""
        transformed_frequencies = [f * frequency_scale for f in pattern.frequencies]
        transformed_amplitudes = [a * amplitude_scale for a in pattern.amplitudes]
        transformed_phases = [(p + phase_shift) % (2 * np.pi) for p in pattern.phases]
        
        return ResonancePattern(
            frequencies=transformed_frequencies,
            amplitudes=transformed_amplitudes,
            phases=transformed_phases,
            pattern_type=pattern.pattern_type,
            energy_level=pattern.energy_level
        )

    def activate_pattern(self, pattern_type: FrequencyType) -> None:
        """Activate a specific resonance pattern."""
        if pattern_type not in self.patterns:
            self.patterns[pattern_type.name] = self.generate_pattern(pattern_type)
        self.active_pattern = self.patterns[pattern_type.name]

    def calculate_resonance(self, time: float) -> float:
        """Calculate the resonance value at a given time."""
        if not self.active_pattern:
            return 0.0
        
        resonance = 0.0
        for freq, amp, phase in zip(
            self.active_pattern.frequencies,
            self.active_pattern.amplitudes,
            self.active_pattern.phases
        ):
            resonance += amp * np.sin(2 * np.pi * freq * time + phase)
        
        return resonance * self.resonance_factor

    def check_energy_alignment(self) -> bool:
        """Check if the current energy level meets the threshold."""
        if not self.active_pattern:
            return False
        return self.active_pattern.energy_level >= self.energy_threshold

    def update_resonance_factor(self, factor: float) -> None:
        """Update the resonance factor."""
        self.resonance_factor = max(0.0, min(1.0, factor)) 
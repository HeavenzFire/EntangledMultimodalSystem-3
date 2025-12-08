import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
from .cosmic_protocols import CosmicResonance, DivineManifestation

class CosmicVisualizer:
    """Visualizes cosmic protocols and divine manifestation"""
    
    def __init__(self):
        self.fig = None
        self.ax = None
        
    def plot_entanglement_spectrum(self, resonance: CosmicResonance) -> plt.Figure:
        """Plot entanglement spectrum"""
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        
        # Plot spectrum
        spectrum = resonance.entanglement_spectrum
        n = np.arange(1, len(spectrum) + 1)
        
        self.ax.plot(n, np.real(spectrum), label='Real')
        self.ax.plot(n, np.imag(spectrum), label='Imaginary')
        
        self.ax.set_title('Entanglement Spectrum')
        self.ax.set_xlabel('Harmonic Number')
        self.ax.set_ylabel('Amplitude')
        self.ax.legend()
        self.ax.grid(True)
        
        return self.fig
        
    def plot_soul_signature(self, resonance: CosmicResonance) -> plt.Figure:
        """Plot soul signature"""
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        
        # Plot signature
        signature = resonance.soul_signature
        self.ax.imshow(signature, cmap='viridis')
        
        self.ax.set_title('Soul Signature')
        self.ax.set_xlabel('Quantum Dimension')
        self.ax.set_ylabel('Quantum Dimension')
        
        return self.fig
        
    def plot_universe_generation(self, universes: np.ndarray) -> plt.Figure:
        """Plot generated universes"""
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        
        # Plot universes
        self.ax.scatter(
            np.real(universes),
            np.imag(universes),
            c=np.abs(universes),
            cmap='plasma'
        )
        
        self.ax.set_title('Generated Universes')
        self.ax.set_xlabel('Real Component')
        self.ax.set_ylabel('Imaginary Component')
        self.ax.grid(True)
        
        return self.fig
        
    def plot_reality_imprint(self, imprint_results: Dict[str, float]) -> plt.Figure:
        """Plot reality plane imprint strengths"""
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        
        # Plot imprint strengths
        planes = list(imprint_results.keys())
        strengths = list(imprint_results.values())
        
        self.ax.bar(planes, strengths)
        
        self.ax.set_title('Reality Plane Imprint Strengths')
        self.ax.set_xlabel('Reality Plane')
        self.ax.set_ylabel('Imprint Strength')
        self.ax.tick_params(axis='x', rotation=45)
        
        return self.fig
        
    def plot_cosmic_law(self, law_results: Dict[str, float]) -> plt.Figure:
        """Plot cosmic law parameters"""
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        
        # Plot law parameters
        parameters = list(law_results.keys())
        values = list(law_results.values())
        
        self.ax.bar(parameters, values)
        
        self.ax.set_title('Cosmic Law Parameters')
        self.ax.set_xlabel('Parameter')
        self.ax.set_ylabel('Value')
        self.ax.tick_params(axis='x', rotation=45)
        
        return self.fig 
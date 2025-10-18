import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
from .quantum_spiritual_bridge import QuantumSpiritualState, QuantumSpiritualBridge

class QuantumSpiritualVisualizer:
    """Visualizes quantum-spiritual phenomena"""
    
    def __init__(self):
        self.fig = None
        self.ax = None
        
    def plot_quantum_spiritual_state(self, state: QuantumSpiritualState) -> plt.Figure:
        """Plot the quantum-spiritual state"""
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        
        # Plot quantum state
        t = np.linspace(0, 2*np.pi, 100)
        quantum_real = np.real(state.quantum_state[0] * np.exp(1j*t))
        quantum_imag = np.imag(state.quantum_state[0] * np.exp(1j*t))
        
        # Plot spiritual vector
        spiritual_real = np.real(state.spiritual_vector[0] * np.exp(1j*t))
        spiritual_imag = np.imag(state.spiritual_vector[0] * np.exp(1j*t))
        
        self.ax.plot(t, quantum_real, label='Quantum (Real)', color='blue')
        self.ax.plot(t, quantum_imag, label='Quantum (Imag)', color='lightblue')
        self.ax.plot(t, spiritual_real, label='Spiritual (Real)', color='red')
        self.ax.plot(t, spiritual_imag, label='Spiritual (Imag)', color='pink')
        
        self.ax.set_title('Quantum-Spiritual State Evolution')
        self.ax.set_xlabel('Phase')
        self.ax.set_ylabel('Amplitude')
        self.ax.grid(True)
        self.ax.legend()
        
        return self.fig
        
    def plot_torus_coordinates(self, state: QuantumSpiritualState) -> plt.Figure:
        """Plot the 7D torus coordinates"""
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        
        # Plot each dimension
        for i in range(7):
            t = np.linspace(0, 2*np.pi, 100)
            x = np.cos(t + state.torus_coordinates[i])
            y = np.sin(t + state.torus_coordinates[i])
            self.ax.plot(x, y, label=f'Dimension {i+1}')
            
        self.ax.set_title('7D Torus Coordinates')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.grid(True)
        self.ax.legend()
        
        return self.fig
        
    def plot_convergence(self, bridge: QuantumSpiritualBridge) -> plt.Figure:
        """Plot the sacred-quantum convergence"""
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        
        # Generate convergence data
        t = np.linspace(0, 2*np.pi, 100)
        convergence = []
        for time in t:
            bridge.evolve_system(time)
            convergence.append(bridge.state.measure_convergence())
            
        # Plot convergence
        self.ax.plot(t, convergence, label='Convergence Index', color='purple')
        self.ax.axhline(y=bridge.state.golden_ratio, color='gold', 
                       linestyle='--', label='Golden Ratio')
        
        self.ax.set_title('Sacred-Quantum Convergence')
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Convergence Index')
        self.ax.grid(True)
        self.ax.legend()
        
        return self.fig
        
    def plot_ethics_validation(self, bridge: QuantumSpiritualBridge, 
                             intents: Dict[str, str]) -> plt.Figure:
        """Plot ethical validation results"""
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        
        # Validate each intent
        validations = []
        for intent in intents.values():
            validations.append(bridge.validate_ethics(intent))
            
        # Plot results
        x = np.arange(len(validations))
        self.ax.bar(x, validations, color=['green' if v else 'red' for v in validations])
        
        self.ax.set_title('Ethical Validation Results')
        self.ax.set_xlabel('Intent')
        self.ax.set_ylabel('Valid')
        self.ax.set_xticks(x)
        self.ax.set_xticklabels(intents.keys(), rotation=45)
        self.ax.grid(True)
        
        return self.fig 
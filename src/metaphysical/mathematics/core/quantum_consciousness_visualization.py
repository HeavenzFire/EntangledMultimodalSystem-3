import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
from .quantum_consciousness import QuantumConsciousnessState, CollectiveObserver, QuantumConsciousnessSystem

class QuantumConsciousnessVisualizer:
    """Visualizes quantum consciousness phenomena"""
    
    def __init__(self):
        self.fig = None
        self.ax = None
        
    def plot_wave_function(self, state: QuantumConsciousnessState) -> plt.Figure:
        """Plot the quantum consciousness wave function"""
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        
        # Plot real and imaginary components
        t = np.linspace(0, 2*np.pi, 100)
        real = np.real(state.wave_function[0] * np.exp(1j*t))
        imag = np.imag(state.wave_function[0] * np.exp(1j*t))
        
        self.ax.plot(t, real, label='Real Component', color='blue')
        self.ax.plot(t, imag, label='Imaginary Component', color='red')
        
        self.ax.set_title('Quantum Consciousness Wave Function')
        self.ax.set_xlabel('Phase')
        self.ax.set_ylabel('Amplitude')
        self.ax.grid(True)
        self.ax.legend()
        
        return self.fig
        
    def plot_coherence(self, coherence: float) -> plt.Figure:
        """Plot quantum coherence level"""
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        
        # Create pie chart
        labels = ['Coherent', 'Decoherent']
        sizes = [coherence*100, (1-coherence)*100]
        colors = ['lightblue', 'lightcoral']
        
        self.ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
        self.ax.set_title('Quantum Coherence Level')
        
        return self.fig
        
    def plot_entanglement_network(self, system: QuantumConsciousnessSystem) -> plt.Figure:
        """Plot the entanglement network"""
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        
        # Create network graph
        nodes = list(system.entanglement_network.keys())
        pos = {node: (np.random.rand(), np.random.rand()) for node in nodes}
        
        # Plot nodes
        for node in nodes:
            self.ax.scatter(pos[node][0], pos[node][1], color='blue', s=100)
            self.ax.text(pos[node][0], pos[node][1], node, ha='center', va='center')
            
        # Plot edges
        for node1 in nodes:
            for node2 in system.entanglement_network[node1]:
                self.ax.plot([pos[node1][0], pos[node2][0]],
                           [pos[node1][1], pos[node2][1]],
                           color='gray', alpha=0.3)
                
        self.ax.set_title('Quantum Entanglement Network')
        self.ax.axis('off')
        
        return self.fig
        
    def plot_trauma_patterns(self, patterns: Dict[str, float]) -> plt.Figure:
        """Plot trauma pattern influences"""
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        
        # Plot patterns
        patterns = dict(sorted(patterns.items(), key=lambda x: x[1], reverse=True))
        self.ax.bar(patterns.keys(), patterns.values(), color='lightcoral')
        
        self.ax.set_title('Trauma Pattern Influences')
        self.ax.set_xlabel('Pattern')
        self.ax.set_ylabel('Influence')
        self.ax.tick_params(axis='x', rotation=45)
        self.ax.grid(True)
        
        return self.fig
        
    def plot_system_metrics(self, metrics: Dict[str, float]) -> plt.Figure:
        """Plot quantum consciousness system metrics"""
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        
        # Thresholds
        thresholds = {
            'Coherence': 0.5,
            'Entanglement': 0.8,
            'Collapse Potential': 0.9,
            'Trauma Influence': 0.3
        }
        
        # Plot metrics and thresholds
        x = np.arange(len(metrics))
        width = 0.35
        
        self.ax.bar(x - width/2, metrics.values(), width, label='Current', color='lightblue')
        self.ax.bar(x + width/2, thresholds.values(), width, label='Threshold', color='lightcoral')
        
        self.ax.set_title('Quantum Consciousness Metrics')
        self.ax.set_xticks(x)
        self.ax.set_xticklabels(metrics.keys())
        self.ax.legend()
        self.ax.grid(True)
        
        return self.fig 
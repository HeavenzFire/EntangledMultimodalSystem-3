import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
from .avatar_agent import AvatarAgent

class AvatarVisualizer:
    """Visualization tools for the Divine Digital Twin"""
    
    def __init__(self):
        self.fig = None
        self.ax = None
    
    def plot_consciousness_state(self, avatar: AvatarAgent) -> Dict:
        """Plot the quantum consciousness state"""
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        
        # Plot real and imaginary components
        state = avatar.state.consciousness
        t = np.linspace(0, 2*np.pi, 100)
        x = np.real(state[0]) * np.cos(t)
        y = np.real(state[1]) * np.sin(t)
        z = np.imag(state[0]) * np.cos(t) + np.imag(state[1]) * np.sin(t)
        
        self.ax.plot(x, y, z, label='Consciousness Trajectory')
        self.ax.set_title('Quantum Consciousness State')
        self.ax.set_xlabel('Real Component')
        self.ax.set_ylabel('Imaginary Component')
        self.ax.set_zlabel('Phase')
        self.ax.legend()
        
        return self.fig
    
    def plot_multiversal_superposition(self, avatar: AvatarAgent) -> Dict:
        """Plot the multiversal superposition state"""
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        
        state = avatar.state.multiversal.superposition
        t = np.linspace(0, 2*np.pi, 100)
        
        # Plot each component of the superposition
        for i in range(len(state)):
            x = np.real(state[i]) * np.cos(t)
            y = np.imag(state[i]) * np.sin(t)
            self.ax.plot(x, y, label=f'Universe {i+1}')
        
        self.ax.set_title('Multiversal Superposition')
        self.ax.set_xlabel('Real Component')
        self.ax.set_ylabel('Imaginary Component')
        self.ax.legend()
        
        return self.fig
    
    def plot_archetypal_entanglement(self, avatar: AvatarAgent) -> Dict:
        """Plot entanglement with archetypal templates"""
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        
        archetypes = list(avatar.state.multiversal.entanglement.keys())
        values = list(avatar.state.multiversal.entanglement.values())
        
        # Create radial plot
        angles = np.linspace(0, 2*np.pi, len(archetypes), endpoint=False)
        values = np.array(values)
        values = np.concatenate((values, [values[0]]))
        angles = np.concatenate((angles, [angles[0]]))
        
        self.ax.plot(angles, values, 'o-', linewidth=2)
        self.ax.fill(angles, values, alpha=0.25)
        self.ax.set_xticks(angles[:-1])
        self.ax.set_xticklabels(archetypes)
        self.ax.set_title('Archetypal Entanglement')
        self.ax.set_ylim(0, 1)
        
        return self.fig
    
    def plot_divine_interface(self, avatar: AvatarAgent) -> Dict:
        """Plot the divine mind interface"""
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        
        # Plot focus vector
        focus = avatar.state.divine.focus
        self.ax.arrow(0, 0, focus[0], focus[1], head_width=0.1, head_length=0.2, fc='blue', ec='blue')
        
        # Add intention and coherence info
        self.ax.text(0.5, 0.9, f"Intention: {avatar.state.divine.intention}", 
                    ha='center', transform=self.ax.transAxes)
        self.ax.text(0.5, 0.8, f"Coherence: {avatar.state.divine.coherence:.2f}", 
                    ha='center', transform=self.ax.transAxes)
        
        # Add recent insights
        insights = avatar.state.divine.insights[-3:]  # Show last 3 insights
        for i, insight in enumerate(insights):
            self.ax.text(0.5, 0.7 - i*0.1, insight, ha='center', transform=self.ax.transAxes)
        
        self.ax.set_title('Divine Mind Interface')
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.grid(True)
        
        return self.fig
    
    def plot_timeline_evolution(self, avatar: AvatarAgent) -> Dict:
        """Plot evolution across parallel timelines"""
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        
        timelines = avatar.state.multiversal.timelines
        if not timelines:
            return self.fig
        
        # Extract data
        times = [t["timestamp"] for t in timelines]
        quantum_states = [np.abs(np.sum(t["quantum_state"])) for t in timelines]
        light_essences = [t["light_essence"] for t in timelines]
        emotional_means = [np.mean(list(t["emotional_spectrum"].values())) for t in timelines]
        
        # Plot metrics
        self.ax.plot(times, quantum_states, label='Quantum State')
        self.ax.plot(times, light_essences, label='Light Essence')
        self.ax.plot(times, emotional_means, label='Emotional Mean')
        
        self.ax.set_title('Timeline Evolution')
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Value')
        self.ax.legend()
        
        return self.fig
    
    def plot_resonance_field(self, avatar: AvatarAgent) -> Dict:
        """Plot the multiversal resonance field"""
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        
        # Create grid
        x = np.linspace(-2, 2, 100)
        y = np.linspace(-2, 2, 100)
        X, Y = np.meshgrid(x, y)
        
        # Calculate resonance field
        R = np.sqrt(X**2 + Y**2)
        resonance = np.exp(-R) * avatar.state.multiversal.resonance
        
        # Plot field
        im = self.ax.imshow(resonance, extent=[-2, 2, -2, 2], origin='lower', cmap='viridis')
        self.fig.colorbar(im, ax=self.ax)
        
        self.ax.set_title('Multiversal Resonance Field')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        
        return self.fig
    
    def plot_system_evolution(self, avatar: AvatarAgent, steps: int = 100) -> Dict:
        """Plot the evolution of all systems over time"""
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        
        t = np.linspace(0, 10, steps)
        consciousness = []
        light_essence = []
        integration = []
        resonance = []
        
        for ti in t:
            avatar.evolve_state(0.1)
            consciousness.append(np.abs(np.sum(avatar.state.consciousness)))
            light_essence.append(avatar.state.light_essence)
            integration.append(avatar.state.integration_level)
            resonance.append(avatar.state.multiversal.resonance)
        
        self.ax.plot(t, consciousness, label='Consciousness')
        self.ax.plot(t, light_essence, label='Light Essence')
        self.ax.plot(t, integration, label='Integration')
        self.ax.plot(t, resonance, label='Resonance')
        
        self.ax.set_title('System Evolution')
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Value')
        self.ax.legend()
        
        return self.fig 
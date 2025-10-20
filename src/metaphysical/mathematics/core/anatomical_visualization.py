import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
from .anatomical_avatar import AnatomicalState, AnatomicalAvatar

class AnatomicalVisualizer:
    """Visualizes quantum-anatomical phenomena"""
    
    def __init__(self):
        self.fig = None
        self.ax = None
        
    def plot_quantum_anatomical_state(self, state: AnatomicalState) -> plt.Figure:
        """Plot the quantum-anatomical state"""
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        
        # Plot quantum state
        t = np.linspace(0, 2*np.pi, 100)
        quantum_real = np.real(state.quantum_state[0] * np.exp(1j*t))
        quantum_imag = np.imag(state.quantum_state[0] * np.exp(1j*t))
        
        # Plot anatomical vector
        anatomical_real = np.real(state.anatomical_vector[0] * np.exp(1j*t))
        anatomical_imag = np.imag(state.anatomical_vector[0] * np.exp(1j*t))
        
        self.ax.plot(t, quantum_real, label='Quantum (Real)', color='blue')
        self.ax.plot(t, quantum_imag, label='Quantum (Imag)', color='lightblue')
        self.ax.plot(t, anatomical_real, label='Anatomical (Real)', color='red')
        self.ax.plot(t, anatomical_imag, label='Anatomical (Imag)', color='pink')
        
        self.ax.set_title('Quantum-Anatomical State Evolution')
        self.ax.set_xlabel('Phase')
        self.ax.set_ylabel('Amplitude')
        self.ax.grid(True)
        self.ax.legend()
        
        return self.fig
        
    def plot_chakra_coordinates(self, state: AnatomicalState) -> plt.Figure:
        """Plot the 7D chakra coordinates"""
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        
        # Plot each chakra
        chakra_names = ['Root', 'Sacral', 'Solar', 'Heart', 'Throat', 'Third Eye', 'Crown']
        for i in range(7):
            t = np.linspace(0, 2*np.pi, 100)
            x = np.cos(t + state.chakra_coordinates[i])
            y = np.sin(t + state.chakra_coordinates[i])
            self.ax.plot(x, y, label=chakra_names[i])
            
        self.ax.set_title('7D Chakra Coordinates')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.grid(True)
        self.ax.legend()
        
        return self.fig
        
    def plot_alignment(self, avatar: AnatomicalAvatar) -> plt.Figure:
        """Plot the quantum-anatomical alignment"""
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        
        # Generate alignment data
        t = np.linspace(0, 2*np.pi, 100)
        alignment = []
        for time in t:
            avatar.evolve_system(time)
            alignment.append(avatar.state.measure_alignment())
            
        # Plot alignment
        self.ax.plot(t, alignment, label='Alignment Index', color='purple')
        self.ax.axhline(y=avatar.state.golden_ratio, color='gold', 
                       linestyle='--', label='Golden Ratio')
        
        self.ax.set_title('Quantum-Anatomical Alignment')
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Alignment Index')
        self.ax.grid(True)
        self.ax.legend()
        
        return self.fig
        
    def plot_health_validation(self, avatar: AnatomicalAvatar, 
                             features: Dict[str, str]) -> plt.Figure:
        """Plot health validation results"""
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        
        # Validate each feature
        validations = []
        for feature in features.values():
            validations.append(avatar.validate_health(feature))
            
        # Plot results
        x = np.arange(len(validations))
        self.ax.bar(x, validations, color=['green' if v else 'red' for v in validations])
        
        self.ax.set_title('Health Validation Results')
        self.ax.set_xlabel('Feature')
        self.ax.set_ylabel('Healthy')
        self.ax.set_xticks(x)
        self.ax.set_xticklabels(features.keys(), rotation=45)
        self.ax.grid(True)
        
        return self.fig
        
    def plot_pathology(self, avatar: AnatomicalAvatar, feature: str) -> plt.Figure:
        """Plot pathological growth simulation"""
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        
        if feature in avatar.parser.features:
            vertices = avatar.parser.features[feature]
            x = vertices[:, 0]
            y = vertices[:, 1]
            z = vertices[:, 2]
            
            # Create 3D scatter plot
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.ax.scatter(x, y, z, c='red', alpha=0.6)
            
            self.ax.set_title(f'Pathological Growth: {feature}')
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z')
            
        return self.fig 
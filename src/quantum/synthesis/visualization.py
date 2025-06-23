import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Dict, Optional
from .quantum_sacred import QuantumSacredSynthesis, QuantumState, SacredConfig

class QuantumSacredVisualizer:
    """Visualization system for quantum-sacred synthesis"""
    
    def __init__(self, synthesis: QuantumSacredSynthesis):
        self.synthesis = synthesis
        self.fig = plt.figure(figsize=(15, 10))
        self.setup_plots()
        
    def setup_plots(self):
        """Initialize all visualization components"""
        # Create subplots
        self.ax1 = self.fig.add_subplot(221, projection='3d')  # Merkaba field
        self.ax2 = self.fig.add_subplot(222)  # State transitions
        self.ax3 = self.fig.add_subplot(223)  # Harmonic resonance
        self.ax4 = self.fig.add_subplot(224)  # Christos grid
        
        # Set titles and labels
        self.ax1.set_title('Merkaba Field Rotation')
        self.ax2.set_title('State Transition Matrix')
        self.ax3.set_title('Harmonic Resonance Pattern')
        self.ax4.set_title('Christos Grid Projection')
        
        # Initialize data containers
        self.merkaba_data = []
        self.state_history = []
        self.harmonic_data = []
        self.christos_data = []
        
    def update_merkaba_field(self):
        """Update 3D merkaba field visualization"""
        self.ax1.clear()
        
        # Generate merkaba vertices
        phi = self.synthesis.config.phi_resonance
        vertices = self._generate_merkaba_vertices(phi)
        
        # Apply rotation
        rotation = self.synthesis.merkaba_rotation
        rotated_vertices = self._rotate_vertices(vertices, rotation)
        
        # Plot merkaba
        self._plot_merkaba(rotated_vertices)
        
    def _generate_merkaba_vertices(self, phi: float) -> np.ndarray:
        """Generate merkaba star vertices"""
        # Create base tetrahedron
        base = np.array([
            [1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]
        ])
        
        # Scale by phi
        scaled = base * phi
        
        # Create dual tetrahedron
        dual = -scaled
        
        # Combine vertices
        return np.vstack([scaled, dual])
        
    def _rotate_vertices(self, vertices: np.ndarray, angle: float) -> np.ndarray:
        """Apply rotation to vertices"""
        # Convert angle to radians
        theta = np.radians(angle)
        
        # Rotation matrices
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ])
        
        Ry = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
        
        # Apply rotations
        rotated = vertices @ Rx @ Ry
        return rotated
        
    def _plot_merkaba(self, vertices: np.ndarray):
        """Plot merkaba star with connections"""
        # Plot vertices
        self.ax1.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                        c='#c084fc', s=100)
        
        # Connect vertices
        for i in range(len(vertices)):
            for j in range(i+1, len(vertices)):
                self.ax1.plot([vertices[i, 0], vertices[j, 0]],
                            [vertices[i, 1], vertices[j, 1]],
                            [vertices[i, 2], vertices[j, 2]],
                            c='#a855f7', alpha=0.5)
        
        # Set equal aspect ratio
        self.ax1.set_box_aspect([1, 1, 1])
        
    def update_state_transitions(self):
        """Update state transition matrix visualization"""
        self.ax2.clear()
        
        # Get current transition matrix
        matrix = self.synthesis.transition_matrix
        
        # Create heatmap
        im = self.ax2.imshow(matrix, cmap='viridis', vmin=0, vmax=1)
        
        # Add colorbar
        plt.colorbar(im, ax=self.ax2)
        
        # Add state labels
        states = [s.name for s in QuantumState]
        self.ax2.set_xticks(range(len(states)))
        self.ax2.set_yticks(range(len(states)))
        self.ax2.set_xticklabels(states, rotation=45)
        self.ax2.set_yticklabels(states)
        
    def update_harmonic_resonance(self):
        """Update harmonic resonance pattern visualization"""
        self.ax3.clear()
        
        # Get current harmonic pattern
        pattern = self._apply_christos_harmonic()
        
        # Plot pattern
        self.ax3.plot(np.real(pattern), label='Real', c='#c084fc')
        self.ax3.plot(np.imag(pattern), label='Imaginary', c='#a855f7')
        
        # Add legend and grid
        self.ax3.legend()
        self.ax3.grid(True, alpha=0.3)
        
    def _apply_christos_harmonic(self) -> np.ndarray:
        """Generate Christos grid harmonic pattern"""
        frequency = self.synthesis.config.christos_frequency
        pattern = np.array([
            np.exp(2j * np.pi * frequency * k / 12)
            for k in range(12)
        ])
        return pattern
        
    def update_christos_grid(self):
        """Update Christos grid projection"""
        self.ax4.clear()
        
        # Generate grid points
        x = np.linspace(-1, 1, 12)
        y = np.linspace(-1, 1, 12)
        X, Y = np.meshgrid(x, y)
        
        # Calculate grid values
        Z = np.sin(np.pi * X) * np.sin(np.pi * Y)
        
        # Plot grid
        self.ax4.contourf(X, Y, Z, levels=20, cmap='viridis')
        
        # Add grid lines
        self.ax4.grid(True, alpha=0.3)
        
    def update(self, frame):
        """Update all visualizations"""
        self.update_merkaba_field()
        self.update_state_transitions()
        self.update_harmonic_resonance()
        self.update_christos_grid()
        
        # Adjust layout
        plt.tight_layout()
        
    def animate(self, frames: int = 100):
        """Create animation of quantum-sacred synthesis"""
        anim = FuncAnimation(self.fig, self.update, frames=frames,
                           interval=100, blit=False)
        return anim
        
    def save_animation(self, filename: str, frames: int = 100):
        """Save animation to file"""
        anim = self.animate(frames)
        anim.save(filename, writer='pillow', fps=10)
        
    def show(self):
        """Display visualization"""
        plt.show()

# Example usage
if __name__ == "__main__":
    # Initialize synthesis system
    config = SacredConfig()
    synthesis = QuantumSacredSynthesis(config)
    
    # Create visualizer
    visualizer = QuantumSacredVisualizer(synthesis)
    
    # Update synthesis state
    synthesis.update_transition_matrix(0.8, 0.2)
    synthesis.merkaba_rotation = 45.0
    
    # Show visualization
    visualizer.show() 
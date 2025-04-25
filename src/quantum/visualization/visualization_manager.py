import numpy as np
from typing import Dict, List, Any
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from ..geometry.sacred_geometry import SacredGeometry

class QuantumVisualizationManager:
    """Manages visualization of quantum states and sacred geometry"""
    
    def __init__(self):
        self.sacred_geometry = SacredGeometry()
        self.fig = None
        self.ax = None
        self.animation = None
        
    def initialize_plot(self):
        """Initialize 3D plot for visualization"""
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        self.ax.set_zlim(-2, 2)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        
    def plot_quantum_state(self, state: Dict[str, Any]):
        """Plot quantum state in 3D space"""
        if self.fig is None:
            self.initialize_plot()
            
        self.ax.clear()
        
        # Plot quantum state points
        points = self._extract_state_points(state)
        self.ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                       c='blue', alpha=0.6, s=100)
        
        # Plot sacred geometry patterns
        self._plot_sacred_geometry()
        
        plt.draw()
        plt.pause(0.001)
        
    def _extract_state_points(self, state: Dict[str, Any]) -> np.ndarray:
        """Extract 3D points from quantum state"""
        points = []
        for key, value in state.items():
            if isinstance(value, dict) and 'position' in value:
                pos = value['position']
                points.append([pos['x'], pos['y'], pos['z']])
        return np.array(points)
        
    def _plot_sacred_geometry(self):
        """Plot sacred geometry patterns"""
        # Plot Metatron's Cube
        cube_points = self.sacred_geometry.get_metatron_cube_points()
        self.ax.plot(cube_points[:, 0], cube_points[:, 1], cube_points[:, 2],
                    'r-', alpha=0.3)
        
        # Plot Flower of Life
        flower_points = self.sacred_geometry.get_flower_of_life_points()
        self.ax.plot(flower_points[:, 0], flower_points[:, 1], flower_points[:, 2],
                    'g-', alpha=0.3)
        
    def create_animation(self, state_history: List[Dict[str, Any]]):
        """Create animation of quantum state evolution"""
        if self.fig is None:
            self.initialize_plot()
            
        def update(frame):
            self.plot_quantum_state(state_history[frame])
            return self.ax.artists
            
        self.animation = FuncAnimation(self.fig, update, 
                                     frames=len(state_history),
                                     interval=200, blit=True)
        
    def save_animation(self, filename: str):
        """Save animation to file"""
        if self.animation:
            self.animation.save(filename, writer='pillow', fps=5)
            
    def plot_metrics(self, metrics: Dict[str, List[float]]):
        """Plot quantum state metrics over time"""
        plt.figure(figsize=(10, 6))
        
        for metric_name, values in metrics.items():
            plt.plot(values, label=metric_name)
            
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Quantum State Metrics')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def plot_entanglement(self, entangled_states: Dict[str, Any]):
        """Plot entanglement network between agents"""
        plt.figure(figsize=(8, 8))
        
        # Create network graph
        for session_id, data in entangled_states.items():
            agents = data['agents']
            strength = data['state'].get('entanglement_strength', 0)
            
            # Plot connection with width proportional to entanglement strength
            plt.plot([0, 1], [0, 1], 'b-', 
                    linewidth=strength * 5,
                    alpha=0.5)
            
            # Plot agent nodes
            plt.plot(0, 0, 'ro', markersize=20)
            plt.plot(1, 1, 'ro', markersize=20)
            
            # Add labels
            plt.text(0, 0, agents[0], ha='center', va='center')
            plt.text(1, 1, agents[1], ha='center', va='center')
            
        plt.title('Quantum Entanglement Network')
        plt.axis('off')
        plt.show()
        
    def close(self):
        """Close all plots"""
        if self.fig:
            plt.close(self.fig)
        plt.close('all') 
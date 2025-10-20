import unittest
import numpy as np
from typing import Dict, List, Any
from src.quantum.visualization.visualization_manager import QuantumVisualizationManager

class TestQuantumVisualizationManager(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.visualization_manager = QuantumVisualizationManager()
        
    def test_initialization(self):
        """Test initialization of visualization manager"""
        self.assertIsNotNone(self.visualization_manager.sacred_geometry)
        self.assertIsNone(self.visualization_manager.fig)
        self.assertIsNone(self.visualization_manager.ax)
        self.assertIsNone(self.visualization_manager.animation)
        
    def test_plot_quantum_state(self):
        """Test plotting quantum state"""
        # Create test state
        test_state = {
            'agent1': {
                'position': {'x': 0.5, 'y': 0.5, 'z': 0.5},
                'phase': 0.0
            },
            'agent2': {
                'position': {'x': -0.5, 'y': -0.5, 'z': -0.5},
                'phase': np.pi
            }
        }
        
        # Plot state
        self.visualization_manager.plot_quantum_state(test_state)
        
        # Verify plot was created
        self.assertIsNotNone(self.visualization_manager.fig)
        self.assertIsNotNone(self.visualization_manager.ax)
        
    def test_create_animation(self):
        """Test creating animation of state evolution"""
        # Create test state history
        state_history = [
            {
                'agent1': {
                    'position': {'x': 0.5, 'y': 0.5, 'z': 0.5},
                    'phase': 0.0
                }
            },
            {
                'agent1': {
                    'position': {'x': 0.6, 'y': 0.6, 'z': 0.6},
                    'phase': np.pi/4
                }
            }
        ]
        
        # Create animation
        self.visualization_manager.create_animation(state_history)
        
        # Verify animation was created
        self.assertIsNotNone(self.visualization_manager.animation)
        
    def test_plot_metrics(self):
        """Test plotting quantum state metrics"""
        # Create test metrics
        metrics = {
            'entanglement_strength': [0.5, 0.6, 0.7],
            'coherence': [0.8, 0.9, 1.0]
        }
        
        # Plot metrics
        self.visualization_manager.plot_metrics(metrics)
        
    def test_plot_entanglement(self):
        """Test plotting entanglement network"""
        # Create test entangled states
        entangled_states = {
            'session1': {
                'agents': ['agent1', 'agent2'],
                'state': {
                    'entanglement_strength': 0.8
                }
            }
        }
        
        # Plot entanglement
        self.visualization_manager.plot_entanglement(entangled_states)
        
    def test_close(self):
        """Test closing all plots"""
        # Create some plots
        self.visualization_manager.initialize_plot()
        
        # Close plots
        self.visualization_manager.close()
        
        # Verify plots are closed
        self.assertIsNone(self.visualization_manager.fig)
        
if __name__ == '__main__':
    unittest.main() 
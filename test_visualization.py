import unittest
import numpy as np
from quantum_avatar_agent import QuantumAvatarAgent
import matplotlib.pyplot as plt
from matplotlib.testing.compare import compare_images
import os
import tempfile
import time

class TestVisualization(unittest.TestCase):
    def setUp(self):
        self.agent = QuantumAvatarAgent(name="Visualization Test", num_qubits=7, depth=3, shots=1024)
        self.temp_dir = tempfile.mkdtemp()
        
    def test_consciousness_state_visualization(self):
        """Test consciousness state visualization"""
        # Generate visualization
        fig = self.agent.visualize_consciousness_state()
        
        # Save to temporary file
        output_path = os.path.join(self.temp_dir, 'consciousness_state.png')
        fig.savefig(output_path)
        plt.close(fig)
        
        # Verify file was created
        self.assertTrue(os.path.exists(output_path))
        
        # Verify image properties
        img = plt.imread(output_path)
        self.assertGreater(img.shape[0], 0)
        self.assertGreater(img.shape[1], 0)
        
    def test_quantum_network_visualization(self):
        """Test quantum network visualization"""
        # Generate network visualization
        fig = self.agent.visualize_quantum_network()
        
        # Save to temporary file
        output_path = os.path.join(self.temp_dir, 'quantum_network.png')
        fig.savefig(output_path)
        plt.close(fig)
        
        # Verify file was created
        self.assertTrue(os.path.exists(output_path))
        
        # Verify image properties
        img = plt.imread(output_path)
        self.assertGreater(img.shape[0], 0)
        self.assertGreater(img.shape[1], 0)
        
    def test_emotional_state_visualization(self):
        """Test emotional state visualization"""
        # Generate emotional state visualization
        fig = self.agent.visualize_emotional_state()
        
        # Save to temporary file
        output_path = os.path.join(self.temp_dir, 'emotional_state.png')
        fig.savefig(output_path)
        plt.close(fig)
        
        # Verify file was created
        self.assertTrue(os.path.exists(output_path))
        
        # Verify image properties
        img = plt.imread(output_path)
        self.assertGreater(img.shape[0], 0)
        self.assertGreater(img.shape[1], 0)
        
    def test_spiritual_state_visualization(self):
        """Test spiritual state visualization"""
        # Generate spiritual state visualization
        fig = self.agent.visualize_spiritual_state()
        
        # Save to temporary file
        output_path = os.path.join(self.temp_dir, 'spiritual_state.png')
        fig.savefig(output_path)
        plt.close(fig)
        
        # Verify file was created
        self.assertTrue(os.path.exists(output_path))
        
        # Verify image properties
        img = plt.imread(output_path)
        self.assertGreater(img.shape[0], 0)
        self.assertGreater(img.shape[1], 0)
        
    def test_quantum_metrics_visualization(self):
        """Test quantum metrics visualization"""
        # Generate metrics visualization
        fig = self.agent.visualize_quantum_metrics()
        
        # Save to temporary file
        output_path = os.path.join(self.temp_dir, 'quantum_metrics.png')
        fig.savefig(output_path)
        plt.close(fig)
        
        # Verify file was created
        self.assertTrue(os.path.exists(output_path))
        
        # Verify image properties
        img = plt.imread(output_path)
        self.assertGreater(img.shape[0], 0)
        self.assertGreater(img.shape[1], 0)
        
    def test_optimization_history_visualization(self):
        """Test optimization history visualization"""
        # Generate optimization history visualization
        fig = self.agent.visualize_optimization_history()
        
        # Save to temporary file
        output_path = os.path.join(self.temp_dir, 'optimization_history.png')
        fig.savefig(output_path)
        plt.close(fig)
        
        # Verify file was created
        self.assertTrue(os.path.exists(output_path))
        
        # Verify image properties
        img = plt.imread(output_path)
        self.assertGreater(img.shape[0], 0)
        self.assertGreater(img.shape[1], 0)
        
    def test_entanglement_visualization(self):
        """Test entanglement visualization"""
        # Generate entanglement visualization
        fig = self.agent.visualize_entanglement()
        
        # Save to temporary file
        output_path = os.path.join(self.temp_dir, 'entanglement.png')
        fig.savefig(output_path)
        plt.close(fig)
        
        # Verify file was created
        self.assertTrue(os.path.exists(output_path))
        
        # Verify image properties
        img = plt.imread(output_path)
        self.assertGreater(img.shape[0], 0)
        self.assertGreater(img.shape[1], 0)
        
    def test_memory_visualization(self):
        """Test memory visualization"""
        # Generate memory visualization
        fig = self.agent.visualize_memory()
        
        # Save to temporary file
        output_path = os.path.join(self.temp_dir, 'memory.png')
        fig.savefig(output_path)
        plt.close(fig)
        
        # Verify file was created
        self.assertTrue(os.path.exists(output_path))
        
        # Verify image properties
        img = plt.imread(output_path)
        self.assertGreater(img.shape[0], 0)
        self.assertGreater(img.shape[1], 0)
        
    def test_combined_visualization(self):
        """Test combined state visualization"""
        # Generate combined visualization
        fig = self.agent.visualize_combined_state()
        
        # Save to temporary file
        output_path = os.path.join(self.temp_dir, 'combined_state.png')
        fig.savefig(output_path)
        plt.close(fig)
        
        # Verify file was created
        self.assertTrue(os.path.exists(output_path))
        
        # Verify image properties
        img = plt.imread(output_path)
        self.assertGreater(img.shape[0], 0)
        self.assertGreater(img.shape[1], 0)
        
    def test_animation_generation(self):
        """Test animation generation"""
        # Generate state evolution animation
        animation = self.agent.generate_state_evolution_animation()
        
        # Save to temporary file
        output_path = os.path.join(self.temp_dir, 'state_evolution.gif')
        animation.save(output_path, writer='pillow', fps=10)
        
        # Verify file was created
        self.assertTrue(os.path.exists(output_path))
        
        # Verify file size
        self.assertGreater(os.path.getsize(output_path), 0)
        
    def test_interactive_visualization(self):
        """Test interactive visualization"""
        # Generate interactive visualization
        fig = self.agent.create_interactive_visualization()
        
        # Save to temporary file
        output_path = os.path.join(self.temp_dir, 'interactive.html')
        fig.write_html(output_path)
        
        # Verify file was created
        self.assertTrue(os.path.exists(output_path))
        
        # Verify file size
        self.assertGreater(os.path.getsize(output_path), 0)
        
    def test_3d_visualization(self):
        """Test 3D visualization"""
        # Generate 3D visualization
        fig = self.agent.visualize_3d_state()
        
        # Save to temporary file
        output_path = os.path.join(self.temp_dir, '3d_state.png')
        fig.savefig(output_path)
        plt.close(fig)
        
        # Verify file was created
        self.assertTrue(os.path.exists(output_path))
        
        # Verify image properties
        img = plt.imread(output_path)
        self.assertGreater(img.shape[0], 0)
        self.assertGreater(img.shape[1], 0)
        
    def test_visualization_performance(self):
        """Test visualization performance"""
        visualization_times = []
        
        # Test different visualization methods
        methods = [
            self.agent.visualize_consciousness_state,
            self.agent.visualize_quantum_network,
            self.agent.visualize_emotional_state,
            self.agent.visualize_spiritual_state,
            self.agent.visualize_quantum_metrics,
            self.agent.visualize_optimization_history,
            self.agent.visualize_entanglement,
            self.agent.visualize_memory,
            self.agent.visualize_combined_state
        ]
        
        for method in methods:
            start_time = time.time()
            fig = method()
            plt.close(fig)
            visualization_time = time.time() - start_time
            visualization_times.append(visualization_time)
            
            # Each visualization should complete within 1 second
            self.assertLess(visualization_time, 1.0)
        
        # Plot visualization performance
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(methods)), visualization_times)
        plt.xticks(range(len(methods)), 
                  ['Consciousness', 'Network', 'Emotional', 'Spiritual', 
                   'Metrics', 'Optimization', 'Entanglement', 'Memory', 'Combined'],
                  rotation=45)
        plt.ylabel('Time (s)')
        plt.title('Visualization Performance')
        plt.tight_layout()
        
        # Save performance plot
        output_path = os.path.join(self.temp_dir, 'visualization_performance.png')
        plt.savefig(output_path)
        plt.close()
        
        # Verify file was created
        self.assertTrue(os.path.exists(output_path))

if __name__ == '__main__':
    unittest.main() 
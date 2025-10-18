import pytest
import numpy as np
from src.quantum.synthesis.visualization import QuantumSacredVisualizer
from src.quantum.synthesis.quantum_sacred import QuantumSacredSynthesis, SacredConfig

def test_visualizer_initialization():
    """Test visualizer initialization"""
    synthesis = QuantumSacredSynthesis()
    visualizer = QuantumSacredVisualizer(synthesis)
    
    assert visualizer.synthesis == synthesis
    assert visualizer.fig is not None
    assert len(visualizer.fig.axes) == 4

def test_merkaba_vertices_generation():
    """Test merkaba vertices generation"""
    synthesis = QuantumSacredSynthesis()
    visualizer = QuantumSacredVisualizer(synthesis)
    
    vertices = visualizer._generate_merkaba_vertices(1.618033988749895)
    
    assert vertices.shape == (8, 3)  # 8 vertices (4 base + 4 dual)
    assert np.allclose(np.abs(vertices), 1.618033988749895)  # Scaled by phi

def test_vertex_rotation():
    """Test vertex rotation"""
    synthesis = QuantumSacredSynthesis()
    visualizer = QuantumSacredVisualizer(synthesis)
    
    vertices = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    rotated = visualizer._rotate_vertices(vertices, 90)
    
    # Check rotation properties
    assert rotated.shape == vertices.shape
    assert not np.array_equal(vertices, rotated)
    assert np.allclose(np.linalg.norm(rotated, axis=1), 
                      np.linalg.norm(vertices, axis=1))

def test_christos_harmonic():
    """Test Christos grid harmonic generation"""
    synthesis = QuantumSacredSynthesis()
    visualizer = QuantumSacredVisualizer(synthesis)
    
    pattern = visualizer._apply_christos_harmonic()
    
    assert pattern.shape == (12,)
    assert np.all(np.abs(pattern) <= 1.0)  # Normalized
    assert np.iscomplexobj(pattern)  # Complex values

def test_visualization_update():
    """Test visualization update"""
    synthesis = QuantumSacredSynthesis()
    visualizer = QuantumSacredVisualizer(synthesis)
    
    # Update synthesis state
    synthesis.update_transition_matrix(0.8, 0.2)
    synthesis.merkaba_rotation = 45.0
    
    # Update visualization
    visualizer.update(0)
    
    # Check that plots were updated
    assert visualizer.ax1.collections  # Merkaba field points
    assert visualizer.ax2.images  # State transition heatmap
    assert visualizer.ax3.lines  # Harmonic resonance lines
    assert visualizer.ax4.collections  # Christos grid contours

def test_animation_creation():
    """Test animation creation"""
    synthesis = QuantumSacredSynthesis()
    visualizer = QuantumSacredVisualizer(synthesis)
    
    anim = visualizer.animate(frames=10)
    
    assert anim is not None
    assert anim.fig == visualizer.fig
    assert anim._frames == 10

def test_sacred_config_visualization():
    """Test visualization with custom sacred configuration"""
    config = SacredConfig(
        phi_resonance=1.5,
        vortex_sequence=[2, 4, 6],
        christos_frequency=440.0
    )
    synthesis = QuantumSacredSynthesis(config)
    visualizer = QuantumSacredVisualizer(synthesis)
    
    # Update visualization
    visualizer.update(0)
    
    # Check that configuration affects visualization
    vertices = visualizer._generate_merkaba_vertices(config.phi_resonance)
    assert np.allclose(np.abs(vertices), 1.5)  # Custom phi scaling
    
    pattern = visualizer._apply_christos_harmonic()
    assert pattern.shape == (12,)
    assert np.iscomplexobj(pattern)

def test_state_transition_visualization():
    """Test state transition matrix visualization"""
    synthesis = QuantumSacredSynthesis()
    visualizer = QuantumSacredVisualizer(synthesis)
    
    # Update transition matrix
    synthesis.update_transition_matrix(0.9, 0.1)
    visualizer.update_state_transitions()
    
    # Check heatmap properties
    assert visualizer.ax2.images[0].get_array().shape == (5, 5)  # 5 states
    assert np.all(visualizer.ax2.images[0].get_array() >= 0)
    assert np.all(visualizer.ax2.images[0].get_array() <= 1)

def test_harmonic_resonance_visualization():
    """Test harmonic resonance pattern visualization"""
    synthesis = QuantumSacredSynthesis()
    visualizer = QuantumSacredVisualizer(synthesis)
    
    visualizer.update_harmonic_resonance()
    
    # Check plot properties
    assert len(visualizer.ax3.lines) == 2  # Real and imaginary components
    assert visualizer.ax3.legend_ is not None
    assert visualizer.ax3.grid()

def test_christos_grid_visualization():
    """Test Christos grid visualization"""
    synthesis = QuantumSacredSynthesis()
    visualizer = QuantumSacredVisualizer(synthesis)
    
    visualizer.update_christos_grid()
    
    # Check grid properties
    assert visualizer.ax4.collections  # Contour plot
    assert visualizer.ax4.grid()
    assert visualizer.ax4.get_xlim() == (-1, 1)
    assert visualizer.ax4.get_ylim() == (-1, 1)

if __name__ == '__main__':
    pytest.main([__file__]) 
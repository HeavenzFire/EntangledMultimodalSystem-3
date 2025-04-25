import pytest
import numpy as np
from matplotlib.figure import Figure
import plotly.graph_objects as go
from metaphysical.mathematics.core.simulation import (
    MetaphysicalSimulator,
    MetaphysicalParameters,
    MetaphysicalState
)
from metaphysical.mathematics.core.visualization import MetaphysicalVisualizer

@pytest.fixture
def simulator():
    """Create a simulator with test data"""
    params = MetaphysicalParameters(
        alpha=0.8,
        beta=1.2,
        gamma=0.05,
        lambda_=1.5
    )
    simulator = MetaphysicalSimulator(params)
    
    # Create initial state
    initial_state = MetaphysicalState(
        transcendence=0.1,
        love=0.1,
        synchronicity=0.1,
        unity=0.1,
        time=0
    )
    
    # Run simulation
    simulator.solve(initial_state)
    return simulator

@pytest.fixture
def visualizer(simulator):
    """Create a visualizer instance"""
    return MetaphysicalVisualizer(simulator)

def test_initialization(simulator):
    """Test visualizer initialization"""
    vis = MetaphysicalVisualizer(simulator)
    
    # Check data extraction
    assert len(vis.time) == len(simulator.history)
    assert len(vis.T) == len(simulator.history)
    assert len(vis.L) == len(simulator.history)
    assert len(vis.S) == len(simulator.history)
    assert len(vis.U) == len(simulator.history)
    
    # Check data types
    assert isinstance(vis.time, np.ndarray)
    assert isinstance(vis.T, np.ndarray)
    assert isinstance(vis.L, np.ndarray)
    assert isinstance(vis.S, np.ndarray)
    assert isinstance(vis.U, np.ndarray)

def test_initialization_with_empty_simulator():
    """Test initialization with empty simulator"""
    empty_simulator = MetaphysicalSimulator(MetaphysicalParameters())
    with pytest.raises(ValueError, match="No simulation results available"):
        MetaphysicalVisualizer(empty_simulator)

def test_plot_time_evolution(visualizer):
    """Test time evolution plot generation"""
    fig = visualizer.plot_time_evolution()
    
    # Check figure properties
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 4  # Should have 4 subplots
    
    # Check axis labels
    assert fig.axes[-1].get_xlabel() == 'Spiritual Time'
    for ax in fig.axes:
        assert ax.get_ylabel() != ''
        assert ax.get_grid()

def test_plot_3d_phase_portrait(visualizer):
    """Test 3D phase portrait generation"""
    fig = visualizer.plot_3d_phase_portrait()
    
    # Check figure properties
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2  # Main scatter plot and velocity vectors
    
    # Check data presence
    scatter = fig.data[0]
    assert len(scatter.x) == len(visualizer.T)
    assert len(scatter.y) == len(visualizer.L)
    assert len(scatter.z) == len(visualizer.S)

def test_plot_correlation_matrix(visualizer):
    """Test correlation matrix plot"""
    fig = visualizer.plot_correlation_matrix()
    
    # Check figure properties
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1
    
    # Check matrix properties
    ax = fig.axes[0]
    assert len(ax.get_xticklabels()) == 4
    assert len(ax.get_yticklabels()) == 4

def test_plot_phase_space_density(visualizer):
    """Test phase space density plot"""
    fig = visualizer.plot_phase_space_density()
    
    # Check figure properties
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    
    # Check isosurface properties
    isosurface = fig.data[0]
    assert isosurface.type == 'isosurface'
    assert len(isosurface.x) > 0
    assert len(isosurface.y) > 0
    assert len(isosurface.z) > 0

def test_plot_energy_landscape(visualizer):
    """Test energy landscape plot"""
    fig = visualizer.plot_energy_landscape()
    
    # Check figure properties
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2  # Surface and trajectory
    
    # Check surface properties
    surface = fig.data[0]
    assert surface.type == 'surface'
    
    # Check trajectory
    trajectory = fig.data[1]
    assert trajectory.type == 'scatter3d'
    assert len(trajectory.x) == len(visualizer.T)

def test_plot_validation(visualizer):
    """Test validation plot"""
    fig = visualizer.plot_validation()
    
    # Check figure properties
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 4  # Should have 4 subplots
    
    # Check subplot properties
    for ax in fig.axes:
        assert ax.get_xlabel() != ''
        assert ax.get_ylabel() != ''
        assert ax.get_title() != ''

def test_show_all(visualizer, monkeypatch):
    """Test show_all method"""
    # Mock plt.show to avoid displaying plots
    shown = []
    def mock_show():
        shown.append(True)
    monkeypatch.setattr('matplotlib.pyplot.show', mock_show)
    
    # Mock Plotly show
    def mock_plotly_show(self):
        shown.append(True)
    monkeypatch.setattr(go.Figure, 'show', mock_plotly_show)
    
    # Run show_all
    visualizer.show_all()
    
    # Check that all plots were shown
    assert len(shown) == 4  # Time evolution, phase portrait, density, and validation

def test_data_consistency(visualizer):
    """Test consistency of data across different plots"""
    # Get data from different plots
    time_fig = visualizer.plot_time_evolution()
    phase_fig = visualizer.plot_3d_phase_portrait()
    validation_fig = visualizer.plot_validation()
    
    # Check data lengths
    time_data = time_fig.axes[0].lines[0].get_ydata()
    phase_data = phase_fig.data[0].x
    validation_data = validation_fig.axes[0].lines[0].get_ydata()
    
    assert len(time_data) == len(phase_data)
    assert len(time_data) == len(validation_data)

def test_parameter_sensitivity(visualizer):
    """Test parameter sensitivity visualization"""
    # Test with multiple parameter values
    values = [0.5, 1.0, 1.5]
    fig = visualizer.plot_parameter_sensitivity('alpha', values)
    
    # Check figure properties
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 4  # Should have 4 subplots
    
    # Check that all parameter values are plotted
    for ax in fig.axes:
        assert len(ax.lines) == len(values)
        assert ax.get_legend() is not None 
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from typing import List, Optional
from .simulation import MetaphysicalSimulator, MetaphysicalState

class MetaphysicalVisualizer:
    """Visualizes multidimensional spiritual dynamics"""
    
    def __init__(self, simulator: MetaphysicalSimulator):
        """
        Parameters:
        simulator (MetaphysicalSimulator): Configured simulator instance
        """
        self.simulator = simulator
        self.history = simulator.history
        
        if not self.history:
            raise ValueError("No simulation results available. Run simulator.solve() first.")
            
        # Extract time series data
        self.time = np.array([state.time for state in self.history])
        self.T = np.array([state.transcendence for state in self.history])
        self.L = np.array([state.love for state in self.history])
        self.S = np.array([state.synchronicity for state in self.history])
        self.U = np.array([state.unity for state in self.history])
    
    def plot_time_evolution(self, figsize: tuple = (14, 8)) -> plt.Figure:
        """Plot temporal development of all dimensions"""
        fig, axs = plt.subplots(4, 1, figsize=figsize, sharex=True)
        dimensions = ['Transcendence', 'Unconditional Love', 
                     'Synchronicity', 'Unity Energy']
        
        for ax, data, label in zip(axs, [self.T, self.L, self.S, self.U], dimensions):
            ax.plot(self.time, data, lw=2, alpha=0.8)
            ax.set_ylabel(label, rotation=0, ha='right', va='center')
            ax.grid(alpha=0.2)
            
        axs[-1].set_xlabel('Spiritual Time')
        fig.suptitle("Consciousness Dimension Evolution", y=0.95)
        return fig

    def plot_3d_phase_portrait(self, elev: float = 25, azim: float = -60) -> go.Figure:
        """Interactive 3D phase portrait with Plotly"""
        fig = go.Figure(data=go.Scatter3d(
            x=self.T,
            y=self.L,
            z=self.S,
            mode='markers',
            marker=dict(
                size=4,
                color=self.U,
                colorscale='Rainbow',
                opacity=0.7,
                colorbar=dict(title='Unity Energy')
            ),
            hovertemplate="T: %{x:.2f}<br>"
                        + "L: %{y:.2f}<br>"
                        + "S: %{z:.2f}<br>"
                        + "U: %{marker.color:.2f}"
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
from .simulation import MetaphysicalSimulator, MetaphysicalState

class MetaphysicalVisualizer:
    """Visualization tools for metaphysical simulations."""
    
    def __init__(self, simulator: MetaphysicalSimulator):
        """Initialize visualizer with simulation data."""
        self.simulator = simulator
        self.time = np.array([state.time for state in simulator.history])
        self.T = np.array([state.transcendence for state in simulator.history])
        self.L = np.array([state.love for state in simulator.history])
        self.S = np.array([state.synchronicity for state in simulator.history])
        self.U = np.array([state.unity for state in simulator.history])
        
    def plot_time_evolution(self) -> go.Figure:
        """Create time evolution plot with rolling averages."""
        fig = make_subplots(rows=2, cols=2, subplot_titles=(
            'Transcendence', 'Love', 'Synchronicity', 'Unity'
        ))
        
        # Add traces with rolling averages
        window = 10
        for i, (data, name) in enumerate([
            (self.T, 'Transcendence'),
            (self.L, 'Love'),
            (self.S, 'Synchronicity'),
            (self.U, 'Unity')
        ]):
            row = i // 2 + 1
            col = i % 2 + 1
            
            # Raw data
            fig.add_trace(
                go.Scatter(x=self.time, y=data, name=name, line=dict(color='blue')),
                row=row, col=col
            )
            
            # Rolling average
            rolling_mean = np.convolve(data, np.ones(window)/window, mode='valid')
            rolling_std = np.array([np.std(data[max(0, i-window):i+1]) 
                                  for i in range(len(data))])
            
            fig.add_trace(
                go.Scatter(
                    x=self.time[window-1:],
                    y=rolling_mean,
                    name=f'{name} (Rolling Avg)',
                    line=dict(color='red')
                ),
                row=row, col=col
            )
            
            # Confidence intervals
            fig.add_trace(
                go.Scatter(
                    x=self.time[window-1:],
                    y=rolling_mean + 1.96*rolling_std[window-1:],
                    fill=None,
                    line=dict(color='rgba(0,0,0,0)'),
                    showlegend=False
                ),
                row=row, col=col
            )
            
            fig.add_trace(
                go.Scatter(
                    x=self.time[window-1:],
                    y=rolling_mean - 1.96*rolling_std[window-1:],
                    fill='tonexty',
                    line=dict(color='rgba(0,0,0,0)'),
                    showlegend=False
                ),
                row=row, col=col
            )
        
        fig.update_layout(height=800, width=1200, title_text="Spiritual Evolution Over Time")
        return fig
    
    def plot_3d_phase_portrait(self) -> go.Figure:
        """Create 3D phase space visualization with velocity vectors."""
        fig = go.Figure()
        
        # Main trajectory
        fig.add_trace(go.Scatter3d(
            x=self.T, y=self.L, z=self.S,
            mode='lines',
            line=dict(color='blue', width=2),
            name='Spiritual Trajectory'
        ))
        
        # Velocity vectors
        dt = np.diff(self.time)
        dT = np.diff(self.T) / dt
        dL = np.diff(self.L) / dt
        dS = np.diff(self.S) / dt
        
        # Sample every 10th point for clarity
        sample = slice(None, None, 10)
        fig.add_trace(go.Cone(
            x=self.T[sample][:-1],
            y=self.L[sample][:-1],
            z=self.S[sample][:-1],
            u=dT[sample],
            v=dL[sample],
            w=dS[sample],
            sizemode="absolute",
            sizeref=0.1,
            showscale=False,
            name='Velocity Field'
        ))
        
        fig.update_layout(
            scene=dict(
                xaxis_title='Transcendence',
                yaxis_title='Unconditional Love',
                zaxis_title='Synchronicity',
                camera=dict(elevation=elev, azimuth=azim)
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            title="Spiritual Phase Space Projection"
        )
        return fig

    def plot_metrics(self) -> go.Figure:
        """Visualize key analysis metrics"""
        metrics = self.simulator.calculate_metrics()
        
        fig = px.pie(values=list(metrics.values()),
                    names=list(metrics.keys()),
                    hole=0.4,
                    color_discrete_sequence=px.colors.sequential.Rainbow)
        
        fig.update_traces(textposition='inside',
                         textinfo='percent+label',
                         hovertemplate="%{label}: %{value:.2f}")
        
        fig.update_layout(title="Metaphysical Metric Distribution",
                         showlegend=False)
        return fig

    def plot_validation(self) -> plt.Figure:
        """Visualize topological and temporal validation"""
        fig = plt.figure(figsize=(12, 5))
        
        # Topological consistency
        ax1 = fig.add_subplot(121)
        ax1.plot(self.U, self.T*self.L, 'purple', lw=2)
        ax1.set_xlabel('Unity Energy')
        ax1.set_ylabel('T×L Product', rotation=0, ha='right')
        ax1.set_title('Topological Consistency')
        
        # Temporal symmetry
        ax2 = fig.add_subplot(122)
        reversed_U = self.U[::-1]
        ax2.scatter(self.U, reversed_U, c=self.time, cmap='plasma')
        ax2.plot([0,1], [0,1], 'k--', alpha=0.3)
        ax2.set_xlabel('U(t)')
        ax2.set_ylabel('U(-t)', rotation=0, ha='left')
        ax2.set_title('Temporal Symmetry')
        
        plt.tight_layout()
        return fig

    def plot_parameter_sensitivity(self, param_name: str, 
                                 values: List[float]) -> plt.Figure:
        """Plot sensitivity analysis for a given parameter"""
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        axs = axs.flatten()
        
        for value in values:
            # Create new simulator with modified parameter
            params = self.simulator.params.__dict__.copy()
            params[param_name] = value
            new_params = type(self.simulator.params)(**params)
            new_sim = MetaphysicalSimulator(new_params)
            
            # Run simulation with new parameters
            initial_state = MetaphysicalState(
                transcendence=self.T[0],
                love=self.L[0],
                synchronicity=self.S[0],
                unity=self.U[0],
                time=self.time[0]
            )
            new_sim.solve(initial_state)
            
            # Plot results
            for i, (data, label) in enumerate(zip(
                [new_sim.T, new_sim.L, new_sim.S, new_sim.U],
                ['Transcendence', 'Love', 'Synchronicity', 'Unity']
            )):
                axs[i].plot(self.time, data, 
                          label=f'{param_name}={value:.2f}')
                axs[i].set_title(label)
                axs[i].grid(alpha=0.2)
        
        plt.suptitle(f'Parameter Sensitivity: {param_name}')
        plt.legend()
        plt.tight_layout()
        return fig

    def show_all(self):
        """Generate complete dashboard"""
        self.plot_time_evolution()
        self.plot_3d_phase_portrait().show()
        self.plot_metrics().show()
        self.plot_validation()
        plt.show()

# Example usage
if __name__ == "__main__":
    from .simulation import MetaphysicalSimulator, MetaphysicalParameters, MetaphysicalState
    
    # Initialize and run simulation
    params = MetaphysicalParameters(alpha=0.85, lambda_=1.6)
    simulator = MetaphysicalSimulator(params)
    
    initial_state = MetaphysicalState(
        transcendence=0.1,
        love=0.1,
        synchronicity=0.1,
        unity=0.1,
        time=0
    )
    
    simulator.solve(initial_state)
    
    # Visualize results
    visualizer = MetaphysicalVisualizer(simulator)
    visualizer.show_all() 
                yaxis_title='Love',
                zaxis_title='Synchronicity'
            ),
            title='3D Phase Space Portrait'
        )
        return fig
    
    def plot_correlation_matrix(self) -> go.Figure:
        """Create correlation matrix heatmap."""
        data = np.vstack([self.T, self.L, self.S, self.U])
        corr = np.corrcoef(data)
        
        fig = go.Figure(data=go.Heatmap(
            z=corr,
            x=['Transcendence', 'Love', 'Synchronicity', 'Unity'],
            y=['Transcendence', 'Love', 'Synchronicity', 'Unity'],
            colorscale='RdBu',
            zmid=0
        ))
        
        fig.update_layout(
            title='Dimension Correlation Matrix',
            xaxis_title='Dimension',
            yaxis_title='Dimension'
        )
        return fig
    
    def plot_energy_landscape(self) -> go.Figure:
        """Create energy landscape visualization."""
        # Create grid for energy surface
        T_grid = np.linspace(min(self.T), max(self.T), 50)
        L_grid = np.linspace(min(self.L), max(self.L), 50)
        T_mesh, L_mesh = np.meshgrid(T_grid, L_grid)
        
        # Calculate energy surface (example: harmonic potential)
        U_surface = 0.5 * (T_mesh**2 + L_mesh**2)
        
        fig = go.Figure()
        
        # Energy surface
        fig.add_trace(go.Surface(
            x=T_mesh, y=L_mesh, z=U_surface,
            colorscale='Viridis',
            opacity=0.7,
            name='Energy Surface'
        ))
        
        # Trajectory
        fig.add_trace(go.Scatter3d(
            x=self.T, y=self.L, z=self.U,
            mode='lines',
            line=dict(color='red', width=3),
            name='Spiritual Path'
        ))
        
        fig.update_layout(
            scene=dict(
                xaxis_title='Transcendence',
                yaxis_title='Love',
                zaxis_title='Unity Energy'
            ),
            title='Unity Energy Landscape'
        )
        return fig
    
    def calculate_lyapunov(self) -> Tuple[float, np.ndarray]:
        """Calculate Lyapunov exponents for stability analysis."""
        # Numerical Jacobian calculation
        dt = np.diff(self.time)
        state_diff = np.diff(np.vstack([self.T, self.L, self.S, self.U]), axis=1)
        J = state_diff / dt.reshape(-1, 1)
        
        # Calculate eigenvalues
        eigenvalues = np.linalg.eigvals(J)
        max_lyapunov = np.max(np.real(eigenvalues))
        
        return max_lyapunov, eigenvalues
    
    def plot_parameter_sensitivity(self, param_name: str, values: List[float]) -> go.Figure:
        """Plot sensitivity analysis for a parameter."""
        fig = make_subplots(rows=2, cols=2, subplot_titles=(
            'Transcendence', 'Love', 'Synchronicity', 'Unity'
        ))
        
        for value in values:
            # Create new simulator with modified parameter
            params = self.simulator.params.copy()
            setattr(params, param_name, value)
            new_simulator = MetaphysicalSimulator(params)
            new_simulator.solve(self.simulator.history[0])
            
            # Extract data
            time = np.array([state.time for state in new_simulator.history])
            T = np.array([state.transcendence for state in new_simulator.history])
            L = np.array([state.love for state in new_simulator.history])
            S = np.array([state.synchronicity for state in new_simulator.history])
            U = np.array([state.unity for state in new_simulator.history])
            
            # Add traces
            for i, (data, name) in enumerate([
                (T, 'Transcendence'),
                (L, 'Love'),
                (S, 'Synchronicity'),
                (U, 'Unity')
            ]):
                row = i // 2 + 1
                col = i % 2 + 1
                
                fig.add_trace(
                    go.Scatter(
                        x=time, y=data,
                        name=f'{param_name}={value}',
                        line=dict(width=2)
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            height=800, width=1200,
            title_text=f'Parameter Sensitivity: {param_name}'
        )
        return fig 

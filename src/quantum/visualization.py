from typing import List, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import networkx as nx
from qiskit.visualization import plot_bloch_multivector, plot_state_qsphere
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

class QuantumVisualizer:
    """Visualization system for quantum states and operations."""
    
    def __init__(self):
        self.logger = logging.getLogger("QuantumVisualizer")
        
    def plot_quantum_state(self, state: np.ndarray, title: str = "Quantum State") -> None:
        """Plot quantum state using multiple visualization methods."""
        fig = make_subplots(rows=2, cols=2,
                           subplot_titles=("Bloch Sphere", "Q-Sphere", 
                                         "Probability Distribution", "Phase Distribution"))
        
        # Bloch Sphere
        bloch_fig = plot_bloch_multivector(state)
        fig.add_trace(bloch_fig.data[0], row=1, col=1)
        
        # Q-Sphere
        qsphere_fig = plot_state_qsphere(state)
        fig.add_trace(qsphere_fig.data[0], row=1, col=2)
        
        # Probability Distribution
        probabilities = np.abs(state)**2
        fig.add_trace(go.Bar(x=list(range(len(probabilities))), 
                            y=probabilities,
                            name="Probability"),
                     row=2, col=1)
        
        # Phase Distribution
        phases = np.angle(state)
        fig.add_trace(go.Scatter(x=list(range(len(phases))),
                                y=phases,
                                mode='lines+markers',
                                name="Phase"),
                     row=2, col=2)
        
        fig.update_layout(title_text=title, showlegend=False)
        fig.show()
        
    def plot_quantum_network(self, thread_names: List[str], connections: List[tuple]) -> None:
        """Plot quantum thread network."""
        G = nx.Graph()
        G.add_nodes_from(thread_names)
        G.add_edges_from(connections)
        
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue',
                node_size=2000, font_size=16, font_weight='bold')
        plt.title("Quantum Thread Network")
        plt.show()
        
    def plot_error_correction(self, error_info: Dict) -> None:
        """Plot error correction information."""
        fig = go.Figure()
        
        # Error rate over time
        fig.add_trace(go.Scatter(x=list(range(len(error_info["error_history"]))),
                                y=error_info["error_history"],
                                mode='lines+markers',
                                name="Error Rate"))
        
        # Threshold line
        fig.add_trace(go.Scatter(x=[0, len(error_info["error_history"])-1],
                                y=[error_info["threshold"]]*2,
                                mode='lines',
                                name="Threshold",
                                line=dict(dash='dash')))
        
        fig.update_layout(title="Error Correction Performance",
                         xaxis_title="Time",
                         yaxis_title="Error Rate")
        fig.show()
        
    def plot_quantum_algorithm(self, algorithm_name: str, results: Dict) -> None:
        """Plot quantum algorithm results."""
        if algorithm_name == "search":
            self._plot_search_results(results)
        elif algorithm_name == "optimization":
            self._plot_optimization_results(results)
        elif algorithm_name == "ml":
            self._plot_ml_results(results)
            
    def _plot_search_results(self, results: Dict) -> None:
        """Plot quantum search results."""
        fig = go.Figure()
        
        # Success probability
        fig.add_trace(go.Scatter(x=list(range(len(results["success_prob"]))),
                                y=results["success_prob"],
                                mode='lines+markers',
                                name="Success Probability"))
        
        # Iterations
        fig.add_trace(go.Scatter(x=list(range(len(results["iterations"]))),
                                y=results["iterations"],
                                mode='lines',
                                name="Iterations",
                                yaxis="y2"))
        
        fig.update_layout(title="Quantum Search Performance",
                         xaxis_title="Time",
                         yaxis_title="Success Probability",
                         yaxis2=dict(title="Iterations",
                                   overlaying="y",
                                   side="right"))
        fig.show()
        
    def _plot_optimization_results(self, results: Dict) -> None:
        """Plot quantum optimization results."""
        fig = go.Figure()
        
        # Cost function
        fig.add_trace(go.Scatter(x=list(range(len(results["cost"]))),
                                y=results["cost"],
                                mode='lines+markers',
                                name="Cost"))
        
        # Energy landscape
        fig.add_trace(go.Contour(z=results["energy_landscape"],
                                name="Energy Landscape"))
        
        fig.update_layout(title="Quantum Optimization Results",
                         xaxis_title="Iteration",
                         yaxis_title="Cost")
        fig.show()
        
    def _plot_ml_results(self, results: Dict) -> None:
        """Plot quantum machine learning results."""
        fig = make_subplots(rows=1, cols=2,
                           subplot_titles=("Training Progress", "Feature Map"))
        
        # Training progress
        fig.add_trace(go.Scatter(x=list(range(len(results["loss"]))),
                                y=results["loss"],
                                mode='lines',
                                name="Loss"),
                     row=1, col=1)
        
        # Feature map
        fig.add_trace(go.Heatmap(z=results["feature_map"],
                                name="Feature Map"),
                     row=1, col=2)
        
        fig.update_layout(title="Quantum Machine Learning Results")
        fig.show()
        
    def create_animation(self, states: List[np.ndarray], interval: int = 200) -> None:
        """Create animation of quantum state evolution."""
        fig = plt.figure(figsize=(10, 8))
        
        def update(frame):
            plt.clf()
            state = states[frame]
            plot_bloch_multivector(state)
            plt.title(f"Quantum State Evolution (Frame {frame})")
            
        anim = FuncAnimation(fig, update, frames=len(states),
                            interval=interval, blit=False)
        plt.show()
        
    def plot_entanglement(self, thread_states: Dict[str, np.ndarray]) -> None:
        """Plot entanglement between quantum threads."""
        fig = go.Figure()
        
        # Create correlation matrix
        num_threads = len(thread_states)
        correlation = np.zeros((num_threads, num_threads))
        
        for i, (name1, state1) in enumerate(thread_states.items()):
            for j, (name2, state2) in enumerate(thread_states.items()):
                if i != j:
                    correlation[i,j] = np.abs(np.vdot(state1, state2))
                    
        # Plot correlation matrix
        fig.add_trace(go.Heatmap(z=correlation,
                                x=list(thread_states.keys()),
                                y=list(thread_states.keys()),
                                colorscale="Viridis"))
        
        fig.update_layout(title="Quantum Thread Entanglement",
                         xaxis_title="Thread",
                         yaxis_title="Thread")
        fig.show() 
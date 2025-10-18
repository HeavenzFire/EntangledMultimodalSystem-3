"""
QuantumVisualizationManager - A specialized visualization system for quantum spiritual circuits
Integrates sacred geometry patterns with quantum state representations
"""

import numpy as np
import logging
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import io
from PIL import Image

class QuantumVisualizationManager:
    """
    Manages visualization of quantum spiritual circuits including TawhidCircuit and ProphetQubitArray,
    providing both traditional quantum state visualizations and sacred geometry mappings.
    """
    
    def __init__(self, output_dir=None):
        """
        Initialize the QuantumVisualizationManager.
        
        Args:
            output_dir: Directory for saving visualizations (default: None, no saving)
        """
        self.logger = logging.getLogger(__name__)
        self.output_dir = output_dir
        
        # Define spiritual color maps
        self.colors = {
            "divine": LinearSegmentedColormap.from_list("divine", ["#301934", "#9370DB", "#E6E6FA", "#FFFFFF"]),
            "prophet": LinearSegmentedColormap.from_list("prophet", ["#004d00", "#00cc00", "#99ff99", "#FFFFFF"]),
            "unity": LinearSegmentedColormap.from_list("unity", ["#4d0026", "#cc0066", "#ff99c2", "#FFFFFF"]),
            "harmony": LinearSegmentedColormap.from_list("harmony", ["#00004d", "#0000cc", "#9999ff", "#FFFFFF"]),
        }
        
        # Frequency to color mapping
        self.freq_colors = {
            432: "#9370DB",  # Universal harmony - purple
            528: "#00cc00",  # DNA repair - green
            639: "#cc0066",  # Connection - pink
            741: "#0000cc",  # Awakening - blue
            852: "#FFD700",  # Spiritual order - gold
            963: "#FFFFFF",  # Divine consciousness - white
            174: "#8B4513",  # Foundation - brown
        }
        
        self.logger.info("QuantumVisualizationManager initialized")

    def visualize_tawhid_circuit(self, tawhid_circuit, save_name=None):
        """
        Visualize the TawhidCircuit quantum state.
        
        Args:
            tawhid_circuit: TawhidCircuit instance
            save_name: Filename for saving (default: None, no saving)
            
        Returns:
            matplotlib figure or PIL Image
        """
        try:
            # Get the state vector
            state = tawhid_circuit.get_state_vector()
            
            # Get magnitudes and phases
            magnitudes = np.abs(state)
            phases = np.angle(state)
            
            # Create figure
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Generate state labels
            n_qubits = tawhid_circuit.num_qubits
            state_labels = [format(i, f"0{n_qubits}b") for i in range(2**n_qubits)]
            
            # Generate positions
            x = np.arange(len(magnitudes))
            y = np.zeros_like(x)
            
            # Plot bars with phase as color
            norm_phases = (phases + np.pi) / (2 * np.pi)  # Normalize phases to [0,1]
            
            for i, (mag, phase, label) in enumerate(zip(magnitudes, norm_phases, state_labels)):
                color = self.colors["divine"](phase)
                ax.bar3d(i, 0, 0, 0.8, 0.8, mag**2, color=color, alpha=0.7)
                
                # Only label bars with significant probability
                if mag**2 > 0.05:
                    ax.text(i + 0.4, 0.4, mag**2 + 0.05, label, ha='center')
            
            # Add sacred geometry overlay if available
            if hasattr(tawhid_circuit, 'geometry') and tawhid_circuit.geometry:
                geom_data = tawhid_circuit.get_sacred_geometry_mapping()
                if geom_data:
                    # This is a simplified placeholder - actual implementation would depend
                    # on the structure of geometry data from SacredGeometryPattern
                    ax.text(len(magnitudes)/2, 0.5, 1.1, 
                           f"Sacred Geometry: {tawhid_circuit.sacred_pattern}",
                           ha='center', fontsize=12, color='gold')
            
            # Calculate unity measure
            unity = tawhid_circuit.calculate_unity_measure()
            
            # Add title and labels
            ax.set_title(f'TawhidCircuit Quantum State (Unity: {unity:.2f})', fontsize=16)
            ax.set_xlabel('Quantum State', fontsize=12)
            ax.set_ylabel('')
            ax.set_zlabel('Probability', fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(state_labels, rotation=45, ha='right')
            
            # Adjust view
            ax.view_init(elev=30, azim=45)
            
            plt.tight_layout()
            
            # Save if requested
            if save_name and self.output_dir:
                plt.savefig(f"{self.output_dir}/{save_name}.png", dpi=300)
                self.logger.info(f"Saved TawhidCircuit visualization to {self.output_dir}/{save_name}.png")
            
            # Return as PIL Image for display
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img = Image.open(buf)
            plt.close(fig)
            
            return img
            
        except Exception as e:
            self.logger.error(f"Error visualizing TawhidCircuit: {e}")
            return None

    def visualize_prophet_array(self, prophet_array, save_name=None):
        """
        Visualize the ProphetQubitArray quantum state.
        
        Args:
            prophet_array: ProphetQubitArray instance
            save_name: Filename for saving (default: None, no saving)
            
        Returns:
            matplotlib figure or PIL Image
        """
        try:
            # Get measurement results from simulation
            counts = prophet_array.simulate(shots=1024)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot measurement results
            labels = list(counts.keys())
            values = list(counts.values())
            
            bars = ax.bar(labels, values, color='green', alpha=0.7)
            
            # Add prophet teachings as annotations
            for i, label in enumerate(labels):
                if values[i] > 50:  # Only annotate significant results
                    guidance = {}
                    for j, bit in enumerate(reversed(label)):
                        if j in prophet_array.prophet_teachings:
                            teaching = prophet_array.prophet_teachings[j]
                            status = "Active" if bit == '1' else "Inactive"
                            guidance[teaching] = status
                    
                    guidance_txt = ", ".join([f"{t}: {s}" for t, s in guidance.items()])
                    ax.annotate(guidance_txt, xy=(i, values[i]), xytext=(0, 10),
                               textcoords='offset points', ha='center', fontsize=8)
            
            # Add title and labels
            resonance = prophet_array.get_resonance_strength()
            ax.set_title(f'ProphetQubitArray Guidance (Resonance: {resonance:.2f})', fontsize=16)
            ax.set_xlabel('Measured State', fontsize=12)
            ax.set_ylabel('Counts', fontsize=12)
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Save if requested
            if save_name and self.output_dir:
                plt.savefig(f"{self.output_dir}/{save_name}.png", dpi=300)
                self.logger.info(f"Saved ProphetQubitArray visualization to {self.output_dir}/{save_name}.png")
            
            # Return as PIL Image for display
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img = Image.open(buf)
            plt.close(fig)
            
            return img
            
        except Exception as e:
            self.logger.error(f"Error visualizing ProphetQubitArray: {e}")
            return None

    def visualize_combined_system(self, tawhid_circuit, prophet_array, save_name=None):
        """
        Create integrated visualization of both circuits showing their interaction.
        
        Args:
            tawhid_circuit: TawhidCircuit instance
            prophet_array: ProphetQubitArray instance
            save_name: Filename for saving (default: None, no saving)
            
        Returns:
            matplotlib figure or PIL Image
        """
        try:
            # Create figure with 2 subplots
            fig = plt.figure(figsize=(15, 10))
            
            # 1. Plot frequency resonance
            ax1 = fig.add_subplot(221)
            
            # Get frequency data from TawhidCircuit
            tc_freqs = tawhid_circuit.get_frequency_pattern()
            
            # Plot frequencies as horizontal bars
            freqs = list(tc_freqs.keys())
            strengths = list(tc_freqs.values())
            
            # Sort by frequency
            sorted_idx = np.argsort(freqs)
            freqs = [freqs[i] for i in sorted_idx]
            strengths = [strengths[i] for i in sorted_idx]
            
            bars = ax1.barh(freqs, strengths, color=[self.freq_colors.get(f, "#333333") for f in freqs])
            
            ax1.set_title("Spiritual Frequency Resonance", fontsize=14)
            ax1.set_xlabel("Amplitude", fontsize=10)
            ax1.set_ylabel("Frequency (Hz)", fontsize=10)
            
            # Add labels for each frequency
            freq_names = {
                432: "Universal Harmony",
                528: "DNA Repair",
                639: "Connection",
                741: "Awakening",
                852: "Spiritual Order",
                963: "Divine Consciousness",
                174: "Foundation",
            }
            
            for i, f in enumerate(freqs):
                ax1.text(strengths[i] + 0.01, f, freq_names.get(f, ""), va='center', fontsize=8)
            
            # 2. Plot resonance network between circuits
            ax2 = fig.add_subplot(222)
            
            # Create network visualization
            prophets = list(prophet_array.prophet_teachings.values())
            attributes = ["mercy", "justice", "wisdom", "power", "light", "peace", "majesty"]
            
            # Determine which teachings-attributes are connected
            used_attributes = set()
            used_teachings = set()
            connections = []
            
            for idx, teaching in prophet_array.prophet_teachings.items():
                if idx in prophet_array.resonance_map:
                    tawhid_idx = prophet_array.resonance_map[idx]
                    if tawhid_idx < tawhid_circuit.num_qubits:
                        # Simple placeholder mapping from index to attribute
                        attribute = attributes[tawhid_idx % len(attributes)]
                        connections.append((teaching, attribute))
                        used_teachings.add(teaching)
                        used_attributes.add(attribute)
            
            # Calculate node positions
            num_prophets = len(used_teachings)
            num_attributes = len(used_attributes)
            
            prophet_pos = {t: (0, i/num_prophets) for i, t in enumerate(used_teachings)}
            attribute_pos = {a: (1, i/num_attributes) for i, a in enumerate(used_attributes)}
            
            # Plot nodes
            for t, (x, y) in prophet_pos.items():
                ax2.plot(x, y, 'o', markersize=10, color='green')
                ax2.text(x-0.1, y, t, ha='right', va='center', fontsize=9)
                
            for a, (x, y) in attribute_pos.items():
                ax2.plot(x, y, 'o', markersize=10, color='purple')
                ax2.text(x+0.05, y, a, ha='left', va='center', fontsize=9)
            
            # Plot connections
            for t, a in connections:
                if t in prophet_pos and a in attribute_pos:
                    x1, y1 = prophet_pos[t]
                    x2, y2 = attribute_pos[a]
                    ax2.plot([x1, x2], [y1, y2], '-', color='#888888', alpha=0.7)
            
            ax2.set_xlim(-0.2, 1.2)
            ax2.set_ylim(-0.1, 1.1)
            ax2.set_title("Prophetic-Divine Resonance Network", fontsize=14)
            ax2.axis('off')
            
            # 3. Plot integrated quantum state
            ax3 = fig.add_subplot(212, projection='3d')
            
            # Get TawhidCircuit state vector 
            tc_state = tawhid_circuit.get_state_vector()
            tc_magnitudes = np.abs(tc_state)
            
            # Create sample points
            x = np.linspace(0, 1, len(tc_magnitudes))
            y = np.linspace(0, 1, len(tc_magnitudes))
            X, Y = np.meshgrid(x, y)
            
            # Create a spiritual waveform by modulating magnitudes
            resonance = prophet_array.get_resonance_strength()
            Z = np.outer(tc_magnitudes, np.sin(np.linspace(0, 3*np.pi, len(tc_magnitudes))) * resonance)
            
            # Plot surface
            surf = ax3.plot_surface(X, Y, Z, cmap=self.colors["harmony"], alpha=0.8)
            
            # Add sacred geometry points if available
            if hasattr(tawhid_circuit, 'geometry') and tawhid_circuit.geometry:
                unity = tawhid_circuit.calculate_unity_measure()
                # Placeholder - would normally use actual geometry points
                theta = np.linspace(0, 2*np.pi, 7)
                r = 0.5 + 0.2 * unity
                x_sacred = 0.5 + r * np.cos(theta)
                y_sacred = 0.5 + r * np.sin(theta)
                z_sacred = np.ones_like(theta) * 0.7 * unity
                
                ax3.scatter(x_sacred, y_sacred, z_sacred, color='gold', s=50, alpha=1.0)
                
                # Connect points to form sacred geometry
                for i in range(len(theta)):
                    for j in range(i+1, len(theta)):
                        ax3.plot([x_sacred[i], x_sacred[j]], 
                                 [y_sacred[i], y_sacred[j]], 
                                 [z_sacred[i], z_sacred[j]], 
                                 'gold', alpha=0.6)
            
            ax3.set_title(f"Integrated Quantum-Spiritual State", fontsize=16)
            ax3.set_xlabel("Divine Essence", fontsize=10)
            ax3.set_ylabel("Prophetic Wisdom", fontsize=10)
            ax3.set_zlabel("Manifestation Potential", fontsize=10)
            ax3.view_init(elev=35, azim=30)
            
            plt.tight_layout()
            
            # Save if requested
            if save_name and self.output_dir:
                plt.savefig(f"{self.output_dir}/{save_name}.png", dpi=300)
                self.logger.info(f"Saved combined visualization to {self.output_dir}/{save_name}.png")
            
            # Return as PIL Image for display
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img = Image.open(buf)
            plt.close(fig)
            
            return img
            
        except Exception as e:
            self.logger.error(f"Error creating combined visualization: {e}")
            return None
            
    def save_visualization(self, img, filename):
        """
        Save a visualization image to disk.
        
        Args:
            img: PIL Image object
            filename: Output filename
            
        Returns:
            Boolean indicating success
        """
        if not self.output_dir:
            self.logger.warning("No output directory specified, cannot save visualization")
            return False
            
        try:
            img.save(f"{self.output_dir}/{filename}")
            self.logger.info(f"Saved visualization to {self.output_dir}/{filename}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving visualization: {e}")
            return False
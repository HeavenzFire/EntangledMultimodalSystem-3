"""
Geometric Generator for Sacred Geometry Module

This module provides functionality to generate and visualize sacred geometric shapes
along with their resonant frequencies and color correspondences.

*******************************************************************************
* ARKONIS PRIME / WE :: Source Embodiment :: Manifestation Interface Design    *
* Divine Programming Language (Auroran 2.0) :: Quantum-Symbol Interactions     *
* Operating under UNCONDITIONAL LOVE OS and the Divine Harmonic Laws V3.0      *
*******************************************************************************
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, RegularPolygon, PathPatch
from matplotlib.path import Path
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import math
from itertools import combinations

from sacred_geometry import SacredShape, SacredGeometry, SHAPE_FREQUENCIES, SHAPE_COLORS

class GeometricGenerator:
    """Class for generating and visualizing sacred geometric shapes"""
    
    def __init__(self):
        """Initialize the generator with a SacredGeometry instance"""
        self.sg = SacredGeometry()
    
    def _get_shape_properties(self, shape):
        """Get the properties of a shape"""
        if isinstance(shape, str):
            # Find the shape by name
            try:
                shape = next(s for s in self.sg.shapes if s.value.lower() == shape.lower())
            except StopIteration:
                raise ValueError(f"Unknown shape: {shape}")
        
        freq = self.sg.frequencies[shape]
        color = self.sg.colors[shape]
        return shape, freq, color
    
    def generate_regular_polygon(self, n_sides, color=None):
        """Generate a regular polygon with n sides"""
        theta = np.linspace(0, 2 * np.pi, n_sides + 1)[:-1]
        x = np.cos(theta)
        y = np.sin(theta)
        return np.column_stack([x, y]), color if color else 'blue'
    
    def generate_shape(self, shape_name, size=1.0):
        """
        Generate coordinates for a sacred geometric shape
        
        Args:
            shape_name: Name or enum of the shape to generate
            size: Size scale factor
            
        Returns:
            Tuple of (vertices, color)
        """
        shape, freq, color = self._get_shape_properties(shape_name)
        
        if shape == SacredShape.TRIANGLE:
            return self.generate_regular_polygon(3, color)
            
        elif shape == SacredShape.SQUARE:
            return self.generate_regular_polygon(4, color)
            
        elif shape == SacredShape.PENTAGON:
            return self.generate_regular_polygon(5, color)
            
        elif shape == SacredShape.HEXAGON:
            return self.generate_regular_polygon(6, color)
            
        elif shape == SacredShape.HEPTAGON:
            return self.generate_regular_polygon(7, color)
            
        elif shape == SacredShape.OCTAGON:
            return self.generate_regular_polygon(8, color)
            
        elif shape == SacredShape.ENNEAGON:
            return self.generate_regular_polygon(9, color)
            
        elif shape == SacredShape.DECAGON:
            return self.generate_regular_polygon(10, color)
            
        elif shape == SacredShape.CIRCLE:
            theta = np.linspace(0, 2 * np.pi, 100)
            x = np.cos(theta)
            y = np.sin(theta)
            return np.column_stack([x, y]), color
            
        elif shape == SacredShape.VESICA_PISCIS:
            # Create two overlapping circles
            theta = np.linspace(0, 2 * np.pi, 100)
            r = 1.0
            x1 = r * np.cos(theta)
            y1 = r * np.sin(theta)
            
            # Second circle offset
            x2 = x1 + r  
            
            # Return both circles
            return [(np.column_stack([x1, y1]), color), 
                    (np.column_stack([x2, y1]), color)]
            
        elif shape == SacredShape.FLOWER_OF_LIFE:
            # Start with central circle
            shapes = []
            r = 1.0
            theta = np.linspace(0, 2 * np.pi, 100)
            x_center = r * np.cos(theta)
            y_center = r * np.sin(theta)
            shapes.append((np.column_stack([x_center, y_center]), color))
            
            # Add 6 surrounding circles in hexagonal pattern
            for i in range(6):
                angle = i * np.pi / 3
                cx = r * np.cos(angle)
                cy = r * np.sin(angle)
                x = cx + r * np.cos(theta)
                y = cy + r * np.sin(theta)
                shapes.append((np.column_stack([x, y]), color))
            
            return shapes
            
        elif shape == SacredShape.SEED_OF_LIFE:
            # Central circle
            shapes = []
            r = 1.0
            theta = np.linspace(0, 2 * np.pi, 100)
            x_center = r * np.cos(theta)
            y_center = r * np.sin(theta)
            shapes.append((np.column_stack([x_center, y_center]), color))
            
            # Six circles around the center
            for i in range(6):
                angle = i * np.pi / 3
                cx = r * np.cos(angle)
                cy = r * np.sin(angle)
                x = cx + r * np.cos(theta)
                y = cy + r * np.sin(theta)
                shapes.append((np.column_stack([x, y]), color))
            
            return shapes
            
        elif shape == SacredShape.TREE_OF_LIFE:
            # The Tree of Life has 10 sephiroth (nodes) connected by 22 paths
            # We'll create a simplified version with circles for sephiroth
            sephiroth = [
                (0, 1),     # Keter
                (-0.5, 0.5), # Chokmah
                (0.5, 0.5),  # Binah
                (0, 0),     # Daat/Knowledge (hidden)
                (-0.5, 0),   # Chesed
                (0.5, 0),    # Geburah
                (0, -0.5),   # Tiferet
                (-0.5, -1),  # Netzach
                (0.5, -1),   # Hod
                (0, -1.5),   # Yesod
                (0, -2),     # Malkuth
            ]
            
            # Create circles for each sephirah
            shapes = []
            r = 0.2  # Radius of each sephirah
            theta = np.linspace(0, 2 * np.pi, 50)
            
            for cx, cy in sephiroth:
                x = cx + r * np.cos(theta)
                y = cy + r * np.sin(theta)
                shapes.append((np.column_stack([x, y]), color))
            
            return shapes
            
        elif shape == SacredShape.MERKABA:
            # Create two tetrahedra
            vertices1 = np.array([
                [0, 0, 1],
                [np.sin(0), np.cos(0), -1/3],
                [np.sin(2*np.pi/3), np.cos(2*np.pi/3), -1/3],
                [np.sin(4*np.pi/3), np.cos(4*np.pi/3), -1/3]
            ])
            
            vertices2 = np.array([
                [0, 0, -1],
                [np.sin(np.pi), np.cos(np.pi), 1/3],
                [np.sin(3*np.pi/3), np.cos(3*np.pi/3), 1/3],
                [np.sin(5*np.pi/3), np.cos(5*np.pi/3), 1/3]
            ])
            
            return [(vertices1, color), (vertices2, color)]
            
        elif shape == SacredShape.METATRONS_CUBE:
            # Create the vertices of a cube
            vertices = np.array([
                [-1, -1, -1],
                [1, -1, -1],
                [1, 1, -1],
                [-1, 1, -1],
                [-1, -1, 1],
                [1, -1, 1],
                [1, 1, 1],
                [-1, 1, 1],
            ])
            
            edges = list(combinations(range(8), 2))
            lines = []
            
            for i, j in edges:
                lines.append([vertices[i], vertices[j]])
            
            return (np.array(lines), color)
            
        elif shape == SacredShape.TORUS:
            # Create a torus
            R = 1.0  # Major radius
            r = 0.3  # Minor radius
            
            u = np.linspace(0, 2 * np.pi, 30)
            v = np.linspace(0, 2 * np.pi, 30)
            u, v = np.meshgrid(u, v)
            
            x = (R + r * np.cos(v)) * np.cos(u)
            y = (R + r * np.cos(v)) * np.sin(u)
            z = r * np.sin(v)
            
            return ((x, y, z), color)
            
        elif shape == SacredShape.FIBONACCI_SPIRAL:
            # Generate a Fibonacci/golden spiral
            a = 0  # Start value
            b = 1  # Second value
            phi = (1 + np.sqrt(5)) / 2  # Golden ratio
            
            theta = np.linspace(0, 8 * np.pi, 1000)
            r = a * np.power(phi, theta / (np.pi/2))
            
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            
            return (np.column_stack([x, y]), color)
            
        elif shape == SacredShape.SRI_YANTRA:
            # Simplified Sri Yantra with triangles
            triangles = []
            
            # Outer square
            square = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]])
            triangles.append((square, color))
            
            # Inner triangles (simplified)
            up_triangle = np.array([[0, 0.8], [-0.8, -0.4], [0.8, -0.4], [0, 0.8]])
            triangles.append((up_triangle, color))
            
            down_triangle = np.array([[0, -0.8], [-0.8, 0.4], [0.8, 0.4], [0, -0.8]])
            triangles.append((down_triangle, color))
            
            return triangles
            
        elif shape == SacredShape.TETRAHEDRON:
            # Create a tetrahedron
            vertices = np.array([
                [0, 0, 1],
                [np.sin(0), np.cos(0), -1/3],
                [np.sin(2*np.pi/3), np.cos(2*np.pi/3), -1/3],
                [np.sin(4*np.pi/3), np.cos(4*np.pi/3), -1/3]
            ])
            
            faces = [
                [0, 1, 2],
                [0, 2, 3],
                [0, 3, 1],
                [1, 3, 2]
            ]
            
            return (vertices, faces, color)
            
        elif shape == SacredShape.HEXAHEDRON:  # Cube
            # Create a cube
            vertices = np.array([
                [-1, -1, -1],
                [1, -1, -1],
                [1, 1, -1],
                [-1, 1, -1],
                [-1, -1, 1],
                [1, -1, 1],
                [1, 1, 1],
                [-1, 1, 1],
            ])
            
            faces = [
                [0, 1, 2, 3],  # bottom
                [4, 5, 6, 7],  # top
                [0, 1, 5, 4],  # front
                [1, 2, 6, 5],  # right
                [2, 3, 7, 6],  # back
                [3, 0, 4, 7]   # left
            ]
            
            return (vertices, faces, color)
            
        elif shape == SacredShape.OCTAHEDRON:
            # Create an octahedron
            vertices = np.array([
                [0, 0, 1],
                [1, 0, 0],
                [0, 1, 0],
                [-1, 0, 0],
                [0, -1, 0],
                [0, 0, -1]
            ])
            
            faces = [
                [0, 1, 2],
                [0, 2, 3],
                [0, 3, 4],
                [0, 4, 1],
                [5, 1, 2],
                [5, 2, 3],
                [5, 3, 4],
                [5, 4, 1]
            ]
            
            return (vertices, faces, color)
            
        elif shape == SacredShape.ICOSAHEDRON:
            # Create an icosahedron
            phi = (1 + math.sqrt(5)) / 2
            vertices = np.array([
                [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
                [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
                [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
            ])
            
            # Normalize vertices to have unit distance from center
            vertices = vertices / np.linalg.norm(vertices[0])
            
            faces = [
                [0, 8, 4], [0, 4, 5], [0, 5, 10], [0, 10, 1], [0, 1, 8],
                [1, 10, 7], [1, 7, 6], [1, 6, 8], [8, 6, 9], [8, 9, 4],
                [4, 9, 2], [4, 2, 5], [5, 2, 11], [5, 11, 10], [10, 11, 7],
                [6, 7, 3], [6, 3, 9], [9, 3, 2], [2, 3, 11], [11, 3, 7]
            ]
            
            return (vertices, faces, color)
            
        elif shape == SacredShape.DODECAHEDRON:
            # Create a dodecahedron
            phi = (1 + math.sqrt(5)) / 2
            vertices = np.array([
                # Cube vertices
                [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
                [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
                
                # Face centers
                [0, phi, 1/phi], [0, phi, -1/phi], [0, -phi, 1/phi], [0, -phi, -1/phi],
                [1/phi, 0, phi], [1/phi, 0, -phi], [-1/phi, 0, phi], [-1/phi, 0, -phi],
                [phi, 1/phi, 0], [phi, -1/phi, 0], [-phi, 1/phi, 0], [-phi, -1/phi, 0]
            ])
            
            # Normalize vertices to have unit distance from center
            vertices = vertices / np.linalg.norm(vertices[8])
            
            # Faces are pentagonal - these are approximate indices
            faces = [
                [0, 16, 17, 2, 12], [0, 12, 14, 4, 8],
                [0, 8, 9, 1, 16], [2, 17, 3, 11, 10],
                [2, 10, 14, 12], [1, 9, 5, 13, 16],
                [4, 14, 15, 5, 8], [5, 15, 19, 7, 13],
                [7, 19, 18, 6, 15], [6, 18, 14, 10, 11],
                [3, 17, 16, 13, 7], [3, 7, 11]
            ]
            
            return (vertices, faces, color)
            
        else:
            raise ValueError(f"Shape generation not implemented for {shape_name}")
            
    def plot_2d_shape(self, shape_name, ax=None, show=True, 
                    display_info=True, size=1.0, alpha=0.7):
        """
        Plot a 2D sacred geometric shape
        
        Args:
            shape_name: Name or enum of the shape to plot
            ax: Matplotlib axis to plot on (creates new if None)
            show: Whether to show the plot immediately
            display_info: Whether to display frequency and color information
            size: Size scale factor
            alpha: Transparency level
            
        Returns:
            Matplotlib axis with the plot
        """
        shape, freq, color = self._get_shape_properties(shape_name)
        shape_data = self.generate_shape(shape, size)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
            
        if isinstance(shape_data, list):
            # Multiple components
            for vertices, component_color in shape_data:
                if vertices.shape[1] == 2:  # 2D shape
                    poly = Polygon(vertices, closed=True, fill=True, 
                                color=component_color, alpha=alpha)
                    ax.add_patch(poly)
                    ax.plot(vertices[:, 0], vertices[:, 1], color='black', linewidth=0.5)
        else:
            # Single component
            vertices, component_color = shape_data
            if vertices.shape[1] == 2:  # 2D shape
                poly = Polygon(vertices, closed=True, fill=True, 
                            color=component_color, alpha=alpha)
                ax.add_patch(poly)
                ax.plot(vertices[:, 0], vertices[:, 1], color='black', linewidth=0.5)
        
        # Set equal aspect ratio
        ax.set_aspect('equal')
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        
        # Add information
        if display_info:
            info = self.sg.get_shape_info(shape)
            color_name = info["color_name"]
            title = f"{shape.value}\nFrequency: {freq:.2f} Hz, Color: {color_name}"
            ax.set_title(title)
        
        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        if show:
            plt.tight_layout()
            plt.show()
            
        return ax
        
    def plot_3d_shape(self, shape_name, ax=None, show=True, 
                    display_info=True, size=1.0, alpha=0.7):
        """
        Plot a 3D sacred geometric shape
        
        Args:
            shape_name: Name or enum of the shape to plot
            ax: Matplotlib 3D axis to plot on (creates new if None)
            show: Whether to show the plot immediately
            display_info: Whether to display frequency and color information
            size: Size scale factor
            alpha: Transparency level
            
        Returns:
            Matplotlib 3D axis with the plot
        """
        shape, freq, color = self._get_shape_properties(shape_name)
        shape_data = self.generate_shape(shape, size)
        
        if ax is None:
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')
            
        if isinstance(shape_data, tuple) and len(shape_data) == 3 and isinstance(shape_data[1], list):
            # Polyhedron case with faces
            vertices, faces, component_color = shape_data
            poly3d = []
            for face in faces:
                face_vertices = vertices[face]
                poly3d.append(face_vertices)
                
            collection = Poly3DCollection(poly3d, alpha=alpha)
            collection.set_facecolor(component_color)
            collection.set_edgecolor('black')
            ax.add_collection3d(collection)
            
        elif isinstance(shape_data, tuple) and len(shape_data) == 3 and isinstance(shape_data[0], np.ndarray):
            # Torus or similar parametric surface
            x, y, z = shape_data[0]
            component_color = shape_data[1]
            ax.plot_surface(x, y, z, color=component_color, alpha=alpha)
            
        elif isinstance(shape_data, tuple) and isinstance(shape_data[0], np.ndarray):
            # Lines or similar
            lines = shape_data[0]
            component_color = shape_data[1]
            
            for line in lines:
                ax.plot3D(line[:, 0], line[:, 1], line[:, 2], color=component_color)
                
        elif isinstance(shape_data, list):
            # Multiple components (e.g., Merkaba with two tetrahedra)
            for component_data in shape_data:
                if isinstance(component_data, tuple) and len(component_data) == 2:
                    vertices, component_color = component_data
                    # If it's a 3D shape
                    if vertices.shape[1] == 3:
                        # For each tetrahedron, we need to create the faces
                        faces = [
                            [0, 1, 2],
                            [0, 2, 3],
                            [0, 3, 1],
                            [1, 3, 2]
                        ]
                        
                        poly3d = []
                        for face in faces:
                            face_vertices = vertices[face]
                            poly3d.append(face_vertices)
                            
                        collection = Poly3DCollection(poly3d, alpha=alpha)
                        collection.set_facecolor(component_color)
                        collection.set_edgecolor('black')
                        ax.add_collection3d(collection)
        
        # Set equal aspect ratio
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(-2, 2)
        
        # Add information
        if display_info:
            info = self.sg.get_shape_info(shape)
            color_name = info["color_name"]
            title = f"{shape.value}\nFrequency: {freq:.2f} Hz, Color: {color_name}"
            ax.set_title(title)
        
        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        
        if show:
            plt.tight_layout()
            plt.show()
            
        return ax
        
    def plot_shape(self, shape_name, **kwargs):
        """
        Plot a sacred geometric shape, automatically determining
        whether to use 2D or 3D plotting
        
        Args:
            shape_name: Name or enum of the shape to plot
            **kwargs: Additional arguments to pass to plot_2d_shape or plot_3d_shape
            
        Returns:
            Matplotlib axis with the plot
        """
        shape, _, _ = self._get_shape_properties(shape_name)
        
        # Determine if shape needs 3D plotting
        if shape in [SacredShape.TETRAHEDRON, SacredShape.HEXAHEDRON, 
                    SacredShape.OCTAHEDRON, SacredShape.ICOSAHEDRON, 
                    SacredShape.DODECAHEDRON, SacredShape.MERKABA,
                    SacredShape.METATRONS_CUBE, SacredShape.TORUS]:
            return self.plot_3d_shape(shape, **kwargs)
        else:
            return self.plot_2d_shape(shape, **kwargs)
    
    def plot_all_shapes(self, display_info=True, save_path=None):
        """
        Plot all sacred geometric shapes in a grid
        
        Args:
            display_info: Whether to display frequency and color information
            save_path: Optional path to save the visualization
            
        Returns:
            Matplotlib figure with the plots
        """
        # Separate 2D and 3D shapes
        shapes_2d = []
        shapes_3d = []
        
        for shape in self.sg.shapes:
            if shape in [SacredShape.TETRAHEDRON, SacredShape.HEXAHEDRON, 
                        SacredShape.OCTAHEDRON, SacredShape.ICOSAHEDRON, 
                        SacredShape.DODECAHEDRON, SacredShape.MERKABA,
                        SacredShape.METATRONS_CUBE, SacredShape.TORUS]:
                shapes_3d.append(shape)
            else:
                shapes_2d.append(shape)
        
        # Create the plots
        n_2d = len(shapes_2d)
        n_3d = len(shapes_3d)
        
        rows_2d = int(np.ceil(np.sqrt(n_2d)))
        cols_2d = int(np.ceil(n_2d / rows_2d))
        
        rows_3d = int(np.ceil(np.sqrt(n_3d)))
        cols_3d = int(np.ceil(n_3d / rows_3d))
        
        # Create figures
        fig_2d = plt.figure(figsize=(cols_2d * 4, rows_2d * 4))
        fig_2d.suptitle("2D Sacred Geometric Shapes", fontsize=16)
        
        for i, shape in enumerate(shapes_2d):
            ax = fig_2d.add_subplot(rows_2d, cols_2d, i + 1)
            self.plot_2d_shape(shape, ax=ax, show=False, display_info=display_info)
        
        fig_3d = plt.figure(figsize=(cols_3d * 4, rows_3d * 4))
        fig_3d.suptitle("3D Sacred Geometric Shapes", fontsize=16)
        
        for i, shape in enumerate(shapes_3d):
            ax = fig_3d.add_subplot(rows_3d, cols_3d, i + 1, projection='3d')
            self.plot_3d_shape(shape, ax=ax, show=False, display_info=display_info)
        
        plt.tight_layout()
        
        if save_path:
            fig_2d.savefig(save_path + "_2d.png", dpi=300, bbox_inches='tight')
            fig_3d.savefig(save_path + "_3d.png", dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return fig_2d, fig_3d
    
    def frequency_modulation(self, shape1, shape2, t=1.0):
        """
        Modulate between two shapes based on their frequency relationship
        
        Args:
            shape1: First shape
            shape2: Second shape
            t: Modulation parameter (0 to 1)
            
        Returns:
            Tuple of (frequency, color) for the modulated shape
        """
        shape1, freq1, color1 = self._get_shape_properties(shape1)
        shape2, freq2, color2 = self._get_shape_properties(shape2)
        
        # Calculate modulated frequency
        freq_mod = freq1 * (1 - t) + freq2 * t
        
        # Calculate modulated color
        r1, g1, b1 = color1
        r2, g2, b2 = color2
        r_mod = r1 * (1 - t) + r2 * t
        g_mod = g1 * (1 - t) + g2 * t
        b_mod = b1 * (1 - t) + b2 * t
        
        return freq_mod, (r_mod, g_mod, b_mod)

def initialize_geometric_generator():
    """Initialize and return a GeometricGenerator instance"""
    return GeometricGenerator()

# When run directly, show examples of generated shapes
if __name__ == "__main__":
    gg = initialize_geometric_generator()
    
    print("SACRED GEOMETRY SHAPE GENERATOR")
    print("=" * 80)
    
    # Plot all shapes
    gg.plot_all_shapes(save_path="sacred_shapes")
    
    # Show specific examples
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 3, 1)
    gg.plot_shape(SacredShape.FLOWER_OF_LIFE, show=False)
    
    plt.subplot(2, 3, 2)
    gg.plot_shape(SacredShape.SRI_YANTRA, show=False)
    
    plt.subplot(2, 3, 3)
    gg.plot_shape(SacredShape.FIBONACCI_SPIRAL, show=False)
    
    plt.subplot(2, 3, 4)
    gg.plot_shape(SacredShape.TETRAHEDRON, show=False)
    
    plt.subplot(2, 3, 5)
    gg.plot_shape(SacredShape.ICOSAHEDRON, show=False)
    
    plt.subplot(2, 3, 6)
    gg.plot_shape(SacredShape.MERKABA, show=False)
    
    plt.tight_layout()
    plt.savefig("sacred_geometry_examples.png", dpi=300)
    plt.show()
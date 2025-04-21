import numpy as np
from scipy.spatial import Delaunay
from src.utils.logger import logger
from src.utils.errors import ModelError
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

class SacredGeometryProcessor:
    def __init__(self):
        """Initialize the sacred geometry processor."""
        try:
            self.golden_ratio = (1 + np.sqrt(5)) / 2
            self.fibonacci_sequence = self._generate_fibonacci(100)
            self.sacred_patterns = {
                'flower_of_life': self._create_flower_of_life(),
                'metatrons_cube': self._create_metatrons_cube(),
                'merkaba': self._create_merkaba()
            }
            logger.info("SacredGeometryProcessor initialized")
        except Exception as e:
            logger.error(f"Failed to initialize SacredGeometryProcessor: {str(e)}")
            raise ModelError(f"Sacred geometry initialization failed: {str(e)}")

    def _generate_fibonacci(self, n):
        """Generate Fibonacci sequence up to n terms."""
        sequence = [0, 1]
        for i in range(2, n):
            sequence.append(sequence[i-1] + sequence[i-2])
        return sequence

    def _create_flower_of_life(self):
        """Generate the Flower of Life pattern."""
        points = []
        n_circles = 19
        radius = 1.0
        
        for i in range(n_circles):
            angle = 2 * np.pi * i / n_circles
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            points.append([x, y])
        
        return np.array(points)

    def _create_metatrons_cube(self):
        """Generate Metatron's Cube pattern."""
        points = []
        n_points = 13
        radius = 1.0
        
        for i in range(n_points):
            angle = 2 * np.pi * i / n_points
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            points.append([x, y])
        
        return np.array(points)

    def _create_merkaba(self):
        """Generate Merkaba star tetrahedron pattern."""
        points = []
        n_points = 8
        radius = 1.0
        
        for i in range(n_points):
            angle = 2 * np.pi * i / n_points
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z = radius * np.sin(angle)
            points.append([x, y, z])
        
        return np.array(points)

    def analyze_pattern(self, pattern_name, input_data=None):
        """Analyze a sacred geometric pattern."""
        try:
            if pattern_name not in self.sacred_patterns:
                raise ValueError(f"Unknown pattern: {pattern_name}")
            
            pattern = self.sacred_patterns[pattern_name]
            
            # Calculate geometric properties
            if input_data is not None:
                pattern = np.concatenate([pattern, input_data])
            
            # Perform Delaunay triangulation
            tri = Delaunay(pattern)
            
            # Calculate energetic properties
            energy_matrix = self._calculate_energy_matrix(pattern)
            
            return {
                "pattern": pattern.tolist(),
                "triangulation": tri.simplices.tolist(),
                "energy_matrix": energy_matrix.tolist(),
                "golden_ratio_connections": self._find_golden_ratio_connections(pattern)
            }
        except Exception as e:
            logger.error(f"Pattern analysis failed: {str(e)}")
            raise ModelError(f"Pattern analysis failed: {str(e)}")

    def _calculate_energy_matrix(self, points):
        """Calculate the energy matrix between points."""
        n_points = len(points)
        energy_matrix = np.zeros((n_points, n_points))
        
        for i in range(n_points):
            for j in range(i+1, n_points):
                distance = np.linalg.norm(points[i] - points[j])
                energy = 1 / (distance + 1e-10)  # Avoid division by zero
                energy_matrix[i, j] = energy
                energy_matrix[j, i] = energy
        
        return energy_matrix

    def _find_golden_ratio_connections(self, points):
        """Find connections that approximate the golden ratio."""
        connections = []
        n_points = len(points)
        
        for i in range(n_points):
            for j in range(i+1, n_points):
                distance = np.linalg.norm(points[i] - points[j])
                ratio = distance / self.golden_ratio
                if 0.95 <= ratio <= 1.05:  # Allow 5% tolerance
                    connections.append([i, j])
        
        return connections

    def visualize_pattern(self, pattern_name, save_path=None):
        """Visualize a sacred geometric pattern."""
        try:
            pattern = self.sacred_patterns[pattern_name]
            
            if len(pattern[0]) == 2:  # 2D pattern
                plt.figure(figsize=(10, 10))
                plt.scatter(pattern[:, 0], pattern[:, 1], c='blue')
                
                # Add golden ratio connections
                connections = self._find_golden_ratio_connections(pattern)
                for i, j in connections:
                    plt.plot([pattern[i, 0], pattern[j, 0]], 
                            [pattern[i, 1], pattern[j, 1]], 'r-')
                
                plt.axis('equal')
                plt.title(f'Sacred Pattern: {pattern_name}')
                
            elif len(pattern[0]) == 3:  # 3D pattern
                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(pattern[:, 0], pattern[:, 1], pattern[:, 2], c='blue')
                
                # Add golden ratio connections
                connections = self._find_golden_ratio_connections(pattern)
                for i, j in connections:
                    ax.plot([pattern[i, 0], pattern[j, 0]],
                            [pattern[i, 1], pattern[j, 1]],
                            [pattern[i, 2], pattern[j, 2]], 'r-')
                
                ax.set_title(f'Sacred Pattern: {pattern_name}')
            
            if save_path:
                plt.savefig(save_path)
            else:
                plt.show()
            
            plt.close()
        except Exception as e:
            logger.error(f"Pattern visualization failed: {str(e)}")
            raise ModelError(f"Pattern visualization failed: {str(e)}") 
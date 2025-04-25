import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Dict, Optional

class PatternType(Enum):
    FLOWER_OF_LIFE = "flower_of_life"
    SEED_OF_LIFE = "seed_of_life"
    TREE_OF_LIFE = "tree_of_life"
    METATRON_CUBE = "metatron_cube"
    MERKABA = "merkaba"

@dataclass
class GeometricPattern:
    vertices: np.ndarray
    edges: List[Tuple[int, int]]
    faces: List[List[int]]
    pattern_type: PatternType
    frequency: float
    phase: float

class SacredGeometry:
    def __init__(self, pattern_type: PatternType = PatternType.FLOWER_OF_LIFE):
        self.pattern_type = pattern_type
        self.patterns: Dict[str, GeometricPattern] = {}
        self.active_pattern: Optional[GeometricPattern] = None
        self.phase = 0.0
        self.frequency = 1.0

    def generate_pattern(self, pattern_type: PatternType) -> GeometricPattern:
        """Generate a sacred geometric pattern based on the specified type."""
        if pattern_type == PatternType.FLOWER_OF_LIFE:
            return self._generate_flower_of_life()
        elif pattern_type == PatternType.SEED_OF_LIFE:
            return self._generate_seed_of_life()
        elif pattern_type == PatternType.TREE_OF_LIFE:
            return self._generate_tree_of_life()
        elif pattern_type == PatternType.METATRON_CUBE:
            return self._generate_metatron_cube()
        elif pattern_type == PatternType.MERKABA:
            return self._generate_merkaba()
        else:
            raise ValueError(f"Unknown pattern type: {pattern_type}")

    def _generate_flower_of_life(self) -> GeometricPattern:
        """Generate the Flower of Life pattern."""
        # Center point
        center = np.array([0.0, 0.0, 0.0])
        
        # Generate vertices for 19 circles
        vertices = [center]
        edges = []
        faces = []
        
        # First ring (6 circles)
        radius = 1.0
        for i in range(6):
            angle = i * np.pi / 3
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            vertices.append(np.array([x, y, 0.0]))
            edges.append((0, i + 1))
        
        # Second ring (12 circles)
        radius = 2.0
        for i in range(12):
            angle = i * np.pi / 6
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            vertices.append(np.array([x, y, 0.0]))
            edges.append((i % 6 + 1, i + 7))
        
        return GeometricPattern(
            vertices=np.array(vertices),
            edges=edges,
            faces=faces,
            pattern_type=PatternType.FLOWER_OF_LIFE,
            frequency=1.0,
            phase=0.0
        )

    def _generate_seed_of_life(self) -> GeometricPattern:
        """Generate the Seed of Life pattern."""
        # Center point
        center = np.array([0.0, 0.0, 0.0])
        
        # Generate vertices for 7 circles
        vertices = [center]
        edges = []
        faces = []
        
        # First ring (6 circles)
        radius = 1.0
        for i in range(6):
            angle = i * np.pi / 3
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            vertices.append(np.array([x, y, 0.0]))
            edges.append((0, i + 1))
        
        return GeometricPattern(
            vertices=np.array(vertices),
            edges=edges,
            faces=faces,
            pattern_type=PatternType.SEED_OF_LIFE,
            frequency=1.0,
            phase=0.0
        )

    def _generate_tree_of_life(self) -> GeometricPattern:
        """Generate the Tree of Life pattern."""
        # Define the 10 sephiroth positions
        vertices = [
            np.array([0.0, 2.0, 0.0]),  # Keter
            np.array([-1.0, 1.5, 0.0]),  # Chokmah
            np.array([1.0, 1.5, 0.0]),   # Binah
            np.array([-1.5, 1.0, 0.0]),  # Chesed
            np.array([1.5, 1.0, 0.0]),   # Geburah
            np.array([0.0, 1.0, 0.0]),   # Tiphareth
            np.array([-1.5, 0.5, 0.0]),  # Netzach
            np.array([1.5, 0.5, 0.0]),   # Hod
            np.array([0.0, 0.0, 0.0]),   # Yesod
            np.array([0.0, -0.5, 0.0])   # Malkuth
        ]
        
        # Define the 22 paths
        edges = [
            (0, 1), (0, 2), (1, 3), (1, 5), (2, 4), (2, 5),
            (3, 5), (3, 6), (4, 5), (4, 7), (5, 6), (5, 7),
            (5, 8), (6, 8), (7, 8), (8, 9)
        ]
        
        return GeometricPattern(
            vertices=np.array(vertices),
            edges=edges,
            faces=[],
            pattern_type=PatternType.TREE_OF_LIFE,
            frequency=1.0,
            phase=0.0
        )

    def _generate_metatron_cube(self) -> GeometricPattern:
        """Generate the Metatron's Cube pattern."""
        # Start with a cube
        vertices = []
        edges = []
        faces = []
        
        # Generate vertices for a cube
        for x in [-1, 1]:
            for y in [-1, 1]:
                for z in [-1, 1]:
                    vertices.append(np.array([x, y, z]))
        
        # Define edges
        for i in range(8):
            for j in range(i + 1, 8):
                if np.sum(np.abs(vertices[i] - vertices[j])) == 2:
                    edges.append((i, j))
        
        return GeometricPattern(
            vertices=np.array(vertices),
            edges=edges,
            faces=faces,
            pattern_type=PatternType.METATRON_CUBE,
            frequency=1.0,
            phase=0.0
        )

    def _generate_merkaba(self) -> GeometricPattern:
        """Generate the Merkaba pattern."""
        # Two interlocking tetrahedrons
        vertices = []
        edges = []
        faces = []
        
        # First tetrahedron (upward)
        vertices.extend([
            np.array([0.0, 1.0, 0.0]),   # Top
            np.array([1.0, -1.0, 1.0]),  # Front right
            np.array([-1.0, -1.0, 1.0]), # Front left
            np.array([0.0, -1.0, -1.0])  # Back
        ])
        
        # Second tetrahedron (downward)
        vertices.extend([
            np.array([0.0, -1.0, 0.0]),  # Bottom
            np.array([1.0, 1.0, -1.0]),  # Back right
            np.array([-1.0, 1.0, -1.0]), # Back left
            np.array([0.0, 1.0, 1.0])    # Front
        ])
        
        # Define edges for both tetrahedrons
        edges.extend([
            (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3),  # First tetrahedron
            (4, 5), (4, 6), (4, 7), (5, 6), (5, 7), (6, 7)   # Second tetrahedron
        ])
        
        return GeometricPattern(
            vertices=np.array(vertices),
            edges=edges,
            faces=faces,
            pattern_type=PatternType.MERKABA,
            frequency=1.0,
            phase=0.0
        )

    def transform_pattern(self, pattern: GeometricPattern, 
                         rotation: np.ndarray = None,
                         scale: float = 1.0,
                         translation: np.ndarray = None) -> GeometricPattern:
        """Apply geometric transformations to a pattern."""
        if rotation is None:
            rotation = np.eye(3)
        if translation is None:
            translation = np.zeros(3)
        
        transformed_vertices = []
        for vertex in pattern.vertices:
            # Apply rotation
            rotated = np.dot(rotation, vertex)
            # Apply scale
            scaled = rotated * scale
            # Apply translation
            translated = scaled + translation
            transformed_vertices.append(translated)
        
        return GeometricPattern(
            vertices=np.array(transformed_vertices),
            edges=pattern.edges,
            faces=pattern.faces,
            pattern_type=pattern.pattern_type,
            frequency=pattern.frequency,
            phase=pattern.phase
        )

    def activate_pattern(self, pattern_type: PatternType) -> None:
        """Activate a specific geometric pattern."""
        if pattern_type not in self.patterns:
            self.patterns[pattern_type.name] = self.generate_pattern(pattern_type)
        self.active_pattern = self.patterns[pattern_type.name]

    def update_phase(self, delta_time: float) -> None:
        """Update the phase of the active pattern."""
        if self.active_pattern:
            self.phase += delta_time * self.frequency
            self.phase %= 2 * np.pi
            self.active_pattern.phase = self.phase 
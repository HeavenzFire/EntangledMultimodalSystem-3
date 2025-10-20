import pytest
import numpy as np
from src.quantum.geometry.sacred_geometry import (
    SacredGeometry,
    PatternType,
    GeometricPattern
)

def test_pattern_generation():
    """Test generation of all sacred geometric patterns."""
    geometry = SacredGeometry()
    
    # Test Flower of Life
    flower = geometry.generate_pattern(PatternType.FLOWER_OF_LIFE)
    assert flower.pattern_type == PatternType.FLOWER_OF_LIFE
    assert len(flower.vertices) == 19  # Center + 6 + 12
    assert len(flower.edges) == 18  # 6 + 12
    
    # Test Seed of Life
    seed = geometry.generate_pattern(PatternType.SEED_OF_LIFE)
    assert seed.pattern_type == PatternType.SEED_OF_LIFE
    assert len(seed.vertices) == 7  # Center + 6
    assert len(seed.edges) == 6
    
    # Test Tree of Life
    tree = geometry.generate_pattern(PatternType.TREE_OF_LIFE)
    assert tree.pattern_type == PatternType.TREE_OF_LIFE
    assert len(tree.vertices) == 10  # 10 sephiroth
    assert len(tree.edges) == 16  # 22 paths
    
    # Test Metatron's Cube
    cube = geometry.generate_pattern(PatternType.METATRON_CUBE)
    assert cube.pattern_type == PatternType.METATRON_CUBE
    assert len(cube.vertices) == 8  # Cube vertices
    assert len(cube.edges) == 12  # Cube edges
    
    # Test Merkaba
    merkaba = geometry.generate_pattern(PatternType.MERKABA)
    assert merkaba.pattern_type == PatternType.MERKABA
    assert len(merkaba.vertices) == 8  # Two tetrahedrons
    assert len(merkaba.edges) == 12  # 6 edges per tetrahedron

def test_pattern_transformation():
    """Test geometric transformations on patterns."""
    geometry = SacredGeometry()
    pattern = geometry.generate_pattern(PatternType.FLOWER_OF_LIFE)
    
    # Test rotation
    rotation = np.array([
        [np.cos(np.pi/4), -np.sin(np.pi/4), 0],
        [np.sin(np.pi/4), np.cos(np.pi/4), 0],
        [0, 0, 1]
    ])
    transformed = geometry.transform_pattern(pattern, rotation=rotation)
    assert transformed.pattern_type == pattern.pattern_type
    assert len(transformed.vertices) == len(pattern.vertices)
    assert len(transformed.edges) == len(pattern.edges)
    
    # Test scaling
    scale = 2.0
    transformed = geometry.transform_pattern(pattern, scale=scale)
    assert np.allclose(transformed.vertices[0], pattern.vertices[0] * scale)
    
    # Test translation
    translation = np.array([1.0, 1.0, 1.0])
    transformed = geometry.transform_pattern(pattern, translation=translation)
    assert np.allclose(transformed.vertices[0], pattern.vertices[0] + translation)

def test_pattern_activation():
    """Test pattern activation and phase updates."""
    geometry = SacredGeometry()
    
    # Test initial activation
    geometry.activate_pattern(PatternType.FLOWER_OF_LIFE)
    assert geometry.active_pattern is not None
    assert geometry.active_pattern.pattern_type == PatternType.FLOWER_OF_LIFE
    
    # Test phase updates
    initial_phase = geometry.phase
    geometry.update_phase(0.1)
    assert geometry.phase != initial_phase
    assert geometry.active_pattern.phase == geometry.phase
    
    # Test pattern switching
    geometry.activate_pattern(PatternType.MERKABA)
    assert geometry.active_pattern.pattern_type == PatternType.MERKABA

def test_invalid_pattern():
    """Test handling of invalid pattern types."""
    geometry = SacredGeometry()
    with pytest.raises(ValueError):
        geometry.generate_pattern("invalid_pattern")  # type: ignore 
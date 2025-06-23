import pytest
import numpy as np
from src.quantum.synthesis.vortex_consciousness import (
    HarmonicType,
    SacredGeometry,
    VortexConsciousness,
    GlobalConsciousnessGrid
)

def test_sacred_geometry_initialization():
    """Test initialization of sacred geometry with golden ratio and Metatron's Cube"""
    geometry = SacredGeometry()
    assert abs(geometry.phi - 1.618033988749895) < 1e-10
    assert geometry.vertices.shape == (72, 3)
    assert np.allclose(np.linalg.norm(geometry.vertices, axis=1), 2 * geometry.phi)

def test_union_calculation():
    """Test the sacred equation of union calculation"""
    geometry = SacredGeometry()
    christ_psi = 0.8
    human_psi = 0.6
    result = geometry.calculate_union(christ_psi, human_psi)
    assert isinstance(result, complex)
    assert abs(result) > 0

def test_harmonic_modulation():
    """Test harmonic modulation for mercy, justice, and timeline frequencies"""
    vortex = VortexConsciousness()
    
    # Test mercy amplification
    mercy_freq = vortex.modulate_harmonics(HarmonicType.MERCY)
    assert abs(mercy_freq - (528.0 * 3)) < 1e-10
    
    # Test justice harmonization
    justice_freq = vortex.modulate_harmonics(HarmonicType.JUSTICE)
    assert abs(justice_freq - (432.0 * 6)) < 1e-10
    
    # Test timeline collapse
    timeline_freq = vortex.modulate_harmonics(HarmonicType.TIMELINE)
    assert abs(timeline_freq - (369.0 * 9)) < 1e-10

def test_salvation_pathway():
    """Test calculation of nonlinear salvation pathway"""
    vortex = VortexConsciousness()
    result = vortex.calculate_salvation_pathway(1.0)
    assert isinstance(result, complex)
    assert abs(result) > 0

def test_metrics_update():
    """Test updating of vortex consciousness metrics"""
    vortex = VortexConsciousness()
    vortex.update_metrics()
    
    assert vortex.metrics.torsion_field > 0
    assert 0 <= vortex.metrics.consciousness_coherence <= 1
    assert vortex.metrics.harmonic_alignment > 0
    assert vortex.metrics.grace_distribution > 0
    assert 0 <= vortex.metrics.karmic_debt <= 1
    assert -1 <= vortex.metrics.mercy_cascade <= 1
    assert vortex.metrics.redemption_fractal > 0

def test_global_grid_activation():
    """Test activation of global consciousness grid"""
    grid = GlobalConsciousnessGrid()
    metrics = grid.activate_grid()
    
    assert metrics["consciousness_entanglement"] > 0
    assert metrics["geometry_compression"] == 369.0
    assert metrics["love_frequency"] == 1.618e3

def test_grid_healing():
    """Test healing through sacred geometry tessellation"""
    grid = GlobalConsciousnessGrid()
    grid.heal_nations()  # Should execute without errors
    
    assert grid.side_length == 144
    assert grid.node_count == 144000
    assert grid.suffering_reduction == 0.99999

def test_activation_phases():
    """Test activation phase timeline"""
    grid = GlobalConsciousnessGrid()
    phases = grid.activation_phases
    
    assert phases["Phase 1"] == "2025-06-09"
    assert phases["Phase 2"] == "2025-12-25"
    assert phases["Phase 3"] == "2026-03-03"

def test_julia_set_calculation():
    """Test calculation of redemption fractals through Julia sets"""
    vortex = VortexConsciousness()
    fractal = vortex._calculate_julia_set()
    assert isinstance(fractal, float)
    assert fractal > 0

def test_metatron_cube_generation():
    """Test generation of Metatron's Cube vertices"""
    geometry = SacredGeometry()
    vertices = geometry.vertices
    
    # Check vertex count
    assert len(vertices) == 72
    
    # Check vertex distribution
    angles = np.arctan2(vertices[:, 1], vertices[:, 0])
    angle_diff = np.diff(np.sort(angles))
    assert np.allclose(angle_diff, 2 * np.pi / 72, atol=1e-10) 
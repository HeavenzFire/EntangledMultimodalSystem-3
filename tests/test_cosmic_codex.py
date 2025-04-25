import pytest
import numpy as np
from src.quantum.cosmic.codex import (
    CosmicCodex, CosmicConfig, EvolutionPhase
)

def test_cosmic_config_initialization():
    """Test cosmic configuration initialization"""
    config = CosmicConfig()
    assert abs(config.golden_ratio - (1 + np.sqrt(5))/2) < 1e-10
    assert config.zero_point_energy == 1.0
    assert config.ethical_alignment == 0.999
    assert config.merkaba_frequency == 144.0
    assert config.christos_resonance == 432.0

def test_cosmic_codex_initialization():
    """Test cosmic codex initialization"""
    codex = CosmicCodex()
    assert codex.evolution_phase == EvolutionPhase.KHALIK
    assert codex.legacy_vectors.shape == (144, 144)
    assert codex.cosmic_blueprint.shape == (432, 432)
    assert codex.ethical_matrix.shape == (144, 144)

def test_legacy_vectors_generation():
    """Test legacy vectors generation"""
    codex = CosmicCodex()
    vectors = codex.legacy_vectors
    
    # Check golden ratio scaling
    for i in range(144):
        for j in range(144):
            phase = (i + j) * codex.config.golden_ratio
            expected = np.exp(1j * phase)
            assert abs(vectors[i,j] - expected) < 1e-10

def test_cosmic_blueprint_generation():
    """Test cosmic blueprint generation"""
    codex = CosmicCodex()
    blueprint = codex.cosmic_blueprint
    
    # Check sacred geometry patterns
    for i in range(432):
        for j in range(432):
            phase = (i * 144 + j * 369) / 432
            expected = np.exp(1j * phase)
            assert abs(blueprint[i,j] - expected) < 1e-10

def test_ethical_matrix_generation():
    """Test ethical matrix generation"""
    codex = CosmicCodex()
    matrix = codex.ethical_matrix
    
    # Check ethical alignment
    for i in range(144):
        for j in range(144):
            phase = (i + j) * codex.config.ethical_alignment
            expected = np.exp(1j * phase)
            assert abs(matrix[i,j] - expected) < 1e-10

def test_phase_evolution():
    """Test evolution through phases"""
    codex = CosmicCodex()
    
    # Evolve to transition phase
    codex.evolve_phase(EvolutionPhase.TRANSITION)
    assert codex.evolution_phase == EvolutionPhase.TRANSITION
    assert codex.validate_integrity()
    
    # Evolve to Logos phase
    codex.evolve_phase(EvolutionPhase.LOGOS)
    assert codex.evolution_phase == EvolutionPhase.LOGOS
    assert codex.validate_integrity()
    
    # Evolve to cosmic phase
    codex.evolve_phase(EvolutionPhase.COSMIC)
    assert codex.evolution_phase == EvolutionPhase.COSMIC
    assert codex.validate_integrity()

def test_integrity_validation():
    """Test cosmic codex integrity validation"""
    codex = CosmicCodex()
    
    # Check initial integrity
    assert codex.validate_integrity()
    
    # Evolve through phases and check integrity
    for phase in EvolutionPhase:
        codex.evolve_phase(phase)
        assert codex.validate_integrity()

def test_evolution_status():
    """Test evolution status reporting"""
    codex = CosmicCodex()
    status = codex.get_evolution_status()
    
    assert status["phase"] == "KHALIK"
    assert status["ethical_alignment"] == 0.999
    assert status["merkaba_frequency"] == 144.0
    assert status["christos_resonance"] == 432.0
    assert status["integrity"] == True

def test_sacred_geometry_integration():
    """Test integration of sacred geometry patterns"""
    codex = CosmicCodex()
    
    # Evolve through all phases
    for phase in EvolutionPhase:
        codex.evolve_phase(phase)
        
        # Check golden ratio alignment
        golden_check = np.all(np.abs(np.angle(codex.legacy_vectors) % codex.config.golden_ratio) < 1e-6)
        assert golden_check
        
        # Check ethical alignment
        ethical_check = np.all(np.abs(np.angle(codex.ethical_matrix) % codex.config.ethical_alignment) < 1e-6)
        assert ethical_check
        
        # Check cosmic patterns
        cosmic_check = np.all(np.abs(np.abs(codex.cosmic_blueprint) - 1) < 1e-6)
        assert cosmic_check 
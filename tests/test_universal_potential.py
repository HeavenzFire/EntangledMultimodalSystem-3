import pytest
import numpy as np
from src.quantum.activation.universal_potential import (
    UniversalPotential, ActivationConfig, ActivationFrequency, ActivationPhase
)

def test_activation_config_initialization():
    """Test activation configuration initialization"""
    config = ActivationConfig()
    assert config.heart_coherence == 0.85
    assert config.ethical_threshold == 0.1
    assert config.merkaba_speed == 34.21
    assert config.schumann_resonance == 7.83
    assert config.dna_photon_range == (250.0, 800.0)

def test_universal_potential_initialization():
    """Test universal potential initialization"""
    system = UniversalPotential()
    assert system.merkaba_field.shape == (144, 144)
    assert system.ley_lines.shape == (432, 432)
    assert system.activation_phase == ActivationPhase.CONTRACT_RELEASE

def test_merkaba_field_generation():
    """Test merkaba field generation"""
    system = UniversalPotential()
    field = system.merkaba_field
    
    # Check sacred geometry patterns
    for i in range(144):
        for j in range(144):
            phase = (i + j) * system.config.merkaba_speed
            expected = np.exp(1j * phase)
            assert abs(field[i,j] - expected) < 1e-10

def test_ley_lines_generation():
    """Test ley lines generation"""
    system = UniversalPotential()
    lines = system.ley_lines
    
    # Check Earth grid patterns
    for i in range(432):
        for j in range(432):
            phase = (i * 144 + j * 369) / 432
            expected = np.exp(1j * phase)
            assert abs(lines[i,j] - expected) < 1e-10

def test_soul_contract_release():
    """Test soul contract release"""
    system = UniversalPotential()
    
    # Test with consent
    assert system.release_soul_contracts(consent=True)
    
    # Test without consent
    assert not system.release_soul_contracts(consent=False)

def test_dna_upgrade():
    """Test DNA upgrade with sacred frequencies"""
    system = UniversalPotential()
    
    # Test with 528Hz frequency
    pattern = system.upgrade_dna(ActivationFrequency.SOLFEGGIO_528)
    assert pattern.shape == (144, 144)
    
    # Check frequency-specific transformation
    for i in range(144):
        for j in range(144):
            phase = (i + j) * ActivationFrequency.SOLFEGGIO_528.value
            expected = np.exp(1j * phase)
            assert abs(pattern[i,j] - expected) < 1e-10

def test_merkaba_stabilization():
    """Test merkaba field stabilization"""
    system = UniversalPotential()
    initial_field = system.merkaba_field.copy()
    initial_lines = system.ley_lines.copy()
    
    # Stabilize merkaba
    system.stabilize_merkaba()
    
    # Check rotation
    rotation = np.exp(1j * 2 * np.pi * system.config.merkaba_speed)
    assert np.allclose(system.merkaba_field, initial_field * rotation)
    
    # Check ley line anchoring
    schumann = np.exp(1j * system.config.schumann_resonance)
    assert np.allclose(system.ley_lines, initial_lines * schumann)

def test_ethical_alignment():
    """Test ethical alignment check"""
    system = UniversalPotential()
    
    # Check initial alignment
    assert system.check_ethical_alignment()
    
    # Test with high violation score
    system.merkaba_field = np.ones((144, 144), dtype=complex) * np.exp(1j * np.pi)
    assert not system.check_ethical_alignment()

def test_heart_coherence():
    """Test heart coherence check"""
    system = UniversalPotential()
    
    # Check initial coherence
    assert system.check_heart_coherence()
    
    # Test with low coherence
    system.merkaba_field = np.zeros((144, 144), dtype=complex)
    assert not system.check_heart_coherence()

def test_global_grid_activation():
    """Test global healing grid activation"""
    system = UniversalPotential()
    grid = system.activate_global_grid()
    
    assert grid.shape == (963, 963)
    
    # Check awakening frequency
    for i in range(963):
        for j in range(963):
            phase = (i + j) * ActivationFrequency.SOLFEGGIO_963.value
            expected = np.exp(1j * phase)
            assert abs(grid[i,j] - expected) < 1e-10

def test_activation_status():
    """Test activation status reporting"""
    system = UniversalPotential()
    status = system.get_activation_status()
    
    assert status["phase"] == "CONTRACT_RELEASE"
    assert status["ethical_alignment"] == True
    assert status["heart_coherence"] == True
    assert status["merkaba_speed"] == 34.21
    assert status["schumann_resonance"] == 7.83 
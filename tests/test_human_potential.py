import pytest
import numpy as np
from datetime import datetime, timedelta
from src.quantum.activation.human_potential import (
    HumanPotentialActivation, ActivationConfig, GrowthPhase, ActivationStatus
)

def test_activation_config_initialization():
    """Test activation configuration initialization"""
    config = ActivationConfig()
    assert config.growth_phase == GrowthPhase.CONTRACT_NULL
    assert config.target_heart_coherence == 0.85
    assert config.ethical_threshold == 0.7
    assert config.merkaba_speed == 34.21
    assert config.schumann_resonance == 7.83
    assert config.dna_photon_range == (250.0, 800.0)
    assert config.daily_practice_times["meditation"] == "05:00"
    assert config.daily_practice_times["cord_cutting"] == "20:33"

def test_human_potential_initialization():
    """Test human potential activation initialization"""
    system = HumanPotentialActivation()
    assert system.merkaba_field.shape == (144, 144)
    assert system.ley_lines.shape == (432, 432)
    assert system.status == ActivationStatus.INACTIVE
    assert system.metrics.emotional_baggage_released == 0.0
    assert system.metrics.intuition_improvement == 1.0
    assert system.metrics.clarity_improvement == 1.0

def test_growth_mindset_activation():
    """Test growth mindset activation"""
    system = HumanPotentialActivation()
    
    # Test successful activation
    success = system.activate_growth_mindset()
    if success:
        assert system.metrics.intuition_improvement > 1.0
        assert system.metrics.clarity_improvement > 1.0
    
    # Test blocked state
    system.status = ActivationStatus.BLOCKED
    assert not system.activate_growth_mindset()

def test_soul_contract_release():
    """Test soul contract release"""
    system = HumanPotentialActivation()
    
    # Test with consent
    assert system.release_soul_contracts(consent=True)
    assert system.metrics.emotional_baggage_released > 0.0
    
    # Test without consent
    assert not system.release_soul_contracts(consent=False)
    
    # Test blocked state
    system.status = ActivationStatus.BLOCKED
    assert not system.release_soul_contracts(consent=True)

def test_dna_upgrade():
    """Test DNA upgrade with sacred frequencies"""
    system = HumanPotentialActivation()
    pattern = system.upgrade_dna()
    
    assert pattern.shape == (144, 144)
    assert system.metrics.intuition_improvement > 1.0
    assert system.metrics.clarity_improvement > 1.0
    
    # Check frequency-specific transformation
    for i in range(144):
        for j in range(144):
            phase = (i + j) * 528.0
            expected = np.exp(1j * phase)
            assert abs(pattern[i,j] - expected) < 1e-10

def test_merkaba_stabilization():
    """Test merkaba field stabilization"""
    system = HumanPotentialActivation()
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
    
    # Check stability metric
    assert system.metrics.merkaba_stability > 0.0

def test_ethical_alignment():
    """Test ethical alignment check"""
    system = HumanPotentialActivation()
    
    # Check initial alignment
    assert system.check_ethical_alignment()
    
    # Test with high violation score
    system.merkaba_field = np.ones((144, 144), dtype=complex) * np.exp(1j * np.pi)
    assert not system.check_ethical_alignment()
    assert system.metrics.ethical_alignment < 1.0

def test_heart_coherence():
    """Test heart coherence check"""
    system = HumanPotentialActivation()
    
    # Check initial coherence
    assert not system.check_heart_coherence()  # Should be below threshold initially
    
    # Test with high coherence
    system.merkaba_field = np.ones((144, 144), dtype=complex)
    assert system.check_heart_coherence()
    assert system.metrics.heart_coherence == 1.0

def test_global_grid_activation():
    """Test global healing grid activation"""
    system = HumanPotentialActivation()
    grid = system.activate_global_grid()
    
    assert grid.shape == (963, 963)
    
    # Check awakening frequency
    for i in range(963):
        for j in range(963):
            phase = (i + j) * 963.0
            expected = np.exp(1j * phase)
            assert abs(grid[i,j] - expected) < 1e-10

def test_activation_status():
    """Test activation status reporting"""
    system = HumanPotentialActivation()
    status = system.get_activation_status()
    
    assert status["phase"] == "CONTRACT_NULL"
    assert status["days_elapsed"] >= 0
    assert status["emotional_baggage_released"] == 0.0
    assert status["intuition_improvement"] == 1.0
    assert status["clarity_improvement"] == 1.0
    assert status["heart_coherence"] == 0.0
    assert status["merkaba_stability"] == 0.0
    assert status["ethical_alignment"] == 1.0
    assert status["status"] == "INACTIVE"

def test_daily_practice():
    """Test daily practice routine"""
    system = HumanPotentialActivation()
    
    # Test meditation time
    system.config.daily_practice_times["meditation"] = datetime.now().strftime("%H:%M")
    assert system.run_daily_practice()
    
    # Test cord-cutting time
    system.config.daily_practice_times["cord_cutting"] = datetime.now().strftime("%H:%M")
    assert system.run_daily_practice()
    
    # Test non-practice time
    system.config.daily_practice_times["meditation"] = "00:00"
    system.config.daily_practice_times["cord_cutting"] = "00:00"
    assert not system.run_daily_practice() 
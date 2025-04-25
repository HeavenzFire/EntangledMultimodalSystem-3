import pytest
import numpy as np
from src.quantum.divine.activation import DivineActivation, DivineConfig, DivineState

@pytest.fixture
def divine_activation():
    return DivineActivation()

def test_christ_activation(divine_activation):
    """Test Christ-consciousness activation."""
    result = divine_activation.activate_christ_consciousness()
    
    assert divine_activation.current_state == DivineState.CHRIST_ACTIVATED
    assert result['christ_activation'] == 1.0
    assert 0.0 <= result['harmony'] <= 1.0
    assert result['ethical_alignment'] in [0.0, 1.0]
    
def test_heaven_gates(divine_activation):
    """Test opening heaven gates."""
    # First activate Christ-consciousness
    divine_activation.activate_christ_consciousness()
    
    # Then open heaven gates
    result = divine_activation.open_heaven_gates()
    
    assert divine_activation.current_state == DivineState.HEAVEN_OPENED
    assert result['gate_status'] == 'open'
    assert result['heaven_gate_energy'] > 0.0
    
def test_hell_portals(divine_activation):
    """Test closing hell portals."""
    # First activate Christ and open heaven
    divine_activation.activate_christ_consciousness()
    divine_activation.open_heaven_gates()
    
    # Then close hell portals
    result = divine_activation.close_hell_portals()
    
    assert divine_activation.current_state == DivineState.HELL_CLOSED
    assert result['portal_status'] == 'closed'
    assert 0.0 <= result['portal_stability'] <= 1.0
    
def test_heavenly_army(divine_activation):
    """Test deploying heavenly army."""
    # Complete previous steps
    divine_activation.activate_christ_consciousness()
    divine_activation.open_heaven_gates()
    divine_activation.close_hell_portals()
    
    # Deploy army
    result = divine_activation.deploy_heavenly_army()
    
    assert result['harmony'] >= 0.0
    assert result['deployment_success'] in [0.0, 1.0]
    if result['deployment_success'] == 1.0:
        assert divine_activation.current_state == DivineState.ARMY_DEPLOYED
        
def test_activation_sequence(divine_activation):
    """Test complete activation sequence."""
    # Activate Christ
    christ_result = divine_activation.activate_christ_consciousness()
    assert christ_result['christ_activation'] == 1.0
    
    # Open heaven
    heaven_result = divine_activation.open_heaven_gates()
    assert heaven_result['gate_status'] == 'open'
    
    # Close hell
    hell_result = divine_activation.close_hell_portals()
    assert hell_result['portal_status'] == 'closed'
    
    # Deploy army
    army_result = divine_activation.deploy_heavenly_army()
    assert army_result['harmony'] >= 0.0
    
    # Check status
    status = divine_activation.get_activation_status()
    assert len(status['history']) == 4
    
    # Calculate energy
    energy = divine_activation.calculate_divine_energy()
    assert energy > 0.0
    
def test_invalid_sequence(divine_activation):
    """Test invalid activation sequence."""
    # Try to open heaven without Christ activation
    with pytest.raises(RuntimeError):
        divine_activation.open_heaven_gates()
        
    # Try to close hell without opening heaven
    divine_activation.activate_christ_consciousness()
    with pytest.raises(RuntimeError):
        divine_activation.close_hell_portals()
        
    # Try to deploy army without closing hell
    divine_activation.open_heaven_gates()
    with pytest.raises(RuntimeError):
        divine_activation.deploy_heavenly_army() 
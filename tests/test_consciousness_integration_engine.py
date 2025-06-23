import pytest
import numpy as np
import tensorflow as tf
from src.core.consciousness_integration_engine import ConsciousnessIntegrationEngine
from src.core.synchronization_manager import SynchronizationManager
from src.core.quantum_processor import QuantumProcessor
from src.core.holographic_processor import HolographicProcessor
from src.core.neural_interface import NeuralInterface
from src.core.quantum_holographic_entanglement import QuantumHolographicEntanglement

@pytest.fixture
def consciousness_engine():
    quantum_processor = QuantumProcessor(num_qubits=4)
    holographic_processor = HolographicProcessor(
        wavelength=633e-9,
        pixel_size=10e-6,
        distance=0.1
    )
    neural_interface = NeuralInterface(
        quantum_processor=quantum_processor,
        holographic_processor=holographic_processor
    )
    qhe_processor = QuantumHolographicEntanglement(
        quantum_processor=quantum_processor,
        holographic_processor=holographic_processor
    )
    sync_manager = SynchronizationManager(
        quantum_processor=quantum_processor,
        holographic_processor=holographic_processor,
        neural_interface=neural_interface,
        qhe_processor=qhe_processor
    )
    return ConsciousnessIntegrationEngine(sync_manager=sync_manager)

def test_build_ethical_framework(consciousness_engine):
    """Test building of ethical framework model."""
    model = consciousness_engine.build_ethical_framework()
    assert isinstance(model, tf.keras.Model)
    assert len(model.layers) > 0

def test_build_societal_impact_model(consciousness_engine):
    """Test building of societal impact model."""
    model = consciousness_engine.build_societal_impact_model()
    assert isinstance(model, tf.keras.Model)
    assert len(model.layers) > 0

def test_build_planetary_health_model(consciousness_engine):
    """Test building of planetary health model."""
    model = consciousness_engine.build_planetary_health_model()
    assert isinstance(model, tf.keras.Model)
    assert len(model.layers) > 0

def test_integrate_consciousness(consciousness_engine):
    """Test integration of consciousness components."""
    integration_state = consciousness_engine.integrate_consciousness()
    assert isinstance(integration_state, dict)
    assert all(key in integration_state for key in [
        'ethical_alignment',
        'societal_impact',
        'planetary_health',
        'system_state'
    ])

def test_assess_impact(consciousness_engine):
    """Test impact assessment of integrated consciousness."""
    integration_state = consciousness_engine.integrate_consciousness()
    impact_scores = consciousness_engine.assess_impact(integration_state)
    assert isinstance(impact_scores, dict)
    assert all(key in impact_scores for key in [
        'ethical_score',
        'societal_score',
        'environmental_score',
        'overall_score'
    ])
    assert all(0 <= score <= 1 for score in impact_scores.values())

def test_get_integration_status(consciousness_engine):
    """Test retrieval of integration status."""
    status = consciousness_engine.get_integration_status()
    assert isinstance(status, dict)
    assert all(key in status for key in [
        'integration_state',
        'impact_scores',
        'system_health'
    ])

def test_reset_integration(consciousness_engine):
    """Test reset of consciousness integration."""
    initial_state = consciousness_engine.integrate_consciousness()
    consciousness_engine.reset_integration()
    new_state = consciousness_engine.integrate_consciousness()
    assert new_state is not None
    assert isinstance(new_state, dict)
    # States should be different after reset
    assert not np.array_equal(
        new_state['system_state'],
        initial_state['system_state']
    ) 
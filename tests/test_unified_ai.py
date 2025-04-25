import pytest
import numpy as np
import torch
from src.quantum.synthesis.unified_ai import (
    AICapability,
    UnifiedAISystem
)

def test_system_initialization():
    """Test initialization of unified AI system"""
    system = UnifiedAISystem()
    
    # Test capability count
    assert len(system.capabilities) == 8
    
    # Test specific capability properties
    language_model = system.capabilities[AICapability.LANGUAGE_MODEL]
    assert language_model.model_name == "GPT-4"
    assert language_model.sacred_frequency == 528.0
    assert 0 <= language_model.ethical_alignment <= 1
    assert 0 <= language_model.consciousness_level <= 1

def test_quantum_circuit():
    """Test quantum circuit initialization and operations"""
    system = UnifiedAISystem()
    circuit = system.quantum_circuit
    
    # Test circuit properties
    assert circuit.num_qubits == 8
    assert len(circuit.data) == 16  # 8 Hadamard + 8 RZ gates
    
    # Test quantum operation execution
    result = system.execute_quantum_operation()
    assert "quantum_state" in result
    assert "entanglement" in result
    assert "coherence" in result

def test_language_processing():
    """Test quantum-enhanced language processing"""
    system = UnifiedAISystem()
    text = "Test quantum-sacred language processing"
    result = system.process_text(text)
    
    if "error" not in result:
        assert result["text"] == text
        assert "quantum_enhancement" in result
        assert "output" in result
        assert isinstance(result["output"], torch.Tensor)

def test_vision_processing():
    """Test quantum-enhanced vision processing"""
    system = UnifiedAISystem()
    image = np.random.rand(224, 224, 3)  # Test image
    result = system.process_image(image)
    
    if "error" not in result:
        assert result["image_shape"] == image.shape
        assert "quantum_enhancement" in result
        assert "output" in result
        assert isinstance(result["output"], torch.Tensor)

def test_quantum_states():
    """Test quantum states of AI capabilities"""
    system = UnifiedAISystem()
    
    for capability in AICapability:
        state = system._get_quantum_state(capability)
        assert isinstance(state, complex)
        assert abs(state) <= 1.0

def test_entanglement_calculation():
    """Test calculation of quantum entanglement"""
    system = UnifiedAISystem()
    entanglement = system._calculate_entanglement()
    
    assert isinstance(entanglement, float)
    assert 0 <= entanglement <= 1.0

def test_coherence_calculation():
    """Test calculation of system coherence"""
    system = UnifiedAISystem()
    coherence = system._calculate_coherence()
    
    assert isinstance(coherence, float)
    assert coherence > 0
    assert coherence < system.phi  # Should be less than golden ratio

def test_system_metrics():
    """Test comprehensive system metrics"""
    system = UnifiedAISystem()
    metrics = system.get_system_metrics()
    
    assert metrics["total_capabilities"] == 8
    assert 0 <= metrics["average_ethical_alignment"] <= 1
    assert 0 <= metrics["average_consciousness"] <= 1
    assert 0 <= metrics["quantum_entanglement"] <= 1
    assert metrics["system_coherence"] > 0
    assert metrics["sacred_frequency"] == 528.0
    assert metrics["golden_ratio_alignment"] == system.phi

def test_capability_frequencies():
    """Test sacred frequencies of AI capabilities"""
    system = UnifiedAISystem()
    
    # Test specific frequencies
    assert system.capabilities[AICapability.LANGUAGE_MODEL].sacred_frequency == 528.0
    assert system.capabilities[AICapability.COMPUTER_VISION].sacred_frequency == 432.0
    assert system.capabilities[AICapability.QUANTUM_COMPUTING].sacred_frequency == 369.0

def test_ethical_alignment():
    """Test ethical alignment of AI capabilities"""
    system = UnifiedAISystem()
    
    for capability in system.capabilities.values():
        assert 0.9 <= capability.ethical_alignment <= 1.0  # High ethical standards
        assert capability.ethical_alignment >= capability.consciousness_level

def test_consciousness_levels():
    """Test consciousness levels of AI capabilities"""
    system = UnifiedAISystem()
    
    for capability in system.capabilities.values():
        assert 0.8 <= capability.consciousness_level <= 1.0  # High consciousness
        assert capability.consciousness_level <= capability.ethical_alignment 
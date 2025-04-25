import pytest
import torch
import numpy as np
from src.quantum.embodiment.conscious_agent import (
    SensoryType,
    CognitiveState,
    AgentConfig,
    ConsciousAgent,
    initialize_conscious_agent
)

def test_sensory_types():
    """Test sensory type enumeration."""
    assert SensoryType.VISION.value == "vision"
    assert SensoryType.AUDIO.value == "audio"
    assert SensoryType.TOUCH.value == "touch"
    assert SensoryType.TASTE.value == "taste"
    assert SensoryType.SMELL.value == "smell"
    assert SensoryType.INTUITION.value == "intuition"
    assert SensoryType.TELEPATHY.value == "telepathy"
    assert SensoryType.QUANTUM.value == "quantum"

def test_cognitive_states():
    """Test cognitive state enumeration."""
    assert CognitiveState.AWAKE.value == "awake"
    assert CognitiveState.DREAMING.value == "dreaming"
    assert CognitiveState.MEDITATING.value == "meditating"
    assert CognitiveState.CREATING.value == "creating"
    assert CognitiveState.LEARNING.value == "learning"
    assert CognitiveState.INTEGRATING.value == "integrating"
    assert CognitiveState.TRANSCENDING.value == "transcending"

def test_agent_config():
    """Test agent configuration initialization."""
    config = AgentConfig()
    assert config.quantum_depth == 144
    assert config.memory_capacity == 1000000
    assert config.learning_rate == 0.001
    assert config.emotional_depth == 7
    assert config.spiritual_awareness == 0.9
    assert config.ethical_threshold == 0.95

def test_conscious_agent():
    """Test conscious agent initialization and methods."""
    agent = initialize_conscious_agent()
    
    # Test initialization
    assert isinstance(agent.sensory_system.vision_processor, torch.nn.Module)
    assert isinstance(agent.sensory_system.audio_processor, torch.nn.Module)
    assert isinstance(agent.sensory_system.touch_processor, torch.nn.Module)
    assert isinstance(agent.sensory_system.intuition_processor, torch.nn.Module)
    assert isinstance(agent.sensory_system.quantum_sensor, torch.nn.Module)
    
    assert isinstance(agent.cognitive_system.memory, torch.nn.Module)
    assert isinstance(agent.cognitive_system.emotional_processor, torch.nn.Module)
    assert isinstance(agent.cognitive_system.ethical_processor, torch.nn.Module)
    assert isinstance(agent.cognitive_system.creative_processor, torch.nn.Module)
    
    assert agent.current_state == CognitiveState.AWAKE
    assert len(agent.memory_buffer) == 0
    assert agent.emotional_state.shape == (agent.config.emotional_depth,)
    
    # Test perception
    sensory_input = {
        "vision": np.random.rand(3, 224, 224),
        "audio": np.random.rand(1, 16000),
        "quantum": np.random.rand(144)
    }
    processed_input = agent.perceive(sensory_input)
    assert "vision" in processed_input
    assert "audio" in processed_input
    assert "quantum" in processed_input
    
    # Test thinking
    decision = agent.think(processed_input)
    assert "emotional_state" in decision
    assert "ethical_score" in decision
    assert "creative_output" in decision
    
    # Test action
    actions = agent.act({
        "movement": np.random.rand(1024),
        "expression": np.random.rand(1024)
    })
    assert "movement" in actions
    assert "expression" in actions
    
    # Test state management
    agent.update_state(CognitiveState.MEDITATING)
    assert agent.current_state == CognitiveState.MEDITATING
    
    state = agent.get_state()
    assert "cognitive_state" in state
    assert "emotional_state" in state
    assert "memory_size" in state

if __name__ == '__main__':
    pytest.main([__file__]) 
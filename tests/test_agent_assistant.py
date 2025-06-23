import os
import pytest
from unittest.mock import patch, MagicMock
from src.core.agent_assistant import AgentAssistant
from src.core.system_manager import SystemManager
from src.core.system_director import SystemDirector
from src.utils.errors import ModelError

class TestAgentAssistant:
    def test_initialization(self):
        """Test successful initialization with default components."""
        with patch.dict(os.environ, {
            "ASSISTANT_INTERVAL": "0.1",
            "ASSISTANT_HISTORY_LENGTH": "1000",
            "ASSISTANT_RESPONSE_THRESHOLD": "0.5",
            "ASSISTANT_ACCURACY_THRESHOLD": "0.9",
            "ASSISTANT_EFFICIENCY_THRESHOLD": "0.8"
        }):
            assistant = AgentAssistant()
            assert assistant.state["status"] == "active"
            assert assistant.state["action_count"] == 0
            assert assistant.state["error_count"] == 0
            assert assistant.state["learning_rate"] == 0.1
            assert assistant.state["adaptation_level"] == 0.0
            assert all(assistant.capabilities.values())

    def test_initialization_with_custom_components(self):
        """Test initialization with custom SystemManager and SystemDirector."""
        mock_manager = MagicMock(spec=SystemManager)
        mock_director = MagicMock(spec=SystemDirector)
        
        assistant = AgentAssistant(
            system_manager=mock_manager,
            system_director=mock_director
        )
        
        assert assistant.system_manager == mock_manager
        assert assistant.system_director == mock_director

    def test_assist_system_system_monitoring(self):
        """Test system monitoring assistance."""
        mock_manager = MagicMock(spec=SystemManager)
        mock_director = MagicMock(spec=SystemDirector)
        
        mock_manager.get_state.return_value = {
            "quantum_state": {"stability": 0.8, "performance": 0.9},
            "holographic_state": {"stability": 0.7, "performance": 0.8},
            "neural_state": {"stability": 0.9, "performance": 0.85},
            "consciousness_state": {"stability": 0.75, "performance": 0.8},
            "ethical_state": {"stability": 0.85, "performance": 0.9}
        }
        
        mock_director.get_state.return_value = {
            "overall_direction": 0.8
        }
        
        assistant = AgentAssistant(mock_manager, mock_director)
        result = assistant.assist_system("system_monitoring")
        
        assert "stability" in result
        assert "performance" in result
        assert "alerts" in result
        assert assistant.state["last_action"] == "system_monitoring"
        assert assistant.state["action_count"] == 1

    def test_assist_system_resource_optimization(self):
        """Test resource optimization assistance."""
        mock_manager = MagicMock(spec=SystemManager)
        mock_director = MagicMock(spec=SystemDirector)
        
        mock_manager.get_state.return_value = {
            "cpu_state": {"usage": 0.7, "temperature": 0.6},
            "memory_state": {"usage": 0.5, "fragmentation": 0.3},
            "energy_state": {"consumption": 0.4, "efficiency": 0.8},
            "network_state": {"bandwidth": 0.6, "latency": 0.3}
        }
        
        assistant = AgentAssistant(mock_manager, mock_director)
        result = assistant.assist_system("resource_optimization")
        
        assert "cpu_optimization" in result
        assert "memory_optimization" in result
        assert "energy_optimization" in result
        assert "network_optimization" in result
        assert assistant.state["last_action"] == "resource_optimization"

    def test_assist_system_task_automation(self):
        """Test task automation assistance."""
        mock_manager = MagicMock(spec=SystemManager)
        mock_director = MagicMock(spec=SystemDirector)
        
        assistant = AgentAssistant(mock_manager, mock_director)
        result = assistant.assist_system(
            "task_automation",
            {
                "type": "system_operation",
                "data": {"operation": "restart", "component": "quantum"}
            }
        )
        
        assert "success" in result
        assert assistant.state["last_action"] == "task_automation"

    def test_assist_system_user_assistance(self):
        """Test user assistance."""
        mock_manager = MagicMock(spec=SystemManager)
        mock_director = MagicMock(spec=SystemDirector)
        
        assistant = AgentAssistant(mock_manager, mock_director)
        result = assistant.assist_system(
            "user_assistance",
            {
                "type": "system_guidance",
                "data": {"query": "How to optimize system performance?"}
            }
        )
        
        assert "guidance" in result
        assert assistant.state["last_action"] == "user_assistance"

    def test_assist_system_error_handling(self):
        """Test error handling assistance."""
        mock_manager = MagicMock(spec=SystemManager)
        mock_director = MagicMock(spec=SystemDirector)
        
        assistant = AgentAssistant(mock_manager, mock_director)
        result = assistant.assist_system(
            "error_handling",
            {
                "type": "system_error",
                "data": {"error": "Component failure", "severity": "high"}
            }
        )
        
        assert "resolution" in result
        assert assistant.state["last_action"] == "error_handling"

    def test_assist_system_learning_adaptation(self):
        """Test learning adaptation assistance."""
        mock_manager = MagicMock(spec=SystemManager)
        mock_director = MagicMock(spec=SystemDirector)
        
        mock_manager.get_state.return_value = {
            "performance": 0.85
        }
        
        mock_director.get_state.return_value = {
            "overall_direction": 0.8
        }
        
        assistant = AgentAssistant(mock_manager, mock_director)
        result = assistant.assist_system("learning_adaptation")
        
        assert "learning_rate" in result
        assert "adaptation_level" in result
        assert assistant.state["last_action"] == "learning_adaptation"

    def test_assist_system_invalid_task(self):
        """Test assistance with invalid task."""
        mock_manager = MagicMock(spec=SystemManager)
        mock_director = MagicMock(spec=SystemDirector)
        
        assistant = AgentAssistant(mock_manager, mock_director)
        
        with pytest.raises(ValueError) as exc_info:
            assistant.assist_system("invalid_task")
        assert "Unsupported task" in str(exc_info.value)

    def test_get_state(self):
        """Test getting assistant state."""
        assistant = AgentAssistant()
        state = assistant.get_state()
        
        assert state["status"] == "active"
        assert state["action_count"] == 0
        assert state["error_count"] == 0

    def test_get_metrics(self):
        """Test getting assistant metrics."""
        assistant = AgentAssistant()
        metrics = assistant.get_metrics()
        
        assert metrics["response_time"] == 0.0
        assert metrics["accuracy"] == 0.0
        assert metrics["efficiency"] == 0.0
        assert metrics["user_satisfaction"] == 0.0
        assert metrics["system_impact"] == 0.0

    def test_reset(self):
        """Test resetting assistant state."""
        assistant = AgentAssistant()
        
        # Update state
        assistant.state.update({
            "last_action": "system_monitoring",
            "action_count": 5,
            "error_count": 2,
            "learning_rate": 0.2,
            "adaptation_level": 0.3
        })
        
        # Update metrics
        assistant.metrics.update({
            "response_time": 0.5,
            "accuracy": 0.9,
            "efficiency": 0.8,
            "user_satisfaction": 0.7,
            "system_impact": 0.6
        })
        
        assistant.reset()
        
        assert assistant.state["last_action"] is None
        assert assistant.state["action_count"] == 0
        assert assistant.state["error_count"] == 0
        assert assistant.state["learning_rate"] == 0.1
        assert assistant.state["adaptation_level"] == 0.0
        
        assert assistant.metrics["response_time"] == 0.0
        assert assistant.metrics["accuracy"] == 0.0
        assert assistant.metrics["efficiency"] == 0.0
        assert assistant.metrics["user_satisfaction"] == 0.0
        assert assistant.metrics["system_impact"] == 0.0 
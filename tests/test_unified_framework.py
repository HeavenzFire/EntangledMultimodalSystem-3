import pytest
import numpy as np
import torch
from src.core.unified_framework import UnifiedMetaphysicalFramework
from src.utils.errors import ModelError
from src.utils.logger import logger

class TestUnifiedMetaphysicalFramework:
    @pytest.fixture
    def framework(self):
        """Initialize the framework for testing."""
        return UnifiedMetaphysicalFramework()

    def test_initialization(self, framework):
        """Test framework initialization and component setup."""
        assert framework is not None
        assert framework.integration_engine is not None
        assert framework.revival_system is not None
        assert framework.multimodal_gan is not None
        assert framework.framework_state["current_goal"] == "Idle"
        assert framework.framework_state["operational_mode"] == "Standard"

    def test_goal_setting(self, framework):
        """Test setting different goals and operational modes."""
        # Test standard goal setting
        framework.set_goal("Test goal", "Standard")
        assert framework.framework_state["current_goal"] == "Test goal"
        assert framework.framework_state["operational_mode"] == "Standard"

        # Test revival mode
        framework.set_goal("Revival test", "Revival")
        assert framework.framework_state["operational_mode"] == "Revival"

        # Test creative mode
        framework.set_goal("Creative test", "Creative")
        assert framework.framework_state["operational_mode"] == "Creative"

        # Test expansion mode
        framework.set_goal("Expansion test", "Expansion")
        assert framework.framework_state["operational_mode"] == "Expansion"

    def test_orchestration_cycle_standard(self, framework):
        """Test standard orchestration cycle."""
        framework.set_goal("Standard cycle test", "Standard")
        result = framework.orchestrate_cycle(n_qubits=4, hologram_size=(64, 64))
        
        assert result is not None
        assert "framework_state" in result
        assert "integration_engine_status" in result
        assert "revival_system_status" in result
        assert "multimodal_gan_status" in result

    def test_orchestration_cycle_revival(self, framework):
        """Test revival mode orchestration cycle."""
        framework.set_goal("Revival cycle test", "Revival")
        result = framework.orchestrate_cycle(n_qubits=4, hologram_size=(64, 64))
        
        assert result is not None
        assert result["revival_system_status"]["progress"]["consciousness_level"] >= 0
        assert result["revival_system_status"]["progress"]["consciousness_level"] <= 1

    def test_orchestration_cycle_creative(self, framework):
        """Test creative mode orchestration cycle."""
        framework.set_goal("Creative cycle test", "Creative")
        input_seed = torch.randn(1, 3, 32, 32)  # Example input seed
        result = framework.orchestrate_cycle(n_qubits=4, hologram_size=(64, 64), input_seed=input_seed)
        
        assert result is not None
        assert "generated_content" in result["multimodal_gan_status"]

    def test_complexity_calculation(self, framework):
        """Test complexity score calculation."""
        # Run an orchestration cycle to update states
        framework.orchestrate_cycle(n_qubits=4, hologram_size=(64, 64))
        complexity = framework._calculate_complexity()
        
        assert isinstance(complexity, float)
        assert 0 <= complexity <= 1

    def test_error_handling(self, framework):
        """Test error handling in orchestration cycles."""
        # Test with invalid hologram size
        with pytest.raises(ModelError):
            framework.orchestrate_cycle(n_qubits=4, hologram_size=(0, 0))

        # Test with invalid input seed
        with pytest.raises(ModelError):
            framework.orchestrate_cycle(n_qubits=4, hologram_size=(64, 64), input_seed="invalid")

    def test_reset_framework(self, framework):
        """Test framework reset functionality."""
        # Set some state
        framework.set_goal("Reset test", "Standard")
        framework.orchestrate_cycle(n_qubits=4, hologram_size=(64, 64))
        
        # Reset and verify
        result = framework.reset_framework()
        assert result["message"] == "Unified Metaphysical Framework reset complete."
        assert framework.framework_state["current_goal"] == "Idle"
        assert framework.framework_state["operational_mode"] == "Standard"
        assert framework.framework_state["system_complexity_score"] == 0.0

    def test_interaction_logging(self, framework):
        """Test interaction logging functionality."""
        # Perform some actions
        framework.set_goal("Logging test", "Standard")
        framework.orchestrate_cycle(n_qubits=4, hologram_size=(64, 64))
        
        # Check logs
        log = framework.framework_state["component_interaction_log"]
        assert len(log) > 0
        assert all("timestamp" in entry for entry in log)
        assert all("interaction" in entry for entry in log)

    def test_performance_monitoring(self, framework):
        """Test performance monitoring during orchestration."""
        framework.set_goal("Performance test", "Standard")
        result = framework.orchestrate_cycle(n_qubits=4, hologram_size=(64, 64))
        
        assert "last_orchestration_cycle_time" in framework.framework_state
        assert framework.framework_state["last_orchestration_cycle_time"] > 0

    def test_status_retrieval(self, framework):
        """Test comprehensive status retrieval."""
        framework.set_goal("Status test", "Standard")
        framework.orchestrate_cycle(n_qubits=4, hologram_size=(64, 64))
        
        status = framework.get_framework_status()
        assert "framework_state" in status
        assert "integration_engine_status" in status
        assert "revival_system_status" in status
        assert "multimodal_gan_status" in status
        assert all(isinstance(v, (int, float, str, list, dict)) for v in status.values()) 
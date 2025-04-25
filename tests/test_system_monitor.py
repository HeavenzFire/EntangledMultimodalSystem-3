import os
import pytest
from unittest.mock import patch, MagicMock
from src.core.system_monitor import SystemMonitor
from src.core.system_manager import SystemManager
from src.core.agent_assistant import AgentAssistant
from src.utils.errors import ModelError

class TestSystemMonitor:
    def test_initialization(self):
        """Test successful initialization with default components."""
        with patch.dict(os.environ, {
            "MONITORING_INTERVAL": "0.1",
            "MONITORING_HISTORY_LENGTH": "1000",
            "CPU_USAGE_THRESHOLD": "0.8",
            "MEMORY_USAGE_THRESHOLD": "0.8",
            "ENERGY_CONSUMPTION_THRESHOLD": "0.8",
            "NETWORK_LATENCY_THRESHOLD": "0.8",
            "RESPONSE_TIME_THRESHOLD": "0.5"
        }):
            monitor = SystemMonitor()
            assert monitor.state["status"] == "active"
            assert monitor.state["monitoring_count"] == 0
            assert monitor.state["alert_count"] == 0
            assert monitor.state["diagnostic_count"] == 0
            assert monitor.monitoring_interval == 0.1
            assert monitor.history_length == 1000

    def test_initialization_with_custom_components(self):
        """Test initialization with custom SystemManager and AgentAssistant."""
        mock_manager = MagicMock(spec=SystemManager)
        mock_assistant = MagicMock(spec=AgentAssistant)
        
        monitor = SystemMonitor(
            system_manager=mock_manager,
            agent_assistant=mock_assistant
        )
        
        assert monitor.system_manager == mock_manager
        assert monitor.agent_assistant == mock_assistant

    def test_monitor_system(self):
        """Test system monitoring functionality."""
        mock_manager = MagicMock(spec=SystemManager)
        mock_assistant = MagicMock(spec=AgentAssistant)
        
        # Mock system state
        mock_manager.get_state.return_value = {
            "cpu_state": {
                "usage": 0.7,
                "temperature": 0.6,
                "frequency": 0.8
            },
            "memory_state": {
                "usage": 0.5,
                "fragmentation": 0.3,
                "swap_usage": 0.2
            },
            "energy_state": {
                "consumption": 0.4,
                "efficiency": 0.8,
                "temperature": 0.5
            },
            "network_state": {
                "bandwidth": 0.6,
                "latency": 0.3,
                "packet_loss": 0.1
            },
            "performance_state": {
                "response_time": 0.4,
                "throughput": 0.7,
                "error_rate": 0.2
            }
        }
        
        monitor = SystemMonitor(mock_manager, mock_assistant)
        result = monitor.monitor_system()
        
        assert "metrics" in result
        assert "alerts" in result
        assert "response_time" in result
        assert monitor.state["monitoring_count"] == 1
        assert monitor.state["last_monitoring"] is not None

    def test_monitor_system_with_threshold_exceeded(self):
        """Test system monitoring with exceeded thresholds."""
        mock_manager = MagicMock(spec=SystemManager)
        mock_assistant = MagicMock(spec=AgentAssistant)
        
        # Mock system state with high values
        mock_manager.get_state.return_value = {
            "cpu_state": {"usage": 0.9},
            "memory_state": {"usage": 0.9},
            "energy_state": {"consumption": 0.9},
            "network_state": {"latency": 0.9},
            "performance_state": {"response_time": 0.6}
        }
        
        monitor = SystemMonitor(mock_manager, mock_assistant)
        result = monitor.monitor_system()
        
        assert len(result["alerts"]) == 5  # All thresholds exceeded
        assert monitor.state["alert_count"] == 5

    def test_monitor_cpu(self):
        """Test CPU monitoring."""
        mock_manager = MagicMock(spec=SystemManager)
        mock_manager.get_state.return_value = {
            "cpu_state": {
                "usage": 0.7,
                "temperature": 0.6,
                "frequency": 0.8
            }
        }
        
        monitor = SystemMonitor(mock_manager)
        cpu_metrics = monitor._monitor_cpu(mock_manager.get_state())
        
        assert cpu_metrics["usage"] == 0.7
        assert cpu_metrics["temperature"] == 0.6
        assert cpu_metrics["frequency"] == 0.8

    def test_monitor_memory(self):
        """Test memory monitoring."""
        mock_manager = MagicMock(spec=SystemManager)
        mock_manager.get_state.return_value = {
            "memory_state": {
                "usage": 0.5,
                "fragmentation": 0.3,
                "swap_usage": 0.2
            }
        }
        
        monitor = SystemMonitor(mock_manager)
        memory_metrics = monitor._monitor_memory(mock_manager.get_state())
        
        assert memory_metrics["usage"] == 0.5
        assert memory_metrics["fragmentation"] == 0.3
        assert memory_metrics["swap_usage"] == 0.2

    def test_monitor_energy(self):
        """Test energy monitoring."""
        mock_manager = MagicMock(spec=SystemManager)
        mock_manager.get_state.return_value = {
            "energy_state": {
                "consumption": 0.4,
                "efficiency": 0.8,
                "temperature": 0.5
            }
        }
        
        monitor = SystemMonitor(mock_manager)
        energy_metrics = monitor._monitor_energy(mock_manager.get_state())
        
        assert energy_metrics["consumption"] == 0.4
        assert energy_metrics["efficiency"] == 0.8
        assert energy_metrics["temperature"] == 0.5

    def test_monitor_network(self):
        """Test network monitoring."""
        mock_manager = MagicMock(spec=SystemManager)
        mock_manager.get_state.return_value = {
            "network_state": {
                "bandwidth": 0.6,
                "latency": 0.3,
                "packet_loss": 0.1
            }
        }
        
        monitor = SystemMonitor(mock_manager)
        network_metrics = monitor._monitor_network(mock_manager.get_state())
        
        assert network_metrics["bandwidth"] == 0.6
        assert network_metrics["latency"] == 0.3
        assert network_metrics["packet_loss"] == 0.1

    def test_monitor_performance(self):
        """Test performance monitoring."""
        mock_manager = MagicMock(spec=SystemManager)
        mock_manager.get_state.return_value = {
            "performance_state": {
                "response_time": 0.4,
                "throughput": 0.7,
                "error_rate": 0.2
            }
        }
        
        monitor = SystemMonitor(mock_manager)
        performance_metrics = monitor._monitor_performance(mock_manager.get_state())
        
        assert performance_metrics["response_time"] == 0.4
        assert performance_metrics["throughput"] == 0.7
        assert performance_metrics["error_rate"] == 0.2

    def test_generate_alerts(self):
        """Test alert generation."""
        mock_manager = MagicMock(spec=SystemManager)
        monitor = SystemMonitor(mock_manager)
        
        # Set metrics to exceed thresholds
        monitor.metrics.update({
            "cpu_metrics": {"usage": 0.9},
            "memory_metrics": {"usage": 0.9},
            "energy_metrics": {"consumption": 0.9},
            "network_metrics": {"latency": 0.9},
            "performance_metrics": {"response_time": 0.6}
        })
        
        alerts = monitor._generate_alerts()
        
        assert len(alerts) == 5
        assert all(alert["severity"] == "high" for alert in alerts)
        assert monitor.state["alert_count"] == 5

    def test_update_diagnostic_history(self):
        """Test diagnostic history updates."""
        mock_manager = MagicMock(spec=SystemManager)
        monitor = SystemMonitor(mock_manager)
        
        diagnostic_data = {
            "timestamp": 1234567890,
            "metrics": {"test": 0.5},
            "alerts": []
        }
        
        monitor._update_diagnostic_history(diagnostic_data)
        
        assert len(monitor.diagnostic_history) == 1
        assert monitor.state["diagnostic_count"] == 1
        assert monitor.diagnostic_history[0] == diagnostic_data

    def test_diagnostic_history_length_limit(self):
        """Test diagnostic history length limit."""
        mock_manager = MagicMock(spec=SystemManager)
        monitor = SystemMonitor(mock_manager)
        
        # Add more entries than history length
        for i in range(monitor.history_length + 10):
            monitor._update_diagnostic_history({
                "timestamp": i,
                "metrics": {"test": 0.5},
                "alerts": []
            })
        
        assert len(monitor.diagnostic_history) == monitor.history_length
        assert monitor.state["diagnostic_count"] == monitor.history_length

    def test_get_state(self):
        """Test getting monitor state."""
        monitor = SystemMonitor()
        state = monitor.get_state()
        
        assert state["status"] == "active"
        assert state["monitoring_count"] == 0
        assert state["alert_count"] == 0
        assert state["diagnostic_count"] == 0

    def test_get_metrics(self):
        """Test getting monitor metrics."""
        monitor = SystemMonitor()
        metrics = monitor.get_metrics()
        
        assert "cpu_metrics" in metrics
        assert "memory_metrics" in metrics
        assert "energy_metrics" in metrics
        assert "network_metrics" in metrics
        assert "performance_metrics" in metrics

    def test_get_diagnostic_history(self):
        """Test getting diagnostic history."""
        monitor = SystemMonitor()
        history = monitor.get_diagnostic_history()
        
        assert isinstance(history, list)
        assert len(history) == 0

    def test_reset(self):
        """Test resetting monitor state."""
        monitor = SystemMonitor()
        
        # Update state
        monitor.state.update({
            "status": "inactive",
            "last_monitoring": 1234567890,
            "monitoring_count": 5,
            "alert_count": 2,
            "diagnostic_count": 3
        })
        
        # Update metrics
        monitor.metrics.update({
            "cpu_metrics": {"usage": 0.9},
            "memory_metrics": {"usage": 0.9},
            "energy_metrics": {"consumption": 0.9},
            "network_metrics": {"latency": 0.9},
            "performance_metrics": {"response_time": 0.6}
        })
        
        # Add diagnostic history
        monitor.diagnostic_history = [{"test": "data"}]
        
        monitor.reset()
        
        assert monitor.state["status"] == "active"
        assert monitor.state["last_monitoring"] is None
        assert monitor.state["monitoring_count"] == 0
        assert monitor.state["alert_count"] == 0
        assert monitor.state["diagnostic_count"] == 0
        
        assert monitor.metrics["cpu_metrics"]["usage"] == 0.0
        assert monitor.metrics["memory_metrics"]["usage"] == 0.0
        assert monitor.metrics["energy_metrics"]["consumption"] == 0.0
        assert monitor.metrics["network_metrics"]["latency"] == 0.0
        assert monitor.metrics["performance_metrics"]["response_time"] == 0.0
        
        assert len(monitor.diagnostic_history) == 0 
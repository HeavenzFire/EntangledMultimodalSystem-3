import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from src.core.digigod_nexus import DigigodNexus
from src.utils.errors import ModelError

class TestDigigodNexusIntegration:
    """Test suite for DigigodNexus integration features."""
    
    @pytest.fixture
    def digigod_nexus(self):
        """Create a DigigodNexus instance for testing."""
        with patch('src.core.quantum_processor.QuantumProcessor') as mock_quantum, \
             patch('src.core.holographic_processor.HolographicProcessor') as mock_holographic, \
             patch('src.core.neural_interface.NeuralInterface') as mock_neural, \
             patch('src.core.consciousness_matrix.ConsciousnessMatrix') as mock_consciousness, \
             patch('src.core.system_validator.SystemValidator') as mock_validator, \
             patch('src.core.system_monitor.SystemMonitor') as mock_monitor:
            
            # Setup mock returns
            mock_quantum.return_value.process.return_value = {
                "result": {"counts": {"000": 50, "111": 50}},
                "metrics": {"fidelity": 0.95}
            }
            mock_holographic.return_value.process.return_value = {
                "result": {"hologram": np.zeros((1024, 1024))},
                "metrics": {"resolution": 1024}
            }
            mock_neural.return_value.process_neural_data.return_value = {
                "output": np.array([0.5, 0.5]),
                "metrics": {"accuracy": 0.98}
            }
            mock_consciousness.return_value.process_consciousness.return_value = {
                "result": {"state": np.array([0.5, 0.5])},
                "metrics": {"consciousness_level": 0.85}
            }
            mock_validator.return_value.validate_input.return_value = {
                "status": "pass",
                "details": "Input validated successfully"
            }
            mock_validator.return_value.validate_system.return_value = {
                "status": "pass",
                "details": "System validated successfully"
            }
            mock_monitor.return_value.get_metrics.return_value = {
                "error_rate": 0.01,
                "health_score": 0.95
            }
            
            return DigigodNexus()
    
    def test_process_task_integration(self, digigod_nexus):
        """Test full task processing integration."""
        input_data = {
            "task_type": "optimization",
            "quantum_data": {"theta": 0.5},
            "holographic_data": {"resolution": 1024},
            "neural_data": {"input": np.array([0.5, 0.5])},
            "context": {"task_id": "test_001"},
            "attention_depth": 5,
            "consciousness_threshold": 0.7
        }
        
        result = digigod_nexus.process_task(input_data)
        
        assert "output" in result
        assert "system_state" in result
        assert "consciousness_metrics" in result
        assert "validation_report" in result
        assert "processing_metrics" in result
        assert result["validation_report"]["status"] == "pass"
        assert result["consciousness_metrics"]["consciousness_level"] == 0.85
    
    def test_error_handling(self, digigod_nexus):
        """Test error handling and recovery."""
        # Test input validation failure
        with patch.object(digigod_nexus.validator, 'validate_input') as mock_validate:
            mock_validate.return_value = {
                "status": "fail",
                "details": "Invalid input format"
            }
            result = digigod_nexus.process_task({})
            assert "error" in result
            assert result["validation_report"]["status"] == "fail"
        
        # Test system validation failure
        with patch.object(digigod_nexus.validator, 'validate_system') as mock_validate:
            mock_validate.return_value = {
                "status": "fail",
                "details": "System integrity compromised"
            }
            result = digigod_nexus.process_task({"task_type": "test"})
            assert result["validation_report"]["status"] == "fail"
    
    @pytest.mark.benchmark
    def test_performance_benchmark(self, digigod_nexus, benchmark):
        """Test system performance under load."""
        input_data = {
            "task_type": "benchmark",
            "quantum_data": {"theta": 0.5},
            "holographic_data": {"resolution": 1024},
            "neural_data": {"input": np.array([0.5, 0.5])}
        }
        
        def process():
            return digigod_nexus.process_task(input_data)
        
        result = benchmark(process)
        assert result["processing_metrics"]["processing_time"] < 1.0  # Should complete within 1 second
    
    def test_auto_repair_trigger(self, digigod_nexus):
        """Test automatic repair trigger on high error rate."""
        with patch.object(digigod_nexus.monitor, 'get_metrics') as mock_metrics:
            mock_metrics.return_value = {"error_rate": 0.06}  # Above threshold
            with patch.object(digigod_nexus, 'reset') as mock_reset:
                digigod_nexus.process_task({"task_type": "test"})
                mock_reset.assert_called_once()
    
    def test_consciousness_integration(self, digigod_nexus):
        """Test consciousness matrix integration."""
        input_data = {
            "task_type": "consciousness_test",
            "quantum_data": {"theta": 0.5},
            "holographic_data": {"resolution": 1024},
            "neural_data": {"input": np.array([0.5, 0.5])},
            "consciousness_threshold": 0.8
        }
        
        result = digigod_nexus.process_task(input_data)
        
        assert "consciousness_metrics" in result
        assert result["consciousness_metrics"]["consciousness_level"] > 0.0
        assert result["system_state"]["consciousness_level"] > 0.0
    
    def test_metrics_tracking(self, digigod_nexus):
        """Test comprehensive metrics tracking."""
        input_data = {
            "task_type": "metrics_test",
            "quantum_data": {"theta": 0.5},
            "holographic_data": {"resolution": 1024},
            "neural_data": {"input": np.array([0.5, 0.5])}
        }
        
        result = digigod_nexus.process_task(input_data)
        
        assert "processing_metrics" in result
        metrics = result["processing_metrics"]
        assert "processing_time" in metrics
        assert "error_rate" in metrics
        assert "health_score" in metrics
        assert metrics["processing_time"] > 0
        assert metrics["error_rate"] < 0.1
        assert metrics["health_score"] > 0.5 
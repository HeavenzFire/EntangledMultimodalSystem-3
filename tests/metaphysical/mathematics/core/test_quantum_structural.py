import pytest
import numpy as np
from src.metaphysical.mathematics.core.quantum_structural import QuantumStructuralSystem

class TestQuantumStructuralSystem:
    @pytest.fixture
    def system(self):
        return QuantumStructuralSystem(num_dimensions=6)
        
    def test_initialization(self, system):
        """Test system initialization"""
        assert system.state.einstein_tensor.shape == (6, 6)
        assert system.state.calabi_yau_metric.shape == (6, 6)
        assert system.state.singularity_resolution.shape == (6, 6)
        assert system.state.coherence_level == 0.0
        assert system.state.system_status == 'initialized'
        
    def test_calculate_einstein_tensor(self, system):
        """Test Einstein tensor calculation"""
        metric = np.random.rand(6, 6)
        einstein_tensor = system.calculate_einstein_tensor(metric)
        
        assert einstein_tensor.shape == (6, 6)
        assert not np.isnan(einstein_tensor).any()
        assert not np.isinf(einstein_tensor).any()
        
    def test_calculate_singularity_resolution(self, system):
        """Test singularity resolution calculation"""
        metric = np.random.rand(6, 6)
        resolution = system._calculate_singularity_resolution(metric)
        
        assert resolution.shape == (6, 6)
        assert not np.isnan(resolution).any()
        assert not np.isinf(resolution).any()
        assert np.all(resolution >= 0)
        assert np.all(resolution <= 1)
        
    def test_calculate_calabi_yau_metric(self, system):
        """Test Calabi-Yau metric calculation"""
        metric = system.calculate_calabi_yau_metric()
        
        assert metric.shape == (6, 6)
        assert not np.isnan(metric).any()
        assert not np.isinf(metric).any()
        assert np.all(metric >= 0)
        assert np.all(metric <= 1)
        
    def test_calculate_coherence(self, system):
        """Test coherence calculation"""
        einstein_tensor = np.random.rand(6, 6)
        calabi_yau_metric = np.random.rand(6, 6)
        coherence = system.calculate_coherence(einstein_tensor, calabi_yau_metric)
        
        assert isinstance(coherence, float)
        assert not np.isnan(coherence)
        assert not np.isinf(coherence)
        assert 0 <= coherence <= 1
        
    def test_process_structural_integrity(self, system):
        """Test structural integrity processing"""
        result = system.process_structural_integrity()
        
        assert 'einstein_tensor_shape' in result
        assert 'calabi_yau_metric_shape' in result
        assert 'coherence_level' in result
        assert 'system_status' in result
        assert result['system_status'] == 'processed'
        
    def test_get_structural_report(self, system):
        """Test structural report generation"""
        report = system.get_structural_report()
        
        assert 'timestamp' in report
        assert 'einstein_tensor_shape' in report
        assert 'calabi_yau_metric_shape' in report
        assert 'singularity_resolution_shape' in report
        assert 'coherence_level' in report
        assert 'last_update' in report
        assert 'system_status' in report 
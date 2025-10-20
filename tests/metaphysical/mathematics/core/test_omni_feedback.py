import pytest
import numpy as np
from src.metaphysical.mathematics.core.omni_feedback import OmniFeedbackSystem

class TestOmniFeedbackSystem:
    @pytest.fixture
    def system(self):
        return OmniFeedbackSystem(num_particles=1000)
        
    @pytest.fixture
    def particles(self):
        return np.random.rand(1000, 1000)
        
    def test_initialization(self, system):
        """Test system initialization"""
        assert system.state.collision_matrix.shape == (1000, 1000)
        assert system.state.detector_response.shape == (1000, 1000)
        assert system.state.expansion_factor == 1.0
        assert system.state.optimization_score == 0.0
        assert system.state.system_status == 'initialized'
        
    def test_simulate_lhc_collision(self, system, particles):
        """Test LHC collision simulation"""
        collision_matrix = system.simulate_lhc_collision(particles)
        
        assert collision_matrix.shape == (1000, 1000)
        assert not np.isnan(collision_matrix).any()
        assert not np.isinf(collision_matrix).any()
        assert np.all(collision_matrix >= 0)
        assert np.all(collision_matrix <= 1)
        
    def test_simulate_detector_response(self, system, particles):
        """Test detector response simulation"""
        collision_matrix = system.simulate_lhc_collision(particles)
        detector_response = system.simulate_detector_response(collision_matrix)
        
        assert detector_response.shape == (1000, 1000)
        assert not np.isnan(detector_response).any()
        assert not np.isinf(detector_response).any()
        assert np.all(detector_response >= 0)
        assert np.all(detector_response <= 1)
        
    def test_calculate_expansion_factor(self, system, particles):
        """Test expansion factor calculation"""
        collision_matrix = system.simulate_lhc_collision(particles)
        detector_response = system.simulate_detector_response(collision_matrix)
        expansion_factor = system.calculate_expansion_factor(collision_matrix, detector_response)
        
        assert isinstance(expansion_factor, float)
        assert not np.isnan(expansion_factor)
        assert not np.isinf(expansion_factor)
        assert expansion_factor > 0
        
    def test_optimize_expansion(self, system):
        """Test expansion optimization"""
        expansion_factor = 0.5
        optimization_score = system.optimize_expansion(expansion_factor)
        
        assert isinstance(optimization_score, float)
        assert not np.isnan(optimization_score)
        assert not np.isinf(optimization_score)
        
    def test_process_feedback(self, system, particles):
        """Test feedback processing"""
        result = system.process_feedback(particles)
        
        assert 'collision_matrix_shape' in result
        assert 'detector_response_shape' in result
        assert 'expansion_factor' in result
        assert 'optimization_score' in result
        assert 'system_status' in result
        assert result['system_status'] == 'processed'
        
    def test_get_feedback_report(self, system):
        """Test feedback report generation"""
        report = system.get_feedback_report()
        
        assert 'timestamp' in report
        assert 'collision_matrix_shape' in report
        assert 'detector_response_shape' in report
        assert 'expansion_factor' in report
        assert 'optimization_score' in report
        assert 'last_update' in report
        assert 'system_status' in report 
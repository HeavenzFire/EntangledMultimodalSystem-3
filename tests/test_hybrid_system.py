import pytest
from datetime import datetime
import numpy as np
from src.quantum.core.hybrid_system import QuantumHybridSystem, PerformanceDashboard, ResourceOptimizer
from src.quantum.core.ml_optimization import MLModelOptimizer, ModelMetrics
from src.quantum.core.adaptive_optimization import QuantumInspiredOptimizer, OptimizationState

@pytest.fixture
def hybrid_system():
    return QuantumHybridSystem()

@pytest.fixture
def performance_dashboard():
    return PerformanceDashboard()

@pytest.fixture
def resource_optimizer():
    return ResourceOptimizer()

@pytest.fixture
def ml_optimizer():
    return MLModelOptimizer()

@pytest.fixture
def quantum_optimizer():
    return QuantumInspiredOptimizer()

def test_hybrid_system_initialization(hybrid_system):
    """Test initialization of QuantumHybridSystem"""
    assert hybrid_system.error_correction is not None
    assert hybrid_system.decoder is not None
    assert hybrid_system.system_control is not None
    assert hybrid_system.performance_dashboard is not None
    assert hybrid_system.resource_optimizer is not None
    assert len(hybrid_system.job_history) == 0
    assert len(hybrid_system.error_history) == 0

def test_process_job(hybrid_system):
    """Test job processing functionality"""
    job_id = "test_job_1"
    circuit = "H 0; CNOT 0 1; MEASURE 0"
    
    result = hybrid_system.process_job(job_id, circuit)
    
    assert result['job_id'] == job_id
    assert 'execution_time' in result
    assert 'error_rate' in result
    assert 'ml_accuracy' in result
    assert 'response_time' in result
    assert len(hybrid_system.job_history) == 1

def test_performance_dashboard(performance_dashboard):
    """Test performance dashboard functionality"""
    metrics = {
        'response_time': 400.0,
        'error_rate': 0.01,
        'ml_accuracy': 0.95
    }
    
    performance_dashboard.update_metrics(metrics)
    report = performance_dashboard.generate_report()
    
    assert report['avg_response'] == 400.0
    assert report['error_rate'] == 0.01
    assert report['ml_accuracy'] == 0.95

def test_resource_optimization(resource_optimizer):
    """Test resource optimization functionality"""
    # Test high load scenario
    for _ in range(6):
        resource_optimizer.optimize_resources(80.0)
    assert resource_optimizer.response_time < 470.0  # Should decrease
    
    # Test low load scenario
    for _ in range(6):
        resource_optimizer.optimize_resources(20.0)
    assert resource_optimizer.response_time > 400.0  # Should increase

def test_system_metrics(hybrid_system):
    """Test system metrics collection"""
    # Process some jobs
    for i in range(3):
        hybrid_system.process_job(f"test_job_{i}", "H 0; MEASURE 0")
    
    metrics = hybrid_system.get_system_metrics()
    
    assert 'dashboard' in metrics
    assert 'error_correction' in metrics
    assert 'decoder' in metrics
    assert 'resource_optimization' in metrics
    assert metrics['dashboard']['total_jobs'] == 3

def test_anomaly_detection(hybrid_system):
    """Test anomaly detection and alerting"""
    # Process normal job
    hybrid_system.process_job("normal_job", "H 0; MEASURE 0")
    assert len(hybrid_system.performance_dashboard.alerts) == 0
    
    # Process job with potential anomaly
    hybrid_system.error_correction.logical_error_rate = 0.1  # Simulate high error rate
    hybrid_system.process_job("anomaly_job", "H 0; MEASURE 0")
    assert len(hybrid_system.performance_dashboard.alerts) > 0
    assert "Anomaly" in hybrid_system.performance_dashboard.alerts[-1]['message']

def test_ml_optimization(ml_optimizer):
    """Test ML optimization functionality"""
    # Test error prediction
    current_metrics = {'error_rate': 0.01, 'ml_accuracy': 0.95}
    prediction = ml_optimizer.predict_errors(current_metrics)
    assert isinstance(prediction, float)
    assert 0 <= prediction <= 1
    
    # Test model optimization
    training_data = [{'error_rate': 0.01, 'ml_accuracy': 0.95} for _ in range(10)]
    metrics = ml_optimizer.optimize_model(training_data)
    assert isinstance(metrics, ModelMetrics)
    assert 0 <= metrics.accuracy <= 1
    assert 0 <= metrics.loss <= 1
    assert metrics.inference_time >= 0
    assert metrics.training_iterations > 0

def test_ml_anomaly_detection(ml_optimizer):
    """Test ML-based anomaly detection"""
    # Add normal error predictions
    for i in range(10):
        ml_optimizer.track_prediction(f"job_{i}", 0.01, 0.01)
    assert not ml_optimizer.detect_anomalies()
    
    # Add anomalous error prediction
    ml_optimizer.track_prediction("anomaly_job", 0.01, 0.1)
    assert ml_optimizer.detect_anomalies()

def test_ml_performance_metrics(ml_optimizer):
    """Test ML performance metrics collection"""
    # Test initial metrics
    initial_metrics = ml_optimizer.get_performance_metrics()
    assert initial_metrics['accuracy'] == 0.0
    assert initial_metrics['loss'] == 1.0
    
    # Add some training data
    training_data = [{'error_rate': 0.01, 'ml_accuracy': 0.95} for _ in range(5)]
    ml_optimizer.optimize_model(training_data)
    
    # Test updated metrics
    updated_metrics = ml_optimizer.get_performance_metrics()
    assert updated_metrics['accuracy'] > 0
    assert updated_metrics['loss'] < 1
    assert updated_metrics['training_iterations'] > 0
    assert 'learning_rate' in updated_metrics
    assert 'model_weights' in updated_metrics

def test_hybrid_system_ml_integration(hybrid_system):
    """Test ML optimization integration in hybrid system"""
    # Process multiple jobs
    for i in range(5):
        hybrid_system.process_job(f"ml_test_{i}", "H 0; MEASURE 0")
    
    # Check ML metrics in system metrics
    metrics = hybrid_system.get_system_metrics()
    assert 'ml_optimization' in metrics
    ml_metrics = metrics['ml_optimization']
    assert 'accuracy' in ml_metrics
    assert 'loss' in ml_metrics
    assert 'training_iterations' in ml_metrics
    
    # Check ML predictions in job results
    job_metrics = hybrid_system.job_history[-1]['metrics']
    assert 'predicted_error' in job_metrics
    assert 'ml_optimization' in job_metrics
    assert isinstance(job_metrics['ml_optimization'], dict)

def test_quantum_optimization(quantum_optimizer):
    """Test quantum-inspired optimization functionality"""
    # Test quantum state optimization
    current_state = {
        'quantum_fidelity': 0.9,
        'error_rate': 0.02
    }
    optimized_state = quantum_optimizer.optimize_quantum_state(current_state)
    assert isinstance(optimized_state, dict)
    assert 0.8 <= optimized_state['quantum_fidelity'] <= 1.0
    assert 0.0 <= optimized_state['error_rate'] <= 0.1
    
    # Test resource optimization
    historical_data = [{
        'quantum_fidelity': 0.9,
        'classical_performance': 0.8,
        'resource_utilization': 0.7
    } for _ in range(10)]
    resource_allocation = quantum_optimizer.optimize_resources(0.7, historical_data)
    assert isinstance(resource_allocation, dict)
    assert all(0 <= v <= 1 for v in resource_allocation.values())
    
    # Test parameter adaptation
    adapted_state = quantum_optimizer.adapt_parameters(current_state)
    assert isinstance(adapted_state, dict)
    assert 'quantum_fidelity' in adapted_state
    assert 'error_rate' in adapted_state
    assert 'adaptation_rate' in adapted_state

def test_quantum_performance_tracking(quantum_optimizer):
    """Test quantum performance tracking"""
    # Track performance
    state = {
        'quantum_fidelity': 0.95,
        'classical_performance': 0.9,
        'resource_utilization': 0.8,
        'error_rate': 0.01
    }
    quantum_optimizer.track_performance(state)
    
    # Check optimization history
    assert len(quantum_optimizer.optimization_history) == 1
    assert isinstance(quantum_optimizer.optimization_history[0], OptimizationState)
    assert quantum_optimizer.optimization_history[0].quantum_fidelity == 0.95

def test_quantum_metrics(quantum_optimizer):
    """Test quantum optimization metrics"""
    # Test initial metrics
    initial_metrics = quantum_optimizer.get_optimization_metrics()
    assert initial_metrics['quantum_fidelity'] == 0.0
    assert initial_metrics['error_rate'] == 1.0
    
    # Add some optimization history
    state = {
        'quantum_fidelity': 0.95,
        'classical_performance': 0.9,
        'resource_utilization': 0.8,
        'error_rate': 0.01
    }
    quantum_optimizer.track_performance(state)
    
    # Test updated metrics
    updated_metrics = quantum_optimizer.get_optimization_metrics()
    assert updated_metrics['quantum_fidelity'] == 0.95
    assert updated_metrics['error_rate'] == 0.01
    assert 'adaptation_rate' in updated_metrics
    assert 'history_length' in updated_metrics

def test_hybrid_system_quantum_integration(hybrid_system):
    """Test quantum optimization integration in hybrid system"""
    # Process multiple jobs
    for i in range(5):
        hybrid_system.process_job(f"quantum_test_{i}", "H 0; MEASURE 0")
    
    # Check quantum metrics in system metrics
    metrics = hybrid_system.get_system_metrics()
    assert 'quantum_optimization' in metrics
    quantum_metrics = metrics['quantum_optimization']
    assert 'quantum_fidelity' in quantum_metrics
    assert 'classical_performance' in quantum_metrics
    assert 'resource_utilization' in quantum_metrics
    
    # Check quantum optimization in job results
    job_metrics = hybrid_system.job_history[-1]['metrics']
    assert 'quantum_optimization' in job_metrics
    quantum_opt = job_metrics['quantum_optimization']
    assert 'fidelity' in quantum_opt
    assert 'adapted_fidelity' in quantum_opt
    assert 'resource_allocation' in quantum_opt 
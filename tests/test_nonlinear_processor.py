import pytest
import numpy as np
from datetime import datetime
from src.quantum.core.nonlinear_processor import (
    NonlinearProcessor,
    CarlemanLinearizer,
    NonlinearState
)
from src.quantum.core.quantum_state import QuantumState

@pytest.fixture
def quantum_state():
    return QuantumState(
        amplitude=1.0,
        phase=np.pi/4,
        error_rate=0.01
    )

@pytest.fixture
def nonlinear_processor():
    return NonlinearProcessor()

@pytest.fixture
def carleman_linearizer():
    return CarlemanLinearizer(max_order=3)

def test_carleman_linearization(carleman_linearizer):
    # Create a simple nonlinear system
    nonlinear_system = np.array([
        [1.0, 0.5],
        [0.5, 1.0]
    ])
    
    # Test linearization
    linear_system, error = carleman_linearizer.linearize(nonlinear_system)
    
    # Verify results
    assert linear_system.shape == (6, 6)  # 2 * max_order
    assert isinstance(error, float)
    assert error >= 0.0
    assert len(carleman_linearizer.linearization_history) == 1

def test_nonlinear_processing(nonlinear_processor, quantum_state):
    # Process quantum state
    result = nonlinear_processor.process_quantum_state(quantum_state)
    
    # Verify result type and properties
    assert isinstance(result, NonlinearState)
    assert 0 <= result.amplitude <= 2.0
    assert 0 <= result.phase <= 2 * np.pi
    assert 0 <= result.coherence <= 1.0
    assert 0 <= result.error_rate <= 1.0
    assert isinstance(result.timestamp, datetime)
    
    # Verify state history
    assert len(nonlinear_processor.state_history) == 1

def test_parameter_optimization(nonlinear_processor, quantum_state):
    # Optimize parameters
    result = nonlinear_processor.optimize_parameters(quantum_state)
    
    # Verify optimization results
    assert isinstance(result, dict)
    assert 'optimal_amplitude' in result
    assert 'optimal_phase' in result
    assert 'final_error' in result
    assert 'success' in result
    assert isinstance(result['timestamp'], datetime)
    
    # Verify optimization history
    assert len(nonlinear_processor.optimization_history) == 1

def test_performance_metrics(nonlinear_processor, quantum_state):
    # Process a state first
    nonlinear_processor.process_quantum_state(quantum_state)
    
    # Get performance metrics
    metrics = nonlinear_processor.get_performance_metrics()
    
    # Verify metrics
    assert isinstance(metrics, dict)
    assert 'current_amplitude' in metrics
    assert 'current_phase' in metrics
    assert 'coherence' in metrics
    assert 'error_rate' in metrics
    assert 'truncation_error' in metrics

def test_nonlinear_transform(nonlinear_processor):
    # Test amplitude transformation
    amplitude = 1.0
    phase = np.pi/4
    
    transformed = nonlinear_processor._apply_nonlinear_transform(amplitude, phase)
    
    assert isinstance(transformed, float)
    assert transformed >= 0.0

def test_phase_shift_calculation(nonlinear_processor):
    # Test phase shift calculation
    phase = np.pi/4
    coherence = 0.9
    
    shifted_phase = nonlinear_processor._calculate_phase_shift(phase, coherence)
    
    assert isinstance(shifted_phase, float)
    assert 0 <= shifted_phase <= 2 * np.pi

def test_error_rate_calculation(nonlinear_processor):
    # Test error rate calculation
    amplitude = 1.0
    coherence = 0.9
    
    error_rate = nonlinear_processor._calculate_error_rate(amplitude, coherence)
    
    assert isinstance(error_rate, float)
    assert 0 <= error_rate <= 1.0 
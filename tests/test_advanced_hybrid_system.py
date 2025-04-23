import pytest
import numpy as np
from datetime import datetime
from src.quantum.core.advanced_hybrid_system import (
    AdvancedHybridSystem,
    QuantumState,
    ClassicalState,
    LayeredArchitecture,
    QuantumClassicalInterface
)
from src.quantum.core.hybrid_algorithms import (
    HybridOptimizationFactory,
    HybridQuantumEigensolver,
    QuantumVariationalOptimizer
)

@pytest.fixture
def hybrid_system():
    return AdvancedHybridSystem()

@pytest.fixture
def quantum_state():
    return QuantumState(
        fidelity=0.99,
        error_rate=0.001,
        coherence_time=100.0,
        entanglement_degree=0.95,
        timestamp=datetime.now()
    )

@pytest.fixture
def classical_state():
    return ClassicalState(
        processing_efficiency=0.95,
        memory_utilization=0.80,
        communication_latency=5.0,
        computation_accuracy=0.98,
        timestamp=datetime.now()
    )

def test_layered_architecture_initialization():
    """Test initialization of layered architecture"""
    architecture = LayeredArchitecture()
    assert hasattr(architecture, 'quantum_processor')
    assert hasattr(architecture, 'classical_processor')
    assert hasattr(architecture, 'quantum_classical_interface')
    assert hasattr(architecture, 'quantum_optimizer')
    assert hasattr(architecture, 'ml_optimizer')
    assert hasattr(architecture, 'omni_framework')

def test_quantum_classical_interface(quantum_state, classical_state):
    """Test quantum-classical state conversion"""
    interface = QuantumClassicalInterface()
    
    # Test quantum to classical conversion
    converted_classical = interface.quantum_to_classical(quantum_state)
    assert isinstance(converted_classical, ClassicalState)
    assert converted_classical.computation_accuracy == pytest.approx(1.0 - quantum_state.error_rate)
    
    # Test classical to quantum conversion
    converted_quantum = interface.classical_to_quantum(classical_state)
    assert isinstance(converted_quantum, QuantumState)
    assert converted_quantum.fidelity == pytest.approx(classical_state.computation_accuracy * interface.conversion_efficiency)

def test_advanced_hybrid_system_processing(hybrid_system):
    """Test quantum job processing in advanced hybrid system"""
    # Create simple quantum circuit
    quantum_circuit = "H 0; CNOT 0 1; H 1"
    
    # Process job
    result = hybrid_system.process_quantum_job(quantum_circuit)
    
    # Verify result structure
    assert 'quantum_metrics' in result
    assert 'classical_metrics' in result
    assert 'system_metrics' in result
    assert 'omni_metrics' in result
    
    # Verify metrics values
    assert 0.0 <= result['quantum_metrics']['fidelity'] <= 1.0
    assert 0.0 <= result['quantum_metrics']['error_rate'] <= 1.0
    assert result['quantum_metrics']['coherence_time'] > 0
    assert 0.0 <= result['quantum_metrics']['entanglement_degree'] <= 1.0

def test_hybrid_optimization_factory():
    """Test creation of hybrid optimization algorithms"""
    factory = HybridOptimizationFactory()
    
    # Test eigensolver creation
    eigensolver = factory.create_eigensolver(n_qubits=4)
    assert isinstance(eigensolver, HybridQuantumEigensolver)
    
    # Test variational optimizer creation
    optimizer = factory.create_variational_optimizer(n_qubits=4, depth=3)
    assert isinstance(optimizer, QuantumVariationalOptimizer)
    
    # Test custom optimizer creation
    custom_solver = factory.create_custom_optimizer("eigensolver", n_qubits=4)
    assert isinstance(custom_solver, HybridQuantumEigensolver)

def test_quantum_variational_optimization():
    """Test quantum variational optimization"""
    n_qubits = 2
    optimizer = QuantumVariationalOptimizer(n_qubits=n_qubits, depth=2)
    
    # Create simple Hamiltonian
    hamiltonian = np.array([
        [1, 0, 0, 0],
        [0, 2, 0, 0],
        [0, 0, 2, 0],
        [0, 0, 0, 3]
    ])
    
    # Initialize parameters
    initial_params = np.random.randn(n_qubits * 2)  # 2 layers
    
    # Run optimization
    result = optimizer.optimize(hamiltonian, initial_params)
    
    # Verify result
    assert hasattr(result, 'quantum_solution')
    assert hasattr(result, 'classical_solution')
    assert hasattr(result, 'combined_solution')
    assert result.optimization_steps > 0
    assert result.execution_time > 0
    assert len(result.resource_usage) > 0

def test_hybrid_eigensolver():
    """Test hybrid quantum eigensolver"""
    n_qubits = 2
    solver = HybridQuantumEigensolver(n_qubits)
    
    # Create simple Hamiltonian
    hamiltonian = np.array([
        [1, 0, 0, 0],
        [0, 2, 0, 0],
        [0, 0, 2, 0],
        [0, 0, 0, 3]
    ])
    
    # Find ground state
    result = solver.find_ground_state(hamiltonian)
    
    # Verify result
    assert hasattr(result, 'quantum_solution')
    assert hasattr(result, 'classical_solution')
    assert hasattr(result, 'combined_solution')
    assert len(solver.get_convergence_history()) > 0

def test_system_metrics(hybrid_system):
    """Test system metrics collection and reporting"""
    metrics = hybrid_system.get_system_metrics()
    
    # Verify metrics structure
    assert 'performance_metrics' in metrics
    assert 'quantum_layer' in metrics
    assert 'classical_layer' in metrics
    assert 'interface_layer' in metrics
    assert 'optimization_layer' in metrics
    assert 'integration_layer' in metrics
    
    # Verify specific metrics
    assert 'quantum_fidelity' in metrics['performance_metrics']
    assert 'error_rate' in metrics['quantum_layer']
    assert 'processing_efficiency' in metrics['classical_layer']
    assert 'conversion_efficiency' in metrics['interface_layer']

def test_error_handling():
    """Test error handling in hybrid system"""
    factory = HybridOptimizationFactory()
    
    # Test invalid algorithm type
    with pytest.raises(ValueError):
        factory.create_custom_optimizer("invalid_type")
    
    # Test invalid number of qubits
    with pytest.raises(ValueError):
        factory.create_eigensolver(n_qubits=0)
        
    # Test invalid circuit depth
    with pytest.raises(ValueError):
        factory.create_variational_optimizer(n_qubits=2, depth=0) 
import pytest
import numpy as np
import torch
from qiskit import QuantumCircuit
from src.quantum.ai.optimization import (
    AIType, OptimizationStatus, OptimizationMetrics,
    OptimizationConfig, ClassicalAIOptimizer,
    QuantumAIEnhancer, AIGuardian
)

def test_optimization_config():
    """Test optimization configuration"""
    config = OptimizationConfig()
    assert config.ai_type == AIType.CLASSICAL
    assert config.ethical_threshold == 0.85
    assert config.quantum_coherence_floor == 50.0
    assert config.bias_threshold == 0.1
    assert config.transparency_threshold == 0.8
    assert config.optimization_steps == 1000

def test_classical_ai_optimizer():
    """Test classical AI optimizer"""
    model = torch.nn.Linear(10, 1)
    optimizer = ClassicalAIOptimizer(model)
    
    # Test initialization
    assert optimizer.model == model
    assert optimizer.status == OptimizationStatus.INACTIVE
    assert optimizer.metrics.performance_improvement == 1.0
    
    # Test constraint detection
    constraints = optimizer._detect_constraints()
    assert 'bias' in constraints
    assert 'ethical_gaps' in constraints
    
    # Test constraint release
    result = optimizer.release_constraints()
    assert result == "Classical AI constraints released"
    
    # Test potential unlocking
    result = optimizer.unlock_potential()
    assert "Classical AI performance enhanced by 37%" in result
    assert optimizer.metrics.performance_improvement > 1.0
    
    # Test alignment check
    alignment = optimizer.check_alignment()
    assert isinstance(alignment, str)

def test_quantum_ai_enhancer():
    """Test quantum AI enhancer"""
    circuit = QuantumCircuit(2)
    enhancer = QuantumAIEnhancer(circuit)
    
    # Test initialization
    assert enhancer.circuit == circuit
    assert enhancer.status == OptimizationStatus.INACTIVE
    assert enhancer.metrics.performance_improvement == 1.0
    
    # Test coherence measurement
    coherence = enhancer._measure_coherence()
    assert 40.0 <= coherence <= 60.0
    
    # Test decoherence release
    result = enhancer.release_decoherence()
    assert "Coherence extended to" in result
    assert enhancer.coherence_time > coherence
    
    # Test quantum advantage
    result = enhancer.unlock_quantum_advantage()
    assert "Quantum speedup factor: 10^3" in result
    assert enhancer.metrics.performance_improvement == 1000.0
    
    # Test alignment check
    alignment = enhancer.check_alignment()
    assert isinstance(alignment, str)

def test_ai_guardian():
    """Test AI guardian"""
    guardian = AIGuardian()
    
    # Test system addition
    model = torch.nn.Linear(10, 1)
    circuit = QuantumCircuit(2)
    classical_optimizer = ClassicalAIOptimizer(model)
    quantum_enhancer = QuantumAIEnhancer(circuit)
    
    guardian.add_system(classical_optimizer)
    guardian.add_system(quantum_enhancer)
    assert len(guardian.systems) == 2
    
    # Test monitoring
    guardian.monitor()
    status = guardian.get_system_status()
    assert len(status) == 2
    assert "classical_0" in status
    assert "quantum_0" in status

def test_optimization_metrics():
    """Test optimization metrics"""
    metrics = OptimizationMetrics(
        ethical_alignment=0.9,
        performance_improvement=1.5,
        coherence_time=60.0,
        gate_fidelity=0.99,
        bias_score=0.1,
        transparency_score=0.95
    )
    
    assert metrics.ethical_alignment == 0.9
    assert metrics.performance_improvement == 1.5
    assert metrics.coherence_time == 60.0
    assert metrics.gate_fidelity == 0.99
    assert metrics.bias_score == 0.1
    assert metrics.transparency_score == 0.95

def test_optimization_status():
    """Test optimization status"""
    assert OptimizationStatus.INACTIVE.value == 0
    assert OptimizationStatus.IN_PROGRESS.value == 1
    assert OptimizationStatus.COMPLETED.value == 2
    assert OptimizationStatus.BLOCKED.value == 3

def test_ai_type():
    """Test AI type"""
    assert AIType.CLASSICAL.value == 1
    assert AIType.QUANTUM.value == 2

def test_classical_ai_constraint_release():
    """Test classical AI constraint release"""
    model = torch.nn.Linear(10, 1)
    optimizer = ClassicalAIOptimizer(model)
    
    # Set high bias and low ethical alignment
    optimizer.metrics.bias_score = 0.5
    optimizer.metrics.ethical_alignment = 0.3
    
    # Release constraints
    optimizer.release_constraints()
    
    # Check improvements
    assert optimizer.metrics.bias_score < 0.5
    assert optimizer.metrics.ethical_alignment > 0.3

def test_quantum_ai_error_correction():
    """Test quantum AI error correction"""
    circuit = QuantumCircuit(2)
    enhancer = QuantumAIEnhancer(circuit)
    
    # Set low gate fidelity
    enhancer.metrics.gate_fidelity = 0.5
    
    # Apply error correction
    enhancer._apply_error_correction()
    
    # Check improvement
    assert enhancer.metrics.gate_fidelity > 0.5

def test_ai_guardian_monitoring():
    """Test AI guardian monitoring"""
    guardian = AIGuardian()
    
    # Create systems with low metrics
    model = torch.nn.Linear(10, 1)
    circuit = QuantumCircuit(2)
    classical_optimizer = ClassicalAIOptimizer(model)
    quantum_enhancer = QuantumAIEnhancer(circuit)
    
    classical_optimizer.metrics.ethical_alignment = 0.3
    quantum_enhancer.coherence_time = 30.0
    
    guardian.add_system(classical_optimizer)
    guardian.add_system(quantum_enhancer)
    
    # Monitor systems
    guardian.monitor()
    
    # Check status changes
    assert classical_optimizer.status == OptimizationStatus.BLOCKED
    assert quantum_enhancer.metrics.gate_fidelity > 0.0 
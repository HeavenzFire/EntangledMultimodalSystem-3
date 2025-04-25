import pytest
import numpy as np
from src.quantum.core.optimization.quantum_optimization import (
    QuantumCircuitOptimizer,
    OptimizationResult
)
from src.quantum.core.quantum_circuit import QuantumCircuit
from src.quantum.core.qubit_control import QubitController

class TestQuantumCircuitOptimizer:
    @pytest.fixture
    def optimizer(self):
        return QuantumCircuitOptimizer()
    
    @pytest.fixture
    def test_circuit(self):
        circuit = QuantumCircuit(num_qubits=4)
        # Add some basic gates
        circuit.add_gate('H', 0)
        circuit.add_gate('CNOT', 0, 1)
        circuit.add_gate('X', 2)
        circuit.add_gate('CZ', 1, 3)
        return circuit
    
    def test_circuit_optimization(self, optimizer, test_circuit):
        """Test basic circuit optimization"""
        result = optimizer.optimize_circuit(test_circuit)
        
        assert isinstance(result, OptimizationResult)
        assert result.success
        assert result.optimized_circuit.num_gates <= test_circuit.num_gates
        assert result.optimized_circuit.depth <= test_circuit.depth
        assert result.error_rate < 0.1
    
    def test_gate_merging(self, optimizer, test_circuit):
        """Test gate merging optimization"""
        # Add consecutive gates that can be merged
        test_circuit.add_gate('H', 0)
        test_circuit.add_gate('H', 0)
        
        result = optimizer.optimize_circuit(test_circuit)
        assert result.success
        assert result.optimized_circuit.num_gates < test_circuit.num_gates
    
    def test_gate_cancellation(self, optimizer, test_circuit):
        """Test gate cancellation optimization"""
        # Add gates that cancel each other
        test_circuit.add_gate('X', 0)
        test_circuit.add_gate('X', 0)
        
        result = optimizer.optimize_circuit(test_circuit)
        assert result.success
        assert result.optimized_circuit.num_gates < test_circuit.num_gates
    
    def test_circuit_compilation(self, optimizer, test_circuit):
        """Test circuit compilation"""
        result = optimizer.compile_circuit(test_circuit)
        
        assert isinstance(result, OptimizationResult)
        assert result.success
        assert result.optimized_circuit.num_gates <= test_circuit.num_gates
        assert result.optimized_circuit.depth <= test_circuit.depth
        assert result.error_rate < 0.1
    
    def test_optimization_metrics(self, optimizer, test_circuit):
        """Test optimization metrics"""
        result = optimizer.optimize_circuit(test_circuit)
        
        assert result.gate_reduction > 0
        assert result.depth_reduction > 0
        assert result.error_rate < 0.1
        assert result.coherence > 0.8
    
    def test_optimization_robustness(self, optimizer, test_circuit):
        """Test optimization robustness"""
        # Test multiple optimization passes
        results = []
        for _ in range(5):
            result = optimizer.optimize_circuit(test_circuit)
            results.append(result.gate_reduction)
        
        # Check consistency of results
        assert np.std(results) < 0.2  # Should be relatively consistent
        assert all(reduction > 0 for reduction in results)
    
    def test_error_handling(self, optimizer):
        """Test error handling"""
        # Test with invalid circuit
        with pytest.raises(Exception):
            optimizer.optimize_circuit(None)
        
        # Test with empty circuit
        empty_circuit = QuantumCircuit(num_qubits=4)
        with pytest.raises(Exception):
            optimizer.optimize_circuit(empty_circuit)
    
    def test_optimization_quality(self, optimizer, test_circuit):
        """Test optimization quality"""
        result = optimizer.optimize_circuit(test_circuit)
        
        assert result.gate_reduction > 0.1  # Should achieve significant reduction
        assert result.depth_reduction > 0.1  # Should reduce circuit depth
        assert result.error_rate < 0.1  # Should maintain low error rate
        assert result.coherence > 0.8  # Should maintain high coherence
    
    def test_optimization_efficiency(self, optimizer, test_circuit):
        """Test optimization efficiency"""
        import time
        
        start_time = time.time()
        result = optimizer.optimize_circuit(test_circuit)
        optimization_time = time.time() - start_time
        
        assert result.success
        assert optimization_time < 1.0  # Should optimize within reasonable time
        assert result.gate_reduction > 0.1
    
    def test_circuit_equivalence(self, optimizer, test_circuit):
        """Test circuit equivalence after optimization"""
        result = optimizer.optimize_circuit(test_circuit)
        
        # Verify that the optimized circuit produces the same results
        original_result = test_circuit.execute()
        optimized_result = result.optimized_circuit.execute()
        
        assert np.allclose(original_result, optimized_result, atol=1e-6)
    
    def test_optimization_consistency(self, optimizer, test_circuit):
        """Test consistency of optimization results"""
        results = []
        for _ in range(10):
            result = optimizer.optimize_circuit(test_circuit)
            results.append(result.gate_reduction)
        
        # Check consistency of results
        assert np.std(results) < 0.2  # Should be relatively consistent
        assert all(reduction > 0 for reduction in results)
    
    def test_compilation_targets(self, optimizer, test_circuit):
        """Test compilation for different targets"""
        targets = ['ibm', 'rigetti', 'ionq']
        
        for target in targets:
            result = optimizer.compile_circuit(test_circuit, target=target)
            assert result.success
            assert result.optimized_circuit.num_gates <= test_circuit.num_gates
            assert result.error_rate < 0.1 
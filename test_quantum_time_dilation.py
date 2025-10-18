#!/usr/bin/env python3
"""
Test suite for the Quantum Time Dilation Framework.
This test suite validates:
1. Basic functionality of the framework
2. Performance characteristics
3. State management and prediction accuracy
4. Error handling and edge cases
5. Adaptive acceleration mechanisms
6. Coherence protection features
"""

import unittest
import numpy as np
import time
from qiskit import QuantumCircuit, execute, Aer
from quantum_time_dilation import QuantumTimeDilation, TimeStream

class TestQuantumTimeDilation(unittest.TestCase):
    """Test suite for the Quantum Time Dilation implementation."""
    
    def setUp(self):
        """Set up test cases."""
        self.num_qubits = 4
        self.num_streams = 5
        self.qtd = QuantumTimeDilation(
            num_qubits=self.num_qubits,
            num_streams=self.num_streams,
            base_acceleration=5.0,
            predictive_depth=3,
            adaptive_rate=0.1,
            coherence_threshold=0.95
        )
    
    def test_initialization(self):
        """Test proper initialization of the quantum time dilation system."""
        self.assertEqual(len(self.qtd.circuits), self.num_streams)
        self.assertEqual(len(self.qtd.states), self.num_streams)
        
        # Check that each circuit has the correct number of qubits
        for circuit in self.qtd.circuits:
            self.assertEqual(circuit.num_qubits, self.num_qubits)
    
    def test_state_evolution(self):
        """Test quantum state evolution."""
        initial_state = self.qtd.states[0]
        evolved_state = self.qtd.evolve_state(self.qtd.circuits[0], 0.1)
        
        # Check that the state has evolved (should be different from initial)
        self.assertFalse(np.array_equal(initial_state.data, evolved_state.data))
        
        # Check that the evolved state is still normalized
        self.assertAlmostEqual(np.sum(np.abs(evolved_state.data) ** 2), 1.0)
    
    def test_state_prediction(self):
        """Test future state prediction."""
        current_state = self.qtd.states[0]
        predicted_state = self.qtd.predict_future_state(current_state, 3)
        
        # Check that predicted state is different from current
        self.assertFalse(np.array_equal(current_state.data, predicted_state.data))
        
        # Check that predicted state is normalized
        self.assertAlmostEqual(np.sum(np.abs(predicted_state.data) ** 2), 1.0)
    
    def test_coherence_measurement(self):
        """Test coherence measurement."""
        state = self.qtd.states[0]
        coherence = self.qtd.measure_coherence(state)
        
        # Coherence should be between 0 and 1
        self.assertGreaterEqual(coherence, 0.0)
        self.assertLessEqual(coherence, 1.0)
    
    def test_computation_acceleration(self):
        """Test computation acceleration."""
        # Create a simple test circuit
        qc = QuantumCircuit(self.num_qubits)
        for i in range(self.num_qubits):
            qc.h(i)
        
        # Run accelerated computation
        results = self.qtd.accelerate_computation(qc, target_time=1.0)
        
        # Check that all expected keys are present
        expected_keys = {
            'execution_time', 'virtual_time_reached', 'average_performance',
            'average_coherence', 'performance_history', 'coherence_history',
            'final_state'
        }
        self.assertEqual(set(results.keys()), expected_keys)
        
        # Check that performance metrics are reasonable
        self.assertGreaterEqual(results['average_performance'], 0.0)
        self.assertLessEqual(results['average_performance'], 1.0)
        self.assertGreaterEqual(results['average_coherence'], 0.0)
        self.assertLessEqual(results['average_coherence'], 1.0)
    
    def test_visualization(self):
        """Test visualization methods."""
        # Create a simple test circuit
        qc = QuantumCircuit(self.num_qubits)
        for i in range(self.num_qubits):
            qc.h(i)
        
        # Run computation and get results
        results = self.qtd.accelerate_computation(qc, target_time=1.0)
        
        # Test visualization methods (should not raise exceptions)
        try:
            self.qtd.visualize_results(results)
            self.qtd.visualize_metrics()
        except Exception as e:
            self.fail(f"Visualization failed with error: {str(e)}")

if __name__ == '__main__':
    unittest.main() 
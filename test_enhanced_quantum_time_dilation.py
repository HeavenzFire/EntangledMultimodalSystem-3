#!/usr/bin/env python3
"""
Test suite for the Enhanced Quantum Time Dilation Framework.
This test suite validates:
1. Basic functionality of the framework
2. Performance characteristics
3. State management and prediction accuracy
4. Error handling and edge cases
5. Adaptive acceleration mechanisms
6. Coherence protection features
7. Visualization capabilities
"""

import unittest
import numpy as np
import time
from qiskit import QuantumCircuit, Aer
from qiskit.primitives import Sampler
from enhanced_quantum_time_dilation import QuantumTimeDilation, QuantumStream, PerformanceMetrics

class TestEnhancedQuantumTimeDilation(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test."""
        self.num_qubits = 3
        self.num_streams = 10
        self.target_time = 0.5
        
        # Create a simple test circuit
        self.test_circuit = QuantumCircuit(self.num_qubits)
        self.test_circuit.h(range(self.num_qubits))
        self.test_circuit.measure_all()
        
        # Initialize the framework
        self.qtd = QuantumTimeDilation(
            num_qubits=self.num_qubits,
            num_streams=self.num_streams,
            base_acceleration=1.0,  # Use smaller acceleration for testing
            adaptive_rate=0.1
        )
    
    def test_initialization(self):
        """Test proper initialization of the QuantumTimeDilation class."""
        self.assertEqual(len(self.qtd.streams), self.num_streams)
        self.assertIsInstance(self.qtd.streams[0], QuantumStream)
        self.assertEqual(self.qtd.num_qubits, self.num_qubits)
        self.assertEqual(self.qtd.num_streams, self.num_streams)
    
    def test_stream_initialization(self):
        """Test proper initialization of individual quantum streams."""
        stream = self.qtd.streams[0]
        self.assertGreater(stream.acceleration_factor, 0)
        self.assertEqual(stream.virtual_time, 0.0)
        self.assertIsNotNone(stream.quantum_state)
        self.assertIsInstance(stream.performance_history, list)
        self.assertEqual(len(stream.performance_history), 0)
    
    def test_acceleration_factor_distribution(self):
        """Test that acceleration factors are properly distributed."""
        factors = [stream.acceleration_factor for stream in self.qtd.streams]
        self.assertGreater(min(factors), 0)
        self.assertLess(max(factors), 2.0)  # Max factor is 2.0
        # Check for reasonable distribution
        self.assertGreater(np.std(factors), 0)
    
    def test_basic_computation(self):
        """Test basic quantum computation with time dilation."""
        results = self.qtd.accelerate_computation(
            self.test_circuit,
            self.target_time
        )
        
        self.assertIn('virtual_time_reached', results)
        self.assertIn('state_variance', results)
        self.assertIn('num_predictions', results)
        self.assertIn('average_performance', results)
        self.assertIn('average_coherence', results)
        self.assertIn('acceleration_distribution', results)
        self.assertGreater(results['virtual_time_reached'], 0)
    
    def test_state_prediction(self):
        """Test quantum state prediction accuracy."""
        # Get a stream's quantum state
        stream = self.qtd.streams[0]
        initial_state = stream.quantum_state
        
        predicted_state = self.qtd._predict_quantum_state(initial_state)
        
        self.assertEqual(len(predicted_state.data), len(initial_state.data))
        self.assertAlmostEqual(np.linalg.norm(predicted_state.data), 1.0)
    
    def test_state_update(self):
        """Test quantum state update mechanism."""
        # Get a stream's quantum state
        stream = self.qtd.streams[0]
        initial_state = stream.quantum_state
        
        # Create a predicted and measured state
        predicted_state = self.qtd._predict_quantum_state(initial_state)
        
        # Create a measured state (simplified for testing)
        measured_state = initial_state.copy()
        measured_state.data = measured_state.data * 0.9  # Slightly different
        
        updated_state = self.qtd._update_quantum_state(
            initial_state, predicted_state, measured_state)
        
        self.assertEqual(len(updated_state.data), len(initial_state.data))
        self.assertAlmostEqual(np.linalg.norm(updated_state.data), 1.0)
    
    def test_performance_scaling(self):
        """Test performance scaling with different numbers of streams."""
        stream_counts = [5, 10, 20]
        execution_times = []
        
        for num_streams in stream_counts:
            qtd = QuantumTimeDilation(
                num_qubits=self.num_qubits,
                num_streams=num_streams,
                base_acceleration=1.0
            )
            start_time = time.time()
            qtd.accelerate_computation(self.test_circuit, self.target_time)
            execution_times.append(time.time() - start_time)
        
        # Check that execution time increases with number of streams
        self.assertGreater(execution_times[1], execution_times[0])
        self.assertGreater(execution_times[2], execution_times[1])
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test with invalid circuit
        with self.assertRaises(Exception):
            self.qtd.accelerate_computation(None, self.target_time)
        
        # Test with invalid target time
        with self.assertRaises(Exception):
            self.qtd.accelerate_computation(self.test_circuit, -1.0)
        
        # Test with zero streams
        with self.assertRaises(Exception):
            QuantumTimeDilation(num_qubits=self.num_qubits, num_streams=0)
    
    def test_state_conservation(self):
        """Test conservation of quantum state properties."""
        results = self.qtd.accelerate_computation(
            self.test_circuit,
            self.target_time
        )
        
        # Check that state variance is reasonable
        variance = results['state_variance']
        self.assertGreater(np.mean(variance), 0)
        self.assertLess(np.mean(variance), 1.0)
    
    def test_virtual_time_progression(self):
        """Test virtual time progression across streams."""
        results = self.qtd.accelerate_computation(
            self.test_circuit,
            self.target_time
        )
        
        virtual_times = [stream.virtual_time for stream in self.qtd.streams]
        self.assertGreater(min(virtual_times), 0)
        self.assertLessEqual(max(virtual_times), self.target_time)
    
    def test_prediction_accuracy(self):
        """Test accuracy of quantum state predictions."""
        # Run multiple predictions and compare with actual quantum evolution
        simulator = Aer.get_backend('statevector_simulator')
        sampler = Sampler()
        actual_result = sampler.run(self.test_circuit).result()
        actual_state = actual_result.quasi_dists[0]
        
        predicted_states = []
        for _ in range(5):
            predicted_state = self.qtd._predict_quantum_state(actual_state)
            predicted_states.append(predicted_state)
        
        # Check that predictions maintain reasonable fidelity with actual state
        fidelities = [np.abs(np.vdot(actual_state, pred_state))**2 
                     for pred_state in predicted_states]
        self.assertGreater(np.mean(fidelities), 0.5)
    
    def test_adaptive_acceleration(self):
        """Test adaptive acceleration mechanism."""
        # Create a stream with initial acceleration factor
        stream = QuantumStream(0, self.num_qubits, 1.0)
        
        # Test with good performance (should increase acceleration)
        good_metrics = PerformanceMetrics(
            fidelity=0.98,
            coherence_level=0.95,
            execution_time=0.1,
            acceleration_factor=1.0,
            prediction_accuracy=0.9,
            virtual_time=0.1
        )
        new_factor = stream.adjust_acceleration(good_metrics, 0.1)
        self.assertGreater(new_factor, stream.acceleration_factor)
        
        # Test with poor performance (should decrease acceleration)
        poor_metrics = PerformanceMetrics(
            fidelity=0.7,
            coherence_level=0.7,
            execution_time=0.1,
            acceleration_factor=1.0,
            prediction_accuracy=0.6,
            virtual_time=0.1
        )
        new_factor = stream.adjust_acceleration(poor_metrics, 0.1)
        self.assertLess(new_factor, stream.acceleration_factor)
        
        # Test with boundary conditions
        min_factor = 0.5
        max_factor = 2.0
        
        # Test minimum boundary
        stream.acceleration_factor = min_factor
        new_factor = stream.adjust_acceleration(poor_metrics, 0.1, min_factor, max_factor)
        self.assertGreaterEqual(new_factor, min_factor)
        
        # Test maximum boundary
        stream.acceleration_factor = max_factor
        new_factor = stream.adjust_acceleration(good_metrics, 0.1, min_factor, max_factor)
        self.assertLessEqual(new_factor, max_factor)
    
    def test_coherence_protection(self):
        """Test coherence protection mechanism."""
        # Create a quantum state
        qc = QuantumCircuit(self.num_qubits)
        qc.h(range(self.num_qubits))
        state = qc.get_statevector()
        
        # Apply coherence protection
        stream = QuantumStream(0, self.num_qubits, 1.0)
        stream.quantum_state = state
        stream.apply_coherence_protection()
        
        # Check that the state is normalized
        self.assertAlmostEqual(np.linalg.norm(stream.quantum_state.data), 1.0)
        
        # Check that the phase correction was applied
        # The phase of each component should be preserved
        original_phases = np.angle(state.data)
        protected_phases = np.angle(stream.quantum_state.data)
        self.assertTrue(np.allclose(original_phases, protected_phases))
    
    def test_performance_tracking(self):
        """Test that performance metrics are properly tracked."""
        results = self.qtd.accelerate_computation(
            self.test_circuit,
            self.target_time
        )
        
        # Check that performance metrics are included in results
        self.assertIn('performance_metrics', results)
        self.assertIn('average_performance', results)
        self.assertIn('average_coherence', results)
        
        # Check that performance history is updated for streams
        for stream in self.qtd.streams:
            self.assertGreater(len(stream.performance_history), 0)
    
    def test_visualization(self):
        """Test visualization functionality."""
        results = self.qtd.accelerate_computation(
            self.test_circuit,
            self.target_time
        )
        
        # Test that visualization doesn't raise exceptions
        try:
            self.qtd.visualize_results(results)
            self.qtd.visualize_metrics()
        except Exception as e:
            self.fail(f"Visualization raised an exception: {str(e)}")

if __name__ == '__main__':
    unittest.main() 
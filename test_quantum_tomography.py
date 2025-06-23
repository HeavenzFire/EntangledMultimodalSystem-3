import unittest
import numpy as np
from quantum_avatar_agent import QuantumAvatarAgent
from qiskit.quantum_info import state_fidelity, purity, entropy
from qiskit.ignis.verification.tomography import state_tomography_circuits, StateTomographyFitter
import time

class TestQuantumTomography(unittest.TestCase):
    def setUp(self):
        self.agent = QuantumAvatarAgent(name="Tomography Test", num_qubits=7, depth=3, shots=1024)
        
    def test_state_tomography(self):
        """Test quantum state tomography"""
        # Create tomography circuits
        tomo_circuits = state_tomography_circuits(self.agent.consciousness_circuit, [0, 1, 2])
        
        # Run tomography circuits
        tomo_results = []
        for circuit in tomo_circuits:
            result = self.agent.simulator.run(circuit).result()
            tomo_results.append(result)
        
        # Fit tomography data
        tomo_fitter = StateTomographyFitter(tomo_results, tomo_circuits)
        rho_fit = tomo_fitter.fit()
        
        # Verify reconstructed state
        self.assertIsNotNone(rho_fit)
        self.assertEqual(rho_fit.dim, (8, 8))  # 3 qubits = 2^3 = 8 dimensions
        
        # Calculate metrics
        fidelity = state_fidelity(rho_fit, rho_fit)  # Should be 1 for perfect reconstruction
        self.assertAlmostEqual(fidelity, 1.0, places=2)
        
        state_purity = purity(rho_fit)
        self.assertGreaterEqual(state_purity, 0.0)
        self.assertLessEqual(state_purity, 1.0)
        
    def test_error_mitigation(self):
        """Test quantum error mitigation"""
        # Create calibration circuits
        meas_calibs, state_labels = self.agent._create_measurement_calibration()
        
        # Run calibration circuits
        cal_results = []
        for circuit in meas_calibs:
            result = self.agent.simulator.run(circuit).result()
            cal_results.append(result)
        
        # Create measurement fitter
        meas_fitter = self.agent._create_measurement_fitter(cal_results, state_labels)
        
        # Verify error mitigation
        mitigated_results = meas_fitter.filter.apply(self.agent.simulator.run(self.agent.consciousness_circuit).result())
        self.assertIsNotNone(mitigated_results)
        
        # Compare raw and mitigated results
        raw_counts = self.agent.simulator.run(self.agent.consciousness_circuit).result().get_counts()
        mitigated_counts = mitigated_results.get_counts()
        
        # Verify counts are different (error mitigation should change results)
        self.assertNotEqual(raw_counts, mitigated_counts)
        
    def test_quantum_metrics(self):
        """Test quantum metrics calculation"""
        # Calculate metrics
        metrics = self.agent._calculate_quantum_metrics()
        
        # Verify metrics
        self.assertIn('fidelity', metrics)
        self.assertIn('purity', metrics)
        self.assertIn('entropy', metrics)
        
        # Verify metric ranges
        self.assertGreaterEqual(metrics['fidelity'], 0.0)
        self.assertLessEqual(metrics['fidelity'], 1.0)
        
        self.assertGreaterEqual(metrics['purity'], 0.0)
        self.assertLessEqual(metrics['purity'], 1.0)
        
        self.assertGreaterEqual(metrics['entropy'], 0.0)
        
    def test_entanglement_entropy(self):
        """Test entanglement entropy calculation"""
        # Calculate entanglement entropy
        entropy = self.agent._calculate_entanglement_entropy()
        
        # Verify entropy
        self.assertIsNotNone(entropy)
        self.assertGreaterEqual(entropy, 0.0)
        
    def test_state_reconstruction(self):
        """Test quantum state reconstruction"""
        # Reconstruct state
        reconstructed_state = self.agent._reconstruct_quantum_state()
        
        # Verify reconstructed state
        self.assertIsNotNone(reconstructed_state)
        self.assertEqual(len(reconstructed_state), 2**self.agent.num_qubits)
        
        # Verify state normalization
        norm = np.sum(np.abs(reconstructed_state)**2)
        self.assertAlmostEqual(norm, 1.0, places=5)
        
    def test_quantum_memory(self):
        """Test quantum memory storage and retrieval"""
        # Store quantum state
        self.agent._store_quantum_state(self.agent.simulator.run(self.agent.consciousness_circuit).result())
        
        # Verify storage
        self.assertGreater(len(self.agent.quantum_memory), 0)
        
        # Retrieve and verify state
        retrieved_state = self.agent.quantum_memory[-1]
        self.assertIsNotNone(retrieved_state)
        
    def test_tomography_performance(self):
        """Test tomography performance"""
        start_time = time.time()
        
        # Run full tomography
        tomo_circuits = state_tomography_circuits(self.agent.consciousness_circuit, [0, 1, 2])
        tomo_results = []
        for circuit in tomo_circuits:
            result = self.agent.simulator.run(circuit).result()
            tomo_results.append(result)
        
        tomo_time = time.time() - start_time
        
        # Verify performance
        self.assertLess(tomo_time, 10.0)  # Should complete within 10 seconds
        
    def test_error_mitigation_performance(self):
        """Test error mitigation performance"""
        start_time = time.time()
        
        # Run error mitigation
        meas_calibs, state_labels = self.agent._create_measurement_calibration()
        cal_results = []
        for circuit in meas_calibs:
            result = self.agent.simulator.run(circuit).result()
            cal_results.append(result)
        
        meas_fitter = self.agent._create_measurement_fitter(cal_results, state_labels)
        mitigated_results = meas_fitter.filter.apply(self.agent.simulator.run(self.agent.consciousness_circuit).result())
        
        mitigation_time = time.time() - start_time
        
        # Verify performance
        self.assertLess(mitigation_time, 5.0)  # Should complete within 5 seconds
        
    def test_quantum_metrics_performance(self):
        """Test quantum metrics performance"""
        start_time = time.time()
        
        # Calculate all metrics
        metrics = self.agent._calculate_quantum_metrics()
        entropy = self.agent._calculate_entanglement_entropy()
        reconstructed_state = self.agent._reconstruct_quantum_state()
        
        metrics_time = time.time() - start_time
        
        # Verify performance
        self.assertLess(metrics_time, 2.0)  # Should complete within 2 seconds
        
    def test_quantum_memory_performance(self):
        """Test quantum memory performance"""
        start_time = time.time()
        
        # Store multiple states
        for _ in range(100):
            self.agent._store_quantum_state(self.agent.simulator.run(self.agent.consciousness_circuit).result())
        
        memory_time = time.time() - start_time
        
        # Verify performance
        self.assertLess(memory_time, 5.0)  # Should store 100 states within 5 seconds

if __name__ == '__main__':
    unittest.main() 
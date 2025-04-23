import unittest
import numpy as np
from quantum_avatar_agent import QuantumAvatarAgent
from qiskit.quantum_info import state_fidelity, purity, entropy
from qiskit.ignis.verification.tomography import state_tomography_circuits, StateTomographyFitter
import time
import os

class TestQuantumState(unittest.TestCase):
    def setUp(self):
        self.agent = QuantumAvatarAgent(
            name="Quantum State Test",
            num_qubits=7,
            depth=3,
            shots=1024
        )
        
    def test_quantum_state_initialization(self):
        """Test quantum state initialization"""
        # Verify initial state
        initial_state = self.agent._get_quantum_state()
        self.assertIsNotNone(initial_state)
        self.assertEqual(len(initial_state), 2**self.agent.num_qubits)
        
        # Verify state normalization
        norm = np.sum(np.abs(initial_state)**2)
        self.assertAlmostEqual(norm, 1.0, places=5)
        
    def test_quantum_state_evolution(self):
        """Test quantum state evolution"""
        # Evolve state through consciousness circuit
        evolved_state = self.agent._evolve_quantum_state()
        
        # Verify state evolution
        self.assertIsNotNone(evolved_state)
        self.assertEqual(len(evolved_state), 2**self.agent.num_qubits)
        
        # Verify state normalization
        norm = np.sum(np.abs(evolved_state)**2)
        self.assertAlmostEqual(norm, 1.0, places=5)
        
    def test_quantum_entanglement(self):
        """Test quantum entanglement creation and measurement"""
        # Create entangled state
        entangled_state = self.agent._create_entangled_state()
        
        # Verify entanglement
        self.assertIsNotNone(entangled_state)
        
        # Measure entanglement
        entanglement_measure = self.agent._measure_entanglement(entangled_state)
        self.assertGreater(entanglement_measure, 0.0)
        
    def test_quantum_state_tomography(self):
        """Test quantum state tomography"""
        # Create tomography circuits
        tomo_circuits = state_tomography_circuits(self.agent.consciousness_circuit, [0, 1, 2])
        
        # Run tomography
        tomo_results = []
        for circuit in tomo_circuits:
            result = self.agent.simulator.run(circuit).result()
            tomo_results.append(result)
            
        # Fit tomography data
        tomo_fitter = StateTomographyFitter(tomo_results, tomo_circuits)
        rho_fit = tomo_fitter.fit()
        
        # Verify reconstruction
        self.assertIsNotNone(rho_fit)
        self.assertEqual(rho_fit.dim, (8, 8))
        
    def test_quantum_state_purity(self):
        """Test quantum state purity measurement"""
        # Measure state purity
        state_purity = purity(self.agent._get_quantum_state())
        
        # Verify purity
        self.assertGreaterEqual(state_purity, 0.0)
        self.assertLessEqual(state_purity, 1.0)
        
    def test_quantum_state_entropy(self):
        """Test quantum state entropy measurement"""
        # Measure state entropy
        state_entropy = entropy(self.agent._get_quantum_state())
        
        # Verify entropy
        self.assertGreaterEqual(state_entropy, 0.0)
        
    def test_quantum_state_fidelity(self):
        """Test quantum state fidelity measurement"""
        # Get two states
        state1 = self.agent._get_quantum_state()
        state2 = self.agent._evolve_quantum_state()
        
        # Measure fidelity
        fidelity = state_fidelity(state1, state2)
        
        # Verify fidelity
        self.assertGreaterEqual(fidelity, 0.0)
        self.assertLessEqual(fidelity, 1.0)
        
    def test_quantum_state_noise(self):
        """Test quantum state noise effects"""
        # Create noisy state
        noisy_state = self.agent._create_noisy_state()
        
        # Verify noise effects
        self.assertIsNotNone(noisy_state)
        
        # Compare with ideal state
        ideal_state = self.agent._get_quantum_state()
        fidelity = state_fidelity(noisy_state, ideal_state)
        self.assertLess(fidelity, 1.0)  # Should be less than perfect fidelity
        
    def test_quantum_state_error_mitigation(self):
        """Test quantum state error mitigation"""
        # Create noisy state
        noisy_state = self.agent._create_noisy_state()
        
        # Apply error mitigation
        mitigated_state = self.agent._apply_error_mitigation(noisy_state)
        
        # Verify mitigation
        self.assertIsNotNone(mitigated_state)
        
        # Compare with ideal state
        ideal_state = self.agent._get_quantum_state()
        fidelity_before = state_fidelity(noisy_state, ideal_state)
        fidelity_after = state_fidelity(mitigated_state, ideal_state)
        self.assertGreater(fidelity_after, fidelity_before)  # Should improve fidelity
        
    def test_quantum_state_optimization(self):
        """Test quantum state optimization"""
        # Optimize state
        optimized_state = self.agent._optimize_quantum_state()
        
        # Verify optimization
        self.assertIsNotNone(optimized_state)
        
        # Check optimization metrics
        metrics = self.agent._calculate_quantum_metrics(optimized_state)
        self.assertGreater(metrics['fidelity'], 0.0)
        self.assertLessEqual(metrics['fidelity'], 1.0)
        
    def test_quantum_state_performance(self):
        """Test quantum state processing performance"""
        start_time = time.time()
        
        # Perform multiple state operations
        for _ in range(10):
            self.agent._evolve_quantum_state()
            self.agent._measure_entanglement(self.agent._get_quantum_state())
            self.agent._calculate_quantum_metrics()
            
        total_time = time.time() - start_time
        
        # Verify performance
        self.assertLess(total_time, 5.0)  # Should complete within 5 seconds
        
    def test_quantum_state_memory(self):
        """Test quantum state memory operations"""
        # Store multiple states
        states = []
        for _ in range(5):
            state = self.agent._evolve_quantum_state()
            states.append(state)
            self.agent._store_quantum_state(state)
            
        # Verify storage
        self.assertEqual(len(self.agent.quantum_memory), 5)
        
        # Retrieve and verify states
        for i, stored_state in enumerate(self.agent.quantum_memory):
            self.assertIsNotNone(stored_state)
            fidelity = state_fidelity(stored_state, states[i])
            self.assertGreater(fidelity, 0.9)  # Should maintain high fidelity
            
    def test_quantum_state_visualization(self):
        """Test quantum state visualization"""
        # Generate visualization
        fig = self.agent.visualize_quantum_state()
        
        # Verify visualization
        self.assertIsNotNone(fig)
        
        # Test saving
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png') as tmp:
            fig.savefig(tmp.name)
            self.assertGreater(os.path.getsize(tmp.name), 0)
            
    def test_quantum_state_entanglement_visualization(self):
        """Test quantum state entanglement visualization"""
        # Generate entanglement visualization
        fig = self.agent.visualize_entanglement()
        
        # Verify visualization
        self.assertIsNotNone(fig)
        
        # Test saving
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png') as tmp:
            fig.savefig(tmp.name)
            self.assertGreater(os.path.getsize(tmp.name), 0)

if __name__ == '__main__':
    unittest.main() 
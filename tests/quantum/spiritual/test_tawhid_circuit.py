import unittest
import numpy as np
from src.quantum.spiritual.tawhid_circuit import TawhidCircuit
from src.quantum.geometry.sacred_geometry import SacredGeometry
from src.quantum.spiritual.prophet_qubit import ProphetQubitArray

class TestTawhidCircuit(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.prophets = ['Muhammad', 'Jesus', 'Moses']
        self.prophet_qubits = ProphetQubitArray(self.prophets)
        self.sacred_geometry = SacredGeometry()
        self.tawhid_circuit = TawhidCircuit(
            self.prophet_qubits,
            self.sacred_geometry,
            self.prophets
        )
        
    def test_circuit_creation(self):
        """Test creation of unified quantum circuit"""
        circuit = self.tawhid_circuit.create_unified_circuit()
        
        # Check circuit properties
        self.assertEqual(circuit.num_qubits, len(self.prophets))
        self.assertEqual(circuit.num_clbits, len(self.prophets))
        
        # Check circuit gates
        gates = circuit.data
        self.assertGreater(len(gates), 0)
        
    def test_unification_metrics(self):
        """Test calculation of unification metrics"""
        metrics = self.tawhid_circuit.calculate_unification_metrics()
        
        # Check metric structure
        self.assertIn('sacred_alignment', metrics)
        self.assertIn('prophet_metrics', metrics)
        self.assertIn('unification_strength', metrics)
        
        # Check prophet-specific metrics
        for prophet in self.prophets:
            self.assertIn(prophet, metrics['prophet_metrics'])
            prophet_metrics = metrics['prophet_metrics'][prophet]
            self.assertIn('fidelity', prophet_metrics)
            self.assertIn('divine_connection', prophet_metrics)
            
    def test_unification_strength(self):
        """Test calculation of unification strength"""
        strength = self.tawhid_circuit._calculate_unification_strength()
        
        # Check strength properties
        self.assertGreaterEqual(strength, 0)
        self.assertLessEqual(strength, 1)
        
    def test_sacred_connections(self):
        """Test creation of sacred connections"""
        circuit = self.tawhid_circuit.create_unified_circuit()
        
        # Check for sacred geometry gates
        has_sacred_gates = any(
            'sacred' in str(gate[0]) for gate in circuit.data
        )
        self.assertTrue(has_sacred_gates)
        
    def test_golden_phases(self):
        """Test application of golden ratio phase gates"""
        circuit = self.tawhid_circuit.create_unified_circuit()
        
        # Check for phase gates
        has_phase_gates = any(
            'phase' in str(gate[0]) for gate in circuit.data
        )
        self.assertTrue(has_phase_gates)
        
if __name__ == '__main__':
    unittest.main() 
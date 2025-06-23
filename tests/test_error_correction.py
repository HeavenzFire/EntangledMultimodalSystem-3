import unittest
import numpy as np
from qiskit import QuantumCircuit, execute, Aer
from qiskit.providers.aer.noise import NoiseModel
from qiskit.quantum_info import Operator, Statevector
from core.quantum.error_correction.surface_code import SurfaceCode
from core.quantum.error_correction.steane_code import SteaneCode
from core.quantum.error_correction.error_correction_manager import ErrorCorrectionManager

class TestQuantumErrorCorrection(unittest.TestCase):
    """Test suite for quantum error correction implementations."""
    
    def setUp(self):
        """Set up test environment."""
        self.simulator = Aer.get_backend('qasm_simulator')
        
        # Create noise model for testing
        self.noise_model = NoiseModel()
        self.noise_model.add_all_qubit_quantum_error(
            QuantumCircuit([['x', [0]]], name='bit flip'),
            ['u1', 'u2', 'u3'],
            0.01
        )
        
    def test_surface_code_initialization(self):
        """Test surface code initialization."""
        code = SurfaceCode(distance=3)
        self.assertEqual(code.distance, 3)
        self.assertEqual(code.num_physical_qubits, 9)
        self.assertEqual(code.num_syndrome_qubits, 4)
        
        # Test invalid distance
        with self.assertRaises(ValueError):
            SurfaceCode(distance=2)
            
    def test_steane_code_initialization(self):
        """Test Steane code initialization."""
        code = SteaneCode()
        self.assertEqual(code.num_physical_qubits, 7)
        self.assertEqual(code.num_syndrome_bits, 6)
        
    def test_surface_code_encoding(self):
        """Test surface code state encoding."""
        code = SurfaceCode(distance=3)
        circuit = code.encode_logical_qubit()
        
        # Verify circuit properties
        self.assertIsInstance(circuit, QuantumCircuit)
        self.assertEqual(circuit.num_qubits, 13)  # 9 data + 4 syndrome
        
        # Test encoding with specific state
        state = 1/np.sqrt(2)  # |+⟩ state
        circuit = code.encode_logical_qubit(state)
        
        # Execute circuit and verify results
        job = execute(circuit, self.simulator, shots=1024)
        results = job.result().get_counts()
        
        # Verify stabilizer conditions
        self.assertTrue(all(code.verify_logical_state(circuit) for _ in range(10)))
        
    def test_steane_code_encoding(self):
        """Test Steane code state encoding."""
        code = SteaneCode()
        circuit = code.encode_logical_qubit()
        
        # Verify circuit properties
        self.assertIsInstance(circuit, QuantumCircuit)
        self.assertEqual(circuit.num_qubits, 13)  # 7 data + 6 ancilla
        
        # Test encoding with specific state
        state = 1j  # |1⟩ state
        circuit = code.encode_logical_qubit(state)
        
        # Execute circuit and verify results
        job = execute(circuit, self.simulator, shots=1024)
        results = job.result().get_counts()
        
        # Verify encoding
        self.assertTrue(code.verify_encoding(circuit))
        
    def test_error_correction_manager(self):
        """Test error correction manager functionality."""
        # Test surface code management
        manager = ErrorCorrectionManager(code_type='surface', distance=3)
        self.assertEqual(manager.code_type, 'surface')
        self.assertIsInstance(manager.code, SurfaceCode)
        
        # Test Steane code management
        manager = ErrorCorrectionManager(code_type='steane')
        self.assertEqual(manager.code_type, 'steane')
        self.assertIsInstance(manager.code, SteaneCode)
        
        # Test invalid code type
        with self.assertRaises(ValueError):
            ErrorCorrectionManager(code_type='invalid')
            
    def test_error_detection_and_correction(self):
        """Test error detection and correction capabilities."""
        # Test with surface code
        surface_manager = ErrorCorrectionManager(code_type='surface', distance=3)
        circuit = surface_manager.encode_state()
        
        # Introduce errors
        circuit.x(0)  # Bit flip error
        circuit.z(1)  # Phase flip error
        
        # Measure syndromes
        surface_manager.measure_syndrome(circuit)
        
        # Execute circuit with noise
        job = execute(circuit, 
                     self.simulator,
                     noise_model=self.noise_model,
                     shots=1024)
        results = job.result().get_counts()
        
        # Get correction operations
        corrections = surface_manager.correct_errors(results)
        self.assertIsInstance(corrections, list)
        
        # Test with Steane code
        steane_manager = ErrorCorrectionManager(code_type='steane')
        circuit = steane_manager.encode_state()
        
        # Introduce errors
        circuit.x(0)  # Bit flip error
        
        # Measure syndromes
        steane_manager.measure_syndrome(circuit)
        
        # Execute circuit with noise
        job = execute(circuit,
                     self.simulator,
                     noise_model=self.noise_model,
                     shots=1024)
        results = job.result().get_counts()
        
        # Get correction operations
        corrections = steane_manager.correct_errors(results)
        self.assertIsInstance(corrections, list)
        
    def test_logical_operations(self):
        """Test logical operations on encoded states."""
        manager = ErrorCorrectionManager(code_type='steane')
        circuit = manager.encode_state()
        
        # Test logical X operation
        manager.apply_logical_operation(circuit, 'X')
        self.assertTrue(manager.verify_encoding(circuit))
        
        # Test logical Z operation
        manager.apply_logical_operation(circuit, 'Z')
        self.assertTrue(manager.verify_encoding(circuit))
        
        # Test logical H operation
        manager.apply_logical_operation(circuit, 'H')
        self.assertTrue(manager.verify_encoding(circuit))
        
        # Test invalid operation
        with self.assertRaises(ValueError):
            manager.apply_logical_operation(circuit, 'invalid')
            
    def test_performance_metrics(self):
        """Test performance metrics calculation."""
        manager = ErrorCorrectionManager(code_type='surface', distance=3)
        
        # Update metrics with sample data
        manager.update_metrics(
            logical_errors=5,
            physical_errors=20,
            total_corrections=100,
            successful_corrections=95
        )
        
        metrics = manager.get_metrics()
        self.assertIn('logical_error_rate', metrics)
        self.assertIn('physical_error_rate', metrics)
        self.assertIn('correction_success_rate', metrics)
        self.assertIn('overhead_ratio', metrics)
        
        # Verify metric calculations
        self.assertEqual(metrics['logical_error_rate'], 0.05)
        self.assertEqual(metrics['correction_success_rate'], 0.95)
        
    def test_resource_estimation(self):
        """Test resource requirement estimation."""
        manager = ErrorCorrectionManager(code_type='surface', distance=3)
        
        # Update metrics for estimation
        manager.update_metrics(
            logical_errors=5,
            physical_errors=20,
            total_corrections=100,
            successful_corrections=95
        )
        
        # Test resource estimation
        resources = manager.estimate_resource_requirements(
            target_logical_error_rate=1e-6
        )
        
        self.assertIn('code_distance', resources)
        self.assertIn('physical_qubits', resources)
        self.assertIn('syndrome_qubits', resources)
        
        # Verify resource calculations
        self.assertGreater(resources['code_distance'], 3)
        self.assertEqual(
            resources['physical_qubits'],
            resources['code_distance'] ** 2
        )
        
if __name__ == '__main__':
    unittest.main() 
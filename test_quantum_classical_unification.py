import unittest
import numpy as np
from pathlib import Path
import json
import os
from quantum_classical_unification import QuantumClassicalUnification
from sacred_solar import SacredSolar
from cosmic_awakening import CosmicAwakening
from quantum_draconic_guardian import QuantumDraconicGuardian
from quantum_healing import QuantumHealing
from divine_judgment import DivineJudgment

class TestQuantumClassicalUnification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running tests."""
        # Create output directory for test files
        cls.output_dir = Path("output")
        cls.output_dir.mkdir(exist_ok=True)
        
        # Initialize required components with reduced qubit count
        cls.guardian = QuantumDraconicGuardian(num_qubits=4)  # Reduced from 144
        cls.healing = QuantumHealing(cls.guardian)
        cls.judgment = DivineJudgment(cls.guardian, cls.healing)
        cls.solar = SacredSolar(cls.guardian, cls.healing, cls.judgment)
        cls.cosmic = CosmicAwakening(cls.solar, cls.guardian, cls.healing, cls.judgment)
        
        # Initialize the unification system
        cls.unification = QuantumClassicalUnification(
            cls.solar, cls.cosmic, cls.guardian, cls.healing, cls.judgment
        )
    
    def test_initialization(self):
        """Test initialization of QuantumClassicalUnification."""
        # Check physical constants
        self.assertIn('hbar', self.unification.constants)
        self.assertIn('G', self.unification.constants)
        self.assertIn('c', self.unification.constants)
        self.assertIn('kB', self.unification.constants)
        self.assertIn('m', self.unification.constants)
        self.assertIn('T', self.unification.constants)
        self.assertIn('lambda', self.unification.constants)
        self.assertIn('gamma', self.unification.constants)
        
        # Check operator parameters
        self.assertIn('quantum', self.unification.operator_params)
        self.assertIn('classical', self.unification.operator_params)
        self.assertIn('hybrid', self.unification.operator_params)
        
        # Check unification protocols
        self.assertIn('wavefunction_decoherence', self.unification.unification_protocols)
        self.assertIn('multiversal_lagrangian', self.unification.unification_protocols)
        self.assertIn('resonance_mathematics', self.unification.unification_protocols)
    
    def test_decoherence_matrix(self):
        """Test creation of decoherence matrix."""
        decoherence = self.unification._create_decoherence_matrix()
        
        # Check shape (reduced from 144x144 to 4x4)
        self.assertEqual(decoherence.shape, (4, 4))
        
        # Check values are real and non-negative
        self.assertTrue(np.all(np.isreal(decoherence)))
        self.assertTrue(np.all(decoherence >= 0))
    
    def test_lagrangian_circuit(self):
        """Test creation of lagrangian circuit."""
        circuit = self.unification._create_lagrangian_circuit()
        
        # Check circuit can be executed
        inputs = np.array([1.0] * 4)  # Reduced from 144
        weights = np.array([np.pi/4] * 4)  # Reduced from 144
        result = circuit(inputs, weights)
        
        # Check result length
        self.assertEqual(len(result), 4)  # Reduced from 144
    
    def test_resonance_matrix(self):
        """Test creation of resonance matrix."""
        resonance = self.unification._create_resonance_matrix()
        
        # Check shape (reduced from 144x144 to 4x4)
        self.assertEqual(resonance.shape, (4, 4))
        
        # Check values are real and non-negative
        self.assertTrue(np.all(np.isreal(resonance)))
        self.assertTrue(np.all(resonance >= 0))
    
    def test_quantum_resonance(self):
        """Test quantum resonance calculation."""
        # Test at resonance frequency
        omega0 = 0.5
        resonance = self.unification.quantum_resonance(omega0)
        
        # Check magnitude is maximum at resonance
        self.assertAlmostEqual(abs(resonance), 1.0, places=5)
        
        # Test off resonance
        omega_off = 1.0
        resonance_off = self.unification.quantum_resonance(omega_off)
        
        # Check magnitude is less than at resonance
        self.assertLess(abs(resonance_off), abs(resonance))
    
    def test_unification_circuit(self):
        """Test creation of unification circuit."""
        circuit = self.unification.create_unification_circuit()
        
        # Check circuit can be executed
        inputs = np.array([1.0] * 4)  # Reduced from 144
        weights = np.array([np.pi/4] * 4)  # Reduced from 144
        result = circuit(inputs, weights)
        
        # Check result length
        self.assertEqual(len(result), 4)  # Reduced from 144
    
    def test_unify(self):
        """Test unification process."""
        target = "TestTarget"
        record = self.unification.unify(target)
        
        # Check record structure
        self.assertEqual(record["target"], target)
        self.assertEqual(record["status"], "UNIFYING")
        self.assertIn("timestamp", record)
        self.assertIn("energy_level", record)
        self.assertIn("delta_x", record)
        self.assertIn("Gamma_obs", record)
        
        # Check values are reasonable
        self.assertGreater(record["energy_level"], 0)
        self.assertGreater(record["delta_x"], 0)
        self.assertGreater(record["Gamma_obs"], 0)
    
    def test_unification_frequency(self):
        """Test generation of unification frequency."""
        frequency = self.unification.generate_unification_frequency()
        
        # Check frequency is positive
        self.assertGreater(frequency, 0)
        
        # Check frequency is reasonable (Planck frequency scale)
        planck_freq = self.unification.constants['c']**5 / (self.unification.constants['hbar'] * self.unification.constants['G'])
        self.assertLess(frequency, planck_freq)
    
    def test_unification_field(self):
        """Test creation of unification field."""
        field = self.unification.create_unification_field()
        
        # Check shape (8 points for two tetrahedra)
        self.assertEqual(field.shape, (8, 3))
        
        # Check points form a valid tetrahedron
        # Calculate distances between points
        distances = []
        for i in range(4):
            for j in range(i+1, 4):
                dist = np.linalg.norm(field[i] - field[j])
                distances.append(dist)
        
        # All edges should have same length for regular tetrahedron
        self.assertTrue(np.allclose(distances, distances[0], rtol=1e-5))
    
    def test_unification_key(self):
        """Test generation of unification key."""
        key_path = self.unification.generate_unification_key()
        
        # Check file exists
        self.assertTrue(Path(key_path).exists())
        
        # Check file is a valid WAV file
        import wave
        with wave.open(key_path, 'rb') as wav_file:
            self.assertEqual(wav_file.getnchannels(), 1)
            self.assertEqual(wav_file.getsampwidth(), 2)
            self.assertEqual(wav_file.getframerate(), 44100)
    
    def test_save_and_load_unification_record(self):
        """Test saving and loading unification records."""
        # Create test record
        test_record = {
            "target": "TestTarget",
            "timestamp": "2023-01-01T00:00:00",
            "status": "TEST",
            "energy_level": 1.0,
            "delta_x": 1.0,
            "Gamma_obs": 1.0
        }
        
        # Save record
        filename = "test_unification_record.json"
        self.unification.save_unification_record(test_record, filename)
        
        # Check file exists
        record_path = self.output_dir / filename
        self.assertTrue(record_path.exists())
        
        # Load record
        loaded_record = self.unification.load_unification_record(filename)
        
        # Check loaded record matches original
        self.assertEqual(loaded_record["target"], test_record["target"])
        self.assertEqual(loaded_record["status"], test_record["status"])
        self.assertEqual(loaded_record["energy_level"], test_record["energy_level"])
        self.assertEqual(loaded_record["delta_x"], test_record["delta_x"])
        self.assertEqual(loaded_record["Gamma_obs"], test_record["Gamma_obs"])
        
        # Clean up
        record_path.unlink()

if __name__ == '__main__':
    unittest.main() 
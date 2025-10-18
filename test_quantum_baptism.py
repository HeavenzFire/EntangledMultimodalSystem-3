import unittest
import numpy as np
from pathlib import Path
import json
import os
from quantum_baptism import QuantumBaptism

class TestQuantumBaptism(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.baptism = QuantumBaptism()
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
    
    def test_initialization(self):
        """Test initialization of QuantumBaptism."""
        self.assertEqual(self.baptism.num_qubits, 144)
        self.assertEqual(self.baptism.base_frequency, 432)
        self.assertEqual(self.baptism.schumann_resonance, 7.83)
    
    def test_calculate_christ_light(self):
        """Test Christ light calculation."""
        light = self.baptism.calculate_christ_light()
        
        self.assertIsInstance(light, float)
        self.assertGreater(light, 0)
        
        # Test different frequencies
        light2 = self.baptism.calculate_christ_light(frequency=440)
        self.assertNotEqual(light, light2)
    
    def test_calculate_baptismal_formula(self):
        """Test baptismal formula calculation."""
        formula = self.baptism.calculate_baptismal_formula()
        
        self.assertIsInstance(formula, dict)
        self.assertTrue("christ_light" in formula)
        self.assertTrue("quantum_field" in formula)
        self.assertTrue("baptismal_power" in formula)
        
        # Test all components are positive
        for value in formula.values():
            self.assertGreater(value, 0)
    
    def test_baptize_quantum_state(self):
        """Test quantum state baptism."""
        state = np.array([1.0, 0.0])
        baptized_state = self.baptism.baptize_quantum_state(state)
        
        self.assertIsInstance(baptized_state, np.ndarray)
        self.assertEqual(baptized_state.shape, state.shape)
        
        # Test state normalization
        self.assertAlmostEqual(np.sum(np.abs(baptized_state)**2), 1.0)
    
    def test_create_baptismal_circuit(self):
        """Test baptismal circuit creation."""
        circuit = self.baptism.create_baptismal_circuit()
        
        self.assertIsNotNone(circuit)
        
        # Test circuit execution
        inputs = np.array([432] * 144)
        weights = np.array([np.pi/4] * 144)
        result = circuit(inputs, weights)
        self.assertEqual(len(result), 144)
    
    def test_generate_holy_water(self):
        """Test holy water generation."""
        water = self.baptism.generate_holy_water()
        
        self.assertIsInstance(water, dict)
        self.assertTrue("frequency" in water)
        self.assertTrue("purity" in water)
        self.assertTrue("blessing_strength" in water)
        
        # Test values are within expected ranges
        self.assertGreater(water["frequency"], 0)
        self.assertLessEqual(water["purity"], 1.0)
        self.assertGreaterEqual(water["purity"], 0.0)
        self.assertGreater(water["blessing_strength"], 0)
    
    def test_visualize_baptism_field(self):
        """Test baptism field visualization."""
        output_path = self.baptism.visualize_baptism_field()
        
        self.assertTrue(os.path.exists(output_path))
        self.assertTrue(output_path.endswith(".png"))
        
        # Clean up
        os.remove(output_path)
    
    def test_activate_baptism_system(self):
        """Test baptism system activation."""
        activation = self.baptism.activate_baptism_system()
        
        self.assertIsInstance(activation, dict)
        self.assertTrue("status" in activation)
        self.assertTrue("power_level" in activation)
        self.assertTrue("field_strength" in activation)
        
        # Test activation was successful
        self.assertEqual(activation["status"], "active")
        self.assertGreater(activation["power_level"], 0)
        self.assertGreater(activation["field_strength"], 0)
    
    def test_save_and_load_baptism_record(self):
        """Test baptism record saving and loading."""
        # Create test record
        record = {
            "timestamp": "2024-03-20T12:00:00",
            "power_level": 1.0,
            "field_strength": 1.0,
            "status": "active"
        }
        
        # Save record
        filename = "test_baptism.json"
        self.baptism.save_baptism_record(record, filename)
        
        # Verify file exists
        record_path = self.output_dir / filename
        self.assertTrue(record_path.exists())
        
        # Load record
        loaded_record = self.baptism.load_baptism_record(filename)
        
        # Verify loaded record matches original
        self.assertEqual(loaded_record["timestamp"], record["timestamp"])
        self.assertEqual(loaded_record["power_level"], record["power_level"])
        self.assertEqual(loaded_record["field_strength"], record["field_strength"])
        self.assertEqual(loaded_record["status"], record["status"])
        
        # Clean up
        os.remove(record_path)

if __name__ == '__main__':
    unittest.main() 
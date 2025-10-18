import unittest
import numpy as np
from pathlib import Path
import json
import os
from heavenly_army import HeavenlyArmy

class TestHeavenlyArmy(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.army = HeavenlyArmy()
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
    
    def test_initialization(self):
        """Test initialization of HeavenlyArmy."""
        self.assertEqual(self.army.commander_id, "HULSE-1992-QUANTUM")
        self.assertEqual(self.army.birth_year, 1992)
        self.assertEqual(self.army.birth_time, "10:30PM CST")
        self.assertEqual(self.army.birth_frequency, 432)
        self.assertEqual(len(self.army.legions), 8)
        self.assertTrue(all(archangel in self.army.legions for archangel in 
                          ["Michael", "Gabriel", "Metatron", "Raphael", "Uriel", 
                           "Chamuel", "Jophiel", "Zadkiel"]))
    
    def test_create_quantum_circuit(self):
        """Test quantum circuit creation."""
        circuit = self.army.create_quantum_circuit("Michael")
        self.assertIsNotNone(circuit)
        
        # Test circuit execution
        inputs = np.array([432] * 144)
        weights = np.array([np.pi/4] * 144)
        result = circuit(inputs, weights)
        self.assertEqual(len(result), 144)
    
    def test_deploy(self):
        """Test deployment functionality."""
        target = "TestTarget"
        deployment = self.army.deploy(target)
        
        self.assertEqual(deployment["commander"], self.army.commander_id)
        self.assertEqual(deployment["target"], target)
        self.assertTrue("timestamp" in deployment)
        self.assertTrue("archangels" in deployment)
        
        # Check archangel status updates
        for archangel, details in deployment["archangels"].items():
            self.assertTrue(target in details["status"])
            self.assertTrue("energy_level" in details)
    
    def test_generate_healing_frequency(self):
        """Test healing frequency generation."""
        condition = "test_condition"
        freq = self.army.generate_healing_frequency(condition)
        
        self.assertIsInstance(freq, float)
        self.assertGreater(freq, 0)
        
        # Test different conditions produce different frequencies
        freq2 = self.army.generate_healing_frequency("different_condition")
        self.assertNotEqual(freq, freq2)
    
    def test_create_merkabah_field(self):
        """Test Merkabah field creation."""
        merkabah = self.army.create_merkabah_field()
        
        self.assertIsInstance(merkabah, np.ndarray)
        self.assertEqual(merkabah.shape, (8, 3))  # 8 vertices, 3 coordinates each
        
        # Test field scaling
        radius = 2.0
        merkabah_scaled = self.army.create_merkabah_field(radius)
        self.assertTrue(np.all(np.abs(merkabah_scaled) <= radius))
    
    def test_generate_sonic_key(self):
        """Test sonic key generation."""
        freq = 432
        duration = 0.5
        output_path = self.army.generate_sonic_key(freq, duration)
        
        self.assertTrue(os.path.exists(output_path))
        self.assertTrue(output_path.endswith(".wav"))
        
        # Clean up
        os.remove(output_path)
    
    def test_save_and_load_deployment_record(self):
        """Test deployment record saving and loading."""
        # Create test deployment
        deployment = self.army.deploy("TestTarget")
        
        # Save record
        filename = "test_deployment.json"
        self.army.save_deployment_record(deployment, filename)
        
        # Verify file exists
        record_path = self.output_dir / filename
        self.assertTrue(record_path.exists())
        
        # Load record
        loaded_record = self.army.load_deployment_record(filename)
        
        # Verify loaded record matches original
        self.assertEqual(loaded_record["commander"], deployment["commander"])
        self.assertEqual(loaded_record["target"], deployment["target"])
        
        # Clean up
        os.remove(record_path)

if __name__ == '__main__':
    unittest.main() 
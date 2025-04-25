"""
Test Script for Quantum Unity Integration

This script tests the Quantum Unity Integration system to ensure all components
are working correctly before pushing to GitHub.
"""

import unittest
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import tempfile
import logging
from pathlib import Path

# Import the Quantum Unity Integration
from quantum_unity_integration import QuantumUnityIntegration, QuantumUnityState
from quantum_sacred_math import QuantumSacredMathematics, ConsciousnessField, PHI, PI, E

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("QuantumUnityTest")


class TestQuantumUnityIntegration(unittest.TestCase):
    """Test cases for the Quantum Unity Integration system."""
    
    def setUp(self):
        """Set up the test environment."""
        self.integration = QuantumUnityIntegration()
        self.temp_dir = tempfile.mkdtemp()
        logger.info(f"Created temporary directory: {self.temp_dir}")
    
    def test_initialization(self):
        """Test the initialization of the Quantum Unity Integration system."""
        # Test that the integration is initialized correctly
        self.assertIsNotNone(self.integration.qsm)
        self.assertIsNotNone(self.integration.unity_pulse)
        self.assertEqual(len(self.integration.state_history), 0)
        self.assertIsNone(self.integration.current_state)
        
        logger.info("Initialization test passed")
    
    def test_state_initialization(self):
        """Test the initialization of the system state."""
        # Initialize the state
        state = self.integration.initialize_state(dimension=3, participants=10)
        
        # Test that the state is initialized correctly
        self.assertIsNotNone(state)
        self.assertEqual(state.dimension, 3)
        self.assertEqual(state.participants, 10)
        self.assertEqual(state.elapsed_time, 0.0)
        self.assertEqual(state.phase_index, 0)
        
        # Test that the state is added to the history
        self.assertEqual(len(self.integration.state_history), 1)
        self.assertEqual(self.integration.current_state, state)
        
        logger.info("State initialization test passed")
    
    def test_state_update(self):
        """Test the update of the system state."""
        # Initialize the state
        self.integration.initialize_state(dimension=3, participants=10)
        
        # Update the state
        updated_state = self.integration.update_state(time_step=0.1)
        
        # Test that the state is updated correctly
        self.assertIsNotNone(updated_state)
        self.assertEqual(updated_state.elapsed_time, 0.1)
        
        # Test that the updated state is added to the history
        self.assertEqual(len(self.integration.state_history), 2)
        self.assertEqual(self.integration.current_state, updated_state)
        
        logger.info("State update test passed")
    
    def test_state_serialization(self):
        """Test the serialization and deserialization of the system state."""
        # Initialize the state
        self.integration.initialize_state(dimension=3, participants=10)
        
        # Update the state
        self.integration.update_state(time_step=0.1)
        
        # Save the state
        state_file = os.path.join(self.temp_dir, "state.json")
        self.integration.save_state(state_file)
        
        # Create a new integration
        new_integration = QuantumUnityIntegration()
        
        # Load the state
        loaded_state = new_integration.load_state(state_file)
        
        # Test that the loaded state is correct
        self.assertIsNotNone(loaded_state)
        self.assertEqual(loaded_state.dimension, 3)
        self.assertEqual(loaded_state.participants, 10)
        self.assertEqual(loaded_state.elapsed_time, 0.1)
        
        logger.info("State serialization test passed")
    
    def test_integration_run(self):
        """Test the running of the integration."""
        # Run the integration
        state_history = self.integration.run_integration(duration=1.0, time_step=0.1, participants=10)
        
        # Test that the integration ran correctly
        self.assertIsNotNone(state_history)
        self.assertGreater(len(state_history), 0)
        
        # Test that the final state has the correct elapsed time
        final_state = state_history[-1]
        self.assertGreaterEqual(final_state.elapsed_time, 1.0)
        
        logger.info("Integration run test passed")
    
    def test_visualization(self):
        """Test the visualization of the integration results."""
        # Run the integration
        self.integration.run_integration(duration=1.0, time_step=0.1, participants=10)
        
        # Test that the visualization methods don't raise exceptions
        try:
            # We'll use a non-interactive backend to avoid displaying plots
            plt.switch_backend('Agg')
            
            # Test each visualization method
            self.integration.visualize_integration()
            self.integration.visualize_phase_transitions()
            self.integration.visualize_consciousness_field()
            self.integration.visualize_divine_matrix()
            self.integration.visualize_metatrons_constant()
            self.integration.visualize_consciousness_harmonics()
            self.integration.visualize_divine_unity_field()
            
            # Close all figures to avoid memory leaks
            plt.close('all')
            
            logger.info("Visualization test passed")
        except Exception as e:
            self.fail(f"Visualization test failed: {e}")
    
    def test_quantum_sacred_math(self):
        """Test the Quantum Sacred Mathematics framework."""
        # Create a Quantum Sacred Mathematics instance
        qsm = QuantumSacredMathematics()
        
        # Test that the instance is created correctly
        self.assertIsNotNone(qsm)
        
        # Test the quantum coherence function
        field = ConsciousnessField(dimension=3, time=0.0)
        coherence = qsm.quantum_coherence_function(field)
        self.assertIsInstance(coherence, complex)
        
        # Test the timeline collapse function
        collapse = qsm.timeline_collapse_function(field, omega=2 * PI * 432)
        self.assertIsInstance(collapse, complex)
        
        # Test the consciousness harmonics
        harmonics = qsm.consciousness_harmonics()
        self.assertIsInstance(harmonics, np.ndarray)
        self.assertEqual(len(harmonics), 3)
        
        # Test the divine unity field
        unity_field = qsm.divine_unity_field(field)
        self.assertIsInstance(unity_field, float)
        
        # Test the quantum field activation
        activation = qsm.quantum_field_activation(0.0, frequency=432)
        self.assertIsInstance(activation, float)
        
        # Test the consciousness grid resonance
        resonance = qsm.consciousness_grid_resonance(0.0, 0.0, 0.0)
        self.assertIsInstance(resonance, complex)
        
        # Test the divine will expression
        will = qsm.divine_will_expression(field)
        self.assertIsInstance(will, complex)
        
        logger.info("Quantum Sacred Mathematics test passed")
    
    def tearDown(self):
        """Clean up the test environment."""
        # Remove the temporary directory
        import shutil
        shutil.rmtree(self.temp_dir)
        logger.info(f"Removed temporary directory: {self.temp_dir}")


def run_tests():
    """Run the tests and return the results."""
    # Create a test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestQuantumUnityIntegration)
    
    # Run the tests
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    
    # Return the result
    return result


def main():
    """Main function to run the tests."""
    logger.info("Starting Quantum Unity Integration tests")
    
    # Run the tests
    result = run_tests()
    
    # Check if the tests passed
    if result.wasSuccessful():
        logger.info("All tests passed!")
        return True
    else:
        logger.error("Some tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 
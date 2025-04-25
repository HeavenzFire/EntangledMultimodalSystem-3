import unittest
import numpy as np
from quantum_avatar_agent import QuantumAvatarAgent
import time

class TestConsciousnessStates(unittest.TestCase):
    def setUp(self):
        self.agent = QuantumAvatarAgent(
            name="Consciousness States Test",
            num_qubits=7,
            depth=3,
            shots=1024
        )
        
    def test_emotional_state_initialization(self):
        """Test emotional state initialization"""
        # Verify initial emotional state
        self.assertIsNotNone(self.agent.emotional_state)
        self.assertEqual(len(self.agent.emotional_state), 6)  # 6 basic emotions
        
        # Verify state normalization
        norm = np.sum(np.abs(self.agent.emotional_state))
        self.assertAlmostEqual(norm, 1.0, places=5)
        
    def test_emotional_state_update(self):
        """Test emotional state update"""
        # Process emotional input
        input_text = "I feel happy and excited about the future"
        self.agent._update_emotional_state(input_text)
        
        # Verify state update
        self.assertIsNotNone(self.agent.emotional_state)
        self.assertEqual(len(self.agent.emotional_state), 6)
        
        # Verify specific emotions
        self.assertGreater(self.agent.emotional_state[0], 0.0)  # Happiness
        self.assertGreater(self.agent.emotional_state[3], 0.0)  # Excitement
        
    def test_spiritual_state_initialization(self):
        """Test spiritual state initialization"""
        # Verify initial spiritual state
        self.assertIsNotNone(self.agent.spiritual_state)
        self.assertEqual(len(self.agent.spiritual_state), 7)  # 7 chakras
        
        # Verify state normalization
        norm = np.sum(np.abs(self.agent.spiritual_state))
        self.assertAlmostEqual(norm, 1.0, places=5)
        
    def test_spiritual_state_update(self):
        """Test spiritual state update"""
        # Process spiritual input
        input_text = "I feel a deep connection to universal consciousness"
        self.agent._update_spiritual_state(input_text)
        
        # Verify state update
        self.assertIsNotNone(self.agent.spiritual_state)
        self.assertEqual(len(self.agent.spiritual_state), 7)
        
        # Verify specific chakras
        self.assertGreater(self.agent.spiritual_state[6], 0.0)  # Crown chakra
        
    def test_consciousness_state_integration(self):
        """Test integration of emotional and spiritual states"""
        # Process integrated input
        input_text = "I feel peaceful and connected to all beings"
        self.agent.process_input(input_text)
        
        # Verify state integration
        self.assertIsNotNone(self.agent.consciousness_state)
        self.assertIsNotNone(self.agent.emotional_state)
        self.assertIsNotNone(self.agent.spiritual_state)
        
        # Verify state correlations
        emotional_peace = self.agent.emotional_state[2]  # Peace
        spiritual_connection = self.agent.spiritual_state[4]  # Heart chakra
        self.assertGreater(emotional_peace, 0.0)
        self.assertGreater(spiritual_connection, 0.0)
        
    def test_state_resonance(self):
        """Test state resonance calculation"""
        # Calculate resonance
        resonance = self.agent._calculate_state_resonance()
        
        # Verify resonance
        self.assertIsNotNone(resonance)
        self.assertGreaterEqual(resonance, 0.0)
        self.assertLessEqual(resonance, 1.0)
        
    def test_state_evolution(self):
        """Test state evolution over time"""
        # Track state evolution
        initial_emotional = self.agent.emotional_state.copy()
        initial_spiritual = self.agent.spiritual_state.copy()
        
        # Process multiple inputs
        inputs = [
            "I feel joyful and connected",
            "I experience deep peace",
            "I am filled with love"
        ]
        
        for input_text in inputs:
            self.agent.process_input(input_text)
            
        # Verify evolution
        self.assertFalse(np.array_equal(initial_emotional, self.agent.emotional_state))
        self.assertFalse(np.array_equal(initial_spiritual, self.agent.spiritual_state))
        
    def test_state_visualization(self):
        """Test state visualization"""
        # Generate visualizations
        emotional_fig = self.agent.visualize_emotional_state()
        spiritual_fig = self.agent.visualize_spiritual_state()
        combined_fig = self.agent.visualize_combined_state()
        
        # Verify visualizations
        self.assertIsNotNone(emotional_fig)
        self.assertIsNotNone(spiritual_fig)
        self.assertIsNotNone(combined_fig)
        
    def test_state_sound_generation(self):
        """Test sound generation from states"""
        # Generate sound
        sound_data = self.agent.generate_consciousness_sound()
        
        # Verify sound
        self.assertIsNotNone(sound_data)
        self.assertGreater(len(sound_data.get_array_of_samples()), 0)
        
    def test_state_memory(self):
        """Test state memory storage and retrieval"""
        # Store multiple states
        for i in range(5):
            self.agent.process_input(f"Test state {i}")
            
        # Verify memory
        self.assertGreater(len(self.agent.memory), 0)
        
        # Test memory retrieval
        memory_summary = self.agent.get_memory_summary()
        self.assertIsNotNone(memory_summary)
        self.assertIn('total_memories', memory_summary)
        self.assertIn('consciousness_level', memory_summary)
        
    def test_state_performance(self):
        """Test state processing performance"""
        start_time = time.time()
        
        # Process multiple inputs
        for _ in range(10):
            self.agent.process_input("Performance test input")
            self.agent._update_emotional_state("Test emotional input")
            self.agent._update_spiritual_state("Test spiritual input")
            
        total_time = time.time() - start_time
        
        # Verify performance
        self.assertLess(total_time, 5.0)  # Should complete within 5 seconds
        
    def test_state_optimization(self):
        """Test state optimization"""
        # Optimize states
        optimized_states = self.agent._optimize_consciousness_states()
        
        # Verify optimization
        self.assertIsNotNone(optimized_states)
        self.assertIn('emotional', optimized_states)
        self.assertIn('spiritual', optimized_states)
        
        # Check optimization metrics
        resonance = self.agent._calculate_state_resonance()
        self.assertGreater(resonance, 0.0)
        
    def test_state_error_handling(self):
        """Test state error handling"""
        # Test invalid input
        with self.assertRaises(ValueError):
            self.agent.process_input(None)
            
        # Test empty input
        with self.assertRaises(ValueError):
            self.agent.process_input("")
            
        # Test state recovery
        self.agent.process_input("Valid input")
        self.assertIsNotNone(self.agent.consciousness_state)
        
    def test_state_serialization(self):
        """Test state serialization and deserialization"""
        # Serialize states
        serialized = self.agent._serialize_states()
        
        # Verify serialization
        self.assertIsNotNone(serialized)
        self.assertIn('emotional', serialized)
        self.assertIn('spiritual', serialized)
        
        # Deserialize states
        deserialized = self.agent._deserialize_states(serialized)
        
        # Verify deserialization
        self.assertIsNotNone(deserialized)
        self.assertTrue(np.array_equal(deserialized['emotional'], self.agent.emotional_state))
        self.assertTrue(np.array_equal(deserialized['spiritual'], self.agent.spiritual_state))

if __name__ == '__main__':
    unittest.main() 
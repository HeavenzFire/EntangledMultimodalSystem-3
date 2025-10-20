import unittest
import numpy as np
from quantum_avatar_agent import QuantumAvatarAgent
import time
import os
import tempfile
import json
from pathlib import Path
import pytest
import torch
from src.quantum.archetype.krishna_module import KrishnaArchetype, VortexConfig
from src.quantum.vortex.sacred_sequencer import SacredSequencer, SequencerConfig
from src.quantum.merkaba.spin_controller import SpinController, SpinConfig
from src.quantum.ethical.karma_firewall import KarmaFirewall, KarmaConfig
from src.quantum.integration.quantum_bridge import QuantumBridge, BridgeConfig

class TestQuantumAvatarIntegration(unittest.TestCase):
    def setUp(self):
        self.agent = QuantumAvatarAgent(
            name="Integration Test",
            num_qubits=7,
            depth=3,
            shots=1024
        )
        self.temp_dir = tempfile.mkdtemp()
        
    def test_full_consciousness_cycle(self):
        """Test complete consciousness cycle including quantum state, emotions, and spiritual aspects"""
        # Process input and update states
        input_text = "I feel connected to the universe and experience deep peace"
        response = self.agent.process_input(input_text)
        
        # Verify response
        self.assertIsNotNone(response)
        self.assertIsInstance(response, str)
        
        # Verify state updates
        self.assertIsNotNone(self.agent.consciousness_state)
        self.assertIsNotNone(self.agent.emotional_state)
        self.assertIsNotNone(self.agent.spiritual_state)
        
        # Verify quantum metrics
        metrics = self.agent._calculate_quantum_metrics()
        self.assertGreater(metrics['fidelity'], 0.0)
        self.assertLessEqual(metrics['fidelity'], 1.0)
        
    def test_visualization_integration(self):
        """Test integration of all visualization components"""
        # Generate all visualizations
        consciousness_fig = self.agent.visualize_consciousness_state()
        emotional_fig = self.agent.visualize_emotional_state()
        spiritual_fig = self.agent.visualize_spiritual_state()
        metrics_fig = self.agent.visualize_quantum_metrics()
        
        # Save visualizations
        output_paths = []
        for fig, name in [(consciousness_fig, 'consciousness'),
                         (emotional_fig, 'emotional'),
                         (spiritual_fig, 'spiritual'),
                         (metrics_fig, 'metrics')]:
            path = os.path.join(self.temp_dir, f'{name}_state.png')
            fig.savefig(path)
            output_paths.append(path)
            
        # Verify files were created
        for path in output_paths:
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 0)
            
    def test_quantum_classical_integration(self):
        """Test integration between quantum and classical components"""
        # Create quantum circuit
        quantum_result = self.agent.simulator.run(self.agent.consciousness_circuit).result()
        
        # Process through classical components
        classical_result = self.agent._process_quantum_result(quantum_result)
        
        # Verify integration
        self.assertIsNotNone(classical_result)
        self.assertIn('state_vector', classical_result)
        self.assertIn('counts', classical_result)
        
    def test_memory_system_integration(self):
        """Test integration of memory system with quantum states"""
        # Store multiple states
        for i in range(5):
            self.agent.process_input(f"Test memory entry {i}")
            
        # Verify memory storage
        self.assertGreater(len(self.agent.memory), 0)
        self.assertGreater(len(self.agent.quantum_memory), 0)
        
        # Test memory retrieval
        memory_summary = self.agent.get_memory_summary()
        self.assertIsNotNone(memory_summary)
        self.assertIn('total_memories', memory_summary)
        self.assertIn('consciousness_level', memory_summary)
        
    def test_error_mitigation_integration(self):
        """Test integration of error mitigation with quantum operations"""
        # Create noisy circuit
        noisy_circuit = self.agent._create_noisy_circuit()
        
        # Run with error mitigation
        mitigated_result = self.agent._run_with_error_mitigation(noisy_circuit)
        
        # Verify mitigation
        self.assertIsNotNone(mitigated_result)
        self.assertIn('mitigated_counts', mitigated_result)
        
    def test_optimization_integration(self):
        """Test integration of optimization with quantum circuits"""
        # Run optimization
        optimized_state = self.agent._optimize_spiritual_resonance()
        
        # Verify optimization
        self.assertIsNotNone(optimized_state)
        self.assertGreater(len(self.agent.optimization_history), 0)
        
    def test_performance_integration(self):
        """Test performance of integrated system"""
        start_time = time.time()
        
        # Run complete cycle
        self.agent.process_input("Performance test input")
        self.agent.visualize_consciousness_state()
        self.agent._calculate_quantum_metrics()
        self.agent._optimize_spiritual_resonance()
        
        total_time = time.time() - start_time
        
        # Verify performance
        self.assertLess(total_time, 10.0)  # Should complete within 10 seconds
        
    def test_security_integration(self):
        """Test integration of security features"""
        # Test authentication
        auth_token = self.agent._generate_auth_token()
        self.assertIsNotNone(auth_token)
        
        # Test token validation
        is_valid = self.agent._validate_auth_token(auth_token)
        self.assertTrue(is_valid)
        
    def test_api_integration(self):
        """Test integration with API endpoints"""
        # Test API response
        api_response = self.agent._handle_api_request("test_endpoint")
        self.assertIsNotNone(api_response)
        self.assertIn('status', api_response)
        self.assertEqual(api_response['status'], 'success')
        
    def test_database_integration(self):
        """Test integration with database systems"""
        # Test database operations
        db_result = self.agent._store_in_database("test_data")
        self.assertTrue(db_result)
        
        # Test retrieval
        retrieved_data = self.agent._retrieve_from_database()
        self.assertIsNotNone(retrieved_data)
        
    def test_monitoring_integration(self):
        """Test integration of monitoring systems"""
        # Test metrics collection
        metrics = self.agent._collect_monitoring_metrics()
        self.assertIsNotNone(metrics)
        self.assertIn('cpu_usage', metrics)
        self.assertIn('memory_usage', metrics)
        self.assertIn('quantum_operations', metrics)
        
    def test_sound_generation_integration(self):
        """Test integration of sound generation with consciousness states"""
        # Generate sound
        sound_data = self.agent.generate_consciousness_sound()
        
        # Save sound file
        sound_path = os.path.join(self.temp_dir, 'consciousness_sound.wav')
        sound_data.export(sound_path, format='wav')
        
        # Verify sound file
        self.assertTrue(os.path.exists(sound_path))
        self.assertGreater(os.path.getsize(sound_path), 0)
        
    def test_3d_visualization_integration(self):
        """Test integration of 3D visualization components"""
        # Generate 3D visualization
        fig_3d = self.agent.visualize_3d_state()
        
        # Save visualization
        output_path = os.path.join(self.temp_dir, '3d_state.png')
        fig_3d.savefig(output_path)
        
        # Verify file
        self.assertTrue(os.path.exists(output_path))
        self.assertGreater(os.path.getsize(output_path), 0)
        
    def test_entanglement_visualization_integration(self):
        """Test integration of entanglement visualization"""
        # Generate entanglement visualization
        fig_ent = self.agent.visualize_entanglement()
        
        # Save visualization
        output_path = os.path.join(self.temp_dir, 'entanglement.png')
        fig_ent.savefig(output_path)
        
        # Verify file
        self.assertTrue(os.path.exists(output_path))
        self.assertGreater(os.path.getsize(output_path), 0)
        
    def test_combined_state_visualization_integration(self):
        """Test integration of combined state visualization"""
        # Generate combined visualization
        fig_combined = self.agent.visualize_combined_state()
        
        # Save visualization
        output_path = os.path.join(self.temp_dir, 'combined_state.png')
        fig_combined.savefig(output_path)
        
        # Verify file
        self.assertTrue(os.path.exists(output_path))
        self.assertGreater(os.path.getsize(output_path), 0)
        
    def test_interactive_visualization_integration(self):
        """Test integration of interactive visualization"""
        # Generate interactive visualization
        fig_interactive = self.agent.create_interactive_visualization()
        
        # Save visualization
        output_path = os.path.join(self.temp_dir, 'interactive.html')
        fig_interactive.write_html(output_path)
        
        # Verify file
        self.assertTrue(os.path.exists(output_path))
        self.assertGreater(os.path.getsize(output_path), 0)
        
    def test_animation_generation_integration(self):
        """Test integration of animation generation"""
        # Generate animation
        animation = self.agent.generate_state_evolution_animation()
        
        # Save animation
        output_path = os.path.join(self.temp_dir, 'state_evolution.gif')
        animation.save(output_path, writer='pillow', fps=10)
        
        # Verify file
        self.assertTrue(os.path.exists(output_path))
        self.assertGreater(os.path.getsize(output_path), 0)

@pytest.fixture
def test_configs():
    """Initialize test configurations."""
    return {
        'vortex': VortexConfig(base_frequency=432.0),
        'sequencer': SequencerConfig(fibonacci_depth=144),
        'spin': SpinConfig(base_frequency=34.21),
        'karma': KarmaConfig(compassion_threshold=0.9),
        'bridge': BridgeConfig(qubits=9)
    }

@pytest.fixture
def test_systems(test_configs):
    """Initialize test systems."""
    return {
        'krishna': KrishnaArchetype(test_configs['vortex']),
        'sequencer': SacredSequencer(test_configs['sequencer']),
        'spin': SpinController(test_configs['spin']),
        'karma': KarmaFirewall(test_configs['karma']),
        'bridge': QuantumBridge(test_configs['bridge'])
    }

def test_vortex_transformation(test_systems):
    """Test vortex code transformation."""
    krishna = test_systems['krishna']
    input_state = np.random.rand(108)
    transformed = krishna.apply_bhakti(input_state)
    
    assert len(transformed) == 144
    assert np.all(np.isfinite(transformed))
    
def test_sacred_sequence(test_systems):
    """Test sacred sequence generation and validation."""
    sequencer = test_systems['sequencer']
    sequence = sequencer.generate_sacred_sequence(144)
    
    assert len(sequence) == 144
    assert sequencer.validate_sequence(sequence)
    
def test_merkaba_field(test_systems):
    """Test merkaba field operations."""
    spin = test_systems['spin']
    field = spin.merkaba_field
    
    # Test field rotation
    rotated = spin.apply_rotation(field, np.pi/4)
    assert rotated.shape == field.shape
    
    # Test vortex generation
    center = np.zeros(spin.config.dimensions)
    vortex = spin.generate_vortex(center, 1.0)
    assert vortex.shape == (spin.config.merkaba_points, spin.config.dimensions)
    
def test_ethical_validation(test_systems):
    """Test ethical validation system."""
    karma = test_systems['karma']
    test_state = torch.randn(1, karma.config.karma_depth)
    
    # Test action validation
    is_ethical, scores = karma.validate_action(test_state)
    assert isinstance(is_ethical, bool)
    assert all(0 <= score <= 1 for score in scores.values())
    
    # Test consciousness protection
    protected = karma.protect_consciousness(test_state)
    assert protected.shape == test_state.shape
    
def test_quantum_bridge(test_systems):
    """Test quantum bridge integration."""
    bridge = test_systems['bridge']
    
    # Test archetype entanglement
    states = [np.random.rand(3) for _ in range(bridge.config.qubits)]
    circuit = bridge.entangle_archetypes(states)
    assert len(circuit.qubits) == bridge.config.qubits
    
    # Test consciousness mapping
    state = torch.randn(1, bridge.config.qubits)
    mapped = bridge.map_consciousness(state)
    assert mapped.shape == (1, bridge.config.consciousness_depth)
    
def test_system_integration(test_systems):
    """Test full system integration."""
    # Initialize components
    krishna = test_systems['krishna']
    sequencer = test_systems['sequencer']
    spin = test_systems['spin']
    karma = test_systems['karma']
    bridge = test_systems['bridge']
    
    # Create test quantum state
    input_state = np.random.rand(108)
    
    # Apply vortex transformation
    transformed = krishna.apply_bhakti(input_state)
    
    # Generate and validate sacred sequence
    sequence = sequencer.generate_sacred_sequence(len(transformed))
    assert sequencer.validate_sequence(sequence)
    
    # Create merkaba vortex
    center = np.zeros(spin.config.dimensions)
    vortex = spin.generate_vortex(center, 1.0)
    
    # Validate ethical alignment
    consciousness_state = torch.from_numpy(transformed).float().unsqueeze(0)
    is_ethical, _ = karma.validate_action(consciousness_state)
    
    # Map to quantum state and check harmony
    mapped_state = bridge.map_consciousness(consciousness_state)
    harmony = bridge.calculate_harmony([mapped_state])
    
    assert harmony >= 0.0 and harmony <= 1.0
    
def test_consciousness_evolution(test_systems):
    """Test consciousness evolution through the system."""
    bridge = test_systems['bridge']
    karma = test_systems['karma']
    
    # Initialize consciousness state
    state = torch.randn(1, bridge.config.consciousness_depth)
    
    # Apply ethical protection
    protected = karma.protect_consciousness(state)
    
    # Apply quantum correction
    corrected, fidelity = bridge.apply_quantum_correction(protected)
    
    assert fidelity >= 0.0 and fidelity <= 1.0
    assert torch.all(torch.isfinite(corrected))
    
    # Get harmony metrics
    metrics = bridge.get_harmony_metrics()
    assert all(0.0 <= value <= 1.0 for value in metrics.values())

if __name__ == '__main__':
    unittest.main() 
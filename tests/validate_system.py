import pytest
import numpy as np
from src.quantum.synthesis.quantum_sacred import (
    QuantumSacredSynthesis,
    SacredConfig,
    QuantumState,
    VortexHistoryBuffer,
    VortexPrimes
)
from src.quantum.synthesis.visualization import QuantumSacredVisualizer

class SystemValidator:
    """Comprehensive validation system for quantum-sacred synthesis"""
    
    def __init__(self):
        self.config = SacredConfig()
        self.synthesis = QuantumSacredSynthesis(self.config)
        self.visualizer = QuantumSacredVisualizer(self.synthesis)
        
    def validate_state_transitions(self):
        """Validate state transition system"""
        print("\nValidating State Transitions...")
        
        # Test initial state
        assert self.synthesis.current_state == QuantumState.DISSONANT
        
        # Test transition matrix initialization
        matrix = self.synthesis.transition_matrix
        assert matrix.shape == (5, 5)  # 5 states
        assert np.all(matrix >= 0) and np.all(matrix <= 1)
        assert np.allclose(np.sum(matrix, axis=1), 1.0)  # Row normalization
        
        # Test transition matrix update
        self.synthesis.update_transition_matrix(0.8, 0.2)
        new_matrix = self.synthesis.transition_matrix
        assert not np.array_equal(matrix, new_matrix)
        assert np.all(new_matrix >= 0.7) and np.all(new_matrix <= 0.8)
        
        print("✓ State transition validation complete")
        
    def validate_dissonance_resolution(self):
        """Validate dissonance resolution protocol"""
        print("\nValidating Dissonance Resolution...")
        
        # Set initial state
        self.synthesis.current_state = QuantumState.DISSONANT
        self.synthesis.dissonance_cycles = 0
        
        # Test resolution
        self.synthesis.resolve_dissonance()
        
        # Verify state transition
        assert self.synthesis.current_state == QuantumState.RESONANT
        assert self.synthesis.dissonance_cycles == 0
        
        # Test emergency protocol
        self.synthesis.current_state = QuantumState.DISSONANT
        self.synthesis.dissonance_cycles = 145  # Exceed max history
        
        self.synthesis.resolve_dissonance()
        assert self.synthesis.current_state == QuantumState.MERKABA
        assert self.synthesis.dissonance_cycles == 0
        
        print("✓ Dissonance resolution validation complete")
        
    def validate_harmonic_resonance(self):
        """Validate harmonic resonance system"""
        print("\nValidating Harmonic Resonance...")
        
        # Test Christos harmonic generation
        pattern = self.visualizer._apply_christos_harmonic()
        assert pattern.shape == (12,)
        assert np.all(np.abs(pattern) <= 1.0)
        assert np.iscomplexobj(pattern)
        
        # Test resonance strength
        resonance = self.synthesis._apply_christos_harmonic()
        assert 0 <= resonance <= 1.0
        
        print("✓ Harmonic resonance validation complete")
        
    def validate_merkaba_field(self):
        """Validate merkaba field operations"""
        print("\nValidating Merkaba Field...")
        
        # Test vertex generation
        vertices = self.visualizer._generate_merkaba_vertices(self.config.phi_resonance)
        assert vertices.shape == (8, 3)
        assert np.allclose(np.abs(vertices), self.config.phi_resonance)
        
        # Test rotation
        rotated = self.visualizer._rotate_vertices(vertices, 45.0)
        assert rotated.shape == vertices.shape
        assert not np.array_equal(vertices, rotated)
        assert np.allclose(np.linalg.norm(rotated, axis=1),
                          np.linalg.norm(vertices, axis=1))
        
        print("✓ Merkaba field validation complete")
        
    def validate_memory_management(self):
        """Validate memory management system"""
        print("\nValidating Memory Management...")
        
        buffer = VortexHistoryBuffer(self.config)
        
        # Test state addition
        state = {"a": 0.9, "b": 0.1}
        buffer.add_state(state)
        assert len(buffer.buffer) == 1
        
        # Test entropy calculation
        entropy = buffer._calculate_entropy(state)
        assert isinstance(entropy, float)
        assert entropy >= 0
        
        # Test entropy purging
        high_entropy_state = {"a": 0.5, "b": 0.5}
        buffer.add_state(high_entropy_state)
        buffer.purge_entropy()
        assert len(buffer.buffer) == 1
        assert buffer.buffer[0] == state
        
        print("✓ Memory management validation complete")
        
    def validate_visualization(self):
        """Validate visualization system"""
        print("\nValidating Visualization System...")
        
        # Test initialization
        assert self.visualizer.fig is not None
        assert len(self.visualizer.fig.axes) == 4
        
        # Test update
        self.visualizer.update(0)
        assert self.visualizer.ax1.collections  # Merkaba field
        assert self.visualizer.ax2.images  # State transitions
        assert self.visualizer.ax3.lines  # Harmonic resonance
        assert self.visualizer.ax4.collections  # Christos grid
        
        # Test animation
        anim = self.visualizer.animate(frames=10)
        assert anim is not None
        assert anim.fig == self.visualizer.fig
        
        print("✓ Visualization validation complete")
        
    def validate_system_integration(self):
        """Validate system integration"""
        print("\nValidating System Integration...")
        
        # Test full system workflow
        self.synthesis.update_transition_matrix(0.8, 0.2)
        self.synthesis.merkaba_rotation = 45.0
        
        # Verify state updates
        assert self.synthesis.transition_matrix is not None
        assert self.synthesis.merkaba_rotation == 45.0
        
        # Test visualization update
        self.visualizer.update(0)
        assert self.visualizer.ax1.collections
        assert self.visualizer.ax2.images
        
        print("✓ System integration validation complete")
        
    def run_validation(self):
        """Run all validation tests"""
        print("Starting Quantum-Sacred Synthesis System Validation...")
        print("=" * 50)
        
        self.validate_state_transitions()
        self.validate_dissonance_resolution()
        self.validate_harmonic_resonance()
        self.validate_merkaba_field()
        self.validate_memory_management()
        self.validate_visualization()
        self.validate_system_integration()
        
        print("\n" + "=" * 50)
        print("Validation Complete - All Systems Operational")
        print("Quantum-Sacred Synthesis System Ready for Deployment")

def test_system_validation():
    """Run system validation tests"""
    validator = SystemValidator()
    validator.run_validation()

if __name__ == "__main__":
    # Run validation
    validator = SystemValidator()
    validator.run_validation()
    
    # Run pytest
    pytest.main([__file__]) 
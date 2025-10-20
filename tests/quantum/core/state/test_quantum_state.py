import pytest
import numpy as np
from src.quantum.core.state.quantum_state import (
    QuantumState,
    StateManipulation,
    StateResult
)
from src.quantum.core.gates.quantum_gates import (
    HadamardGate,
    PauliXGate,
    PauliYGate,
    PauliZGate,
    PhaseGate
)

class TestQuantumState:
    @pytest.fixture
    def state_manipulator(self):
        return StateManipulation()
    
    @pytest.fixture
    def test_state(self):
        return QuantumState(
            amplitude=1.0,
            phase=np.pi/4,
            error_rate=0.001
        )
    
    def test_state_initialization(self, state_manipulator):
        """Test quantum state initialization"""
        state = state_manipulator.initialize_state(
            amplitude=1.0,
            phase=np.pi/4
        )
        
        assert isinstance(state, QuantumState)
        assert state.amplitude == 1.0
        assert state.phase == np.pi/4
        assert state.error_rate < 0.1
    
    def test_state_preparation(self, state_manipulator):
        """Test quantum state preparation"""
        # Prepare |0⟩ state
        zero_state = state_manipulator.prepare_zero_state()
        assert zero_state.amplitude == 1.0
        assert zero_state.phase == 0.0
        
        # Prepare |1⟩ state
        one_state = state_manipulator.prepare_one_state()
        assert one_state.amplitude == 1.0
        assert one_state.phase == np.pi
        
        # Prepare |+⟩ state
        plus_state = state_manipulator.prepare_plus_state()
        assert np.isclose(plus_state.amplitude, 1/np.sqrt(2))
        assert plus_state.phase == 0.0
    
    def test_state_manipulation(self, state_manipulator, test_state):
        """Test quantum state manipulation"""
        # Apply Hadamard gate
        result = state_manipulator.apply_gate(test_state, HadamardGate())
        assert isinstance(result, StateResult)
        assert result.success
        assert result.final_state.amplitude != test_state.amplitude
        assert result.final_state.phase != test_state.phase
        
        # Apply Phase gate
        result = state_manipulator.apply_gate(test_state, PhaseGate(np.pi/2))
        assert result.success
        assert result.final_state.phase != test_state.phase
    
    def test_state_measurement(self, state_manipulator, test_state):
        """Test quantum state measurement"""
        measurement = state_manipulator.measure_state(test_state)
        
        assert isinstance(measurement, tuple)
        assert len(measurement) == 2
        assert measurement[0] in [0, 1]  # Measurement outcome
        assert 0 <= measurement[1] <= 1  # Probability
    
    def test_state_evolution(self, state_manipulator, test_state):
        """Test quantum state evolution"""
        result = state_manipulator.evolve_state(test_state, time=1.0)
        
        assert isinstance(result, StateResult)
        assert result.success
        assert result.final_state.amplitude != test_state.amplitude
        assert result.final_state.phase != test_state.phase
        assert result.error_rate < 0.1
    
    def test_error_correction(self, state_manipulator, test_state):
        """Test error correction"""
        # Introduce error
        test_state.error_rate = 0.2
        
        result = state_manipulator.correct_errors(test_state)
        
        assert isinstance(result, StateResult)
        assert result.success
        assert result.error_rate < 0.1
        assert result.correction_applied
    
    def test_state_robustness(self, state_manipulator, test_state):
        """Test state manipulation robustness"""
        results = []
        for _ in range(5):
            result = state_manipulator.evolve_state(test_state, time=1.0)
            results.append(result.final_state)
        
        # Check consistency of results
        for i in range(1, len(results)):
            assert np.isclose(results[i].amplitude, results[0].amplitude, atol=1e-6)
            assert np.isclose(results[i].phase, results[0].phase, atol=1e-6)
    
    def test_error_handling(self, state_manipulator):
        """Test error handling"""
        # Test with invalid state
        with pytest.raises(Exception):
            state_manipulator.evolve_state(None, time=1.0)
        
        # Test with invalid parameters
        with pytest.raises(Exception):
            state_manipulator.apply_gate(None, None)
    
    def test_state_quality(self, state_manipulator, test_state):
        """Test state quality"""
        result = state_manipulator.evolve_state(test_state, time=1.0)
        
        assert result.error_rate < 0.1  # Should have low error rate
        assert result.final_state.coherence > 0.8  # Should maintain high coherence
        assert result.verification_passed  # Should pass verification
    
    def test_state_efficiency(self, state_manipulator, test_state):
        """Test state manipulation efficiency"""
        import time
        
        start_time = time.time()
        result = state_manipulator.evolve_state(test_state, time=1.0)
        manipulation_time = time.time() - start_time
        
        assert result.success
        assert manipulation_time < 1.0  # Should manipulate within reasonable time
        assert result.error_rate < 0.1
    
    def test_state_preservation(self, state_manipulator, test_state):
        """Test state preservation during manipulation"""
        initial_amplitude = test_state.amplitude
        initial_phase = test_state.phase
        
        result = state_manipulator.evolve_state(test_state, time=1.0)
        
        assert result.success
        assert not np.isclose(result.final_state.amplitude, initial_amplitude)
        assert not np.isclose(result.final_state.phase, initial_phase)
    
    def test_state_consistency(self, state_manipulator, test_state):
        """Test consistency of state manipulation results"""
        results = []
        for _ in range(10):
            result = state_manipulator.evolve_state(test_state, time=1.0)
            results.append(result.final_state)
        
        # Check consistency of results
        for i in range(1, len(results)):
            assert np.isclose(results[i].amplitude, results[0].amplitude, atol=1e-6)
            assert np.isclose(results[i].phase, results[0].phase, atol=1e-6)
    
    def test_gate_sequence(self, state_manipulator, test_state):
        """Test application of gate sequence"""
        gates = [
            HadamardGate(),
            PhaseGate(np.pi/2),
            PauliXGate()
        ]
        
        result = state_manipulator.apply_gate_sequence(test_state, gates)
        
        assert isinstance(result, StateResult)
        assert result.success
        assert result.error_rate < 0.1
        assert result.final_state.amplitude != test_state.amplitude
        assert result.final_state.phase != test_state.phase
    
    def test_state_verification(self, state_manipulator, test_state):
        """Test state verification"""
        result = state_manipulator.evolve_state(test_state, time=1.0)
        verification = state_manipulator.verify_state(result.final_state)
        
        assert verification.success
        assert verification.verification_passed
        assert verification.error_rate < 0.1
        assert verification.coherence > 0.8
    
    def test_multiple_operations(self, state_manipulator, test_state):
        """Test multiple state operations"""
        # Perform multiple operations
        current_state = test_state
        for _ in range(3):
            result = state_manipulator.evolve_state(current_state, time=1.0)
            current_state = result.final_state
        
        # Verify final state quality
        assert current_state.coherence > 0.8
        assert current_state.error_rate < 0.1
        assert 0 <= current_state.phase <= 2 * np.pi
        assert current_state.amplitude > 0 
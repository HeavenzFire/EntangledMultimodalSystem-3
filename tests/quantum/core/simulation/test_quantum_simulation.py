import pytest
import numpy as np
from src.quantum.core.simulation.quantum_simulation import (
    QuantumCircuit,
    SimulationResult
)
from src.quantum.core.quantum_state import QuantumState
from src.quantum.core.gates.quantum_gates import (
    HadamardGate,
    PauliXGate,
    PauliYGate,
    PauliZGate,
    CNOTGate,
    PhaseGate
)

class TestQuantumSimulation:
    @pytest.fixture
    def circuit(self):
        return QuantumCircuit(num_qubits=4)
    
    @pytest.fixture
    def test_state(self):
        return QuantumState(
            amplitude=1.0,
            phase=np.pi/4,
            error_rate=0.001
        )
    
    def test_circuit_initialization(self, circuit):
        """Test quantum circuit initialization"""
        assert circuit.num_qubits == 4
        assert len(circuit.qubits) == 4
        assert circuit.gates == []
        assert circuit.measurements == []
    
    def test_gate_application(self, circuit):
        """Test application of quantum gates"""
        # Apply Hadamard gate
        circuit.apply_gate(HadamardGate(), 0)
        assert len(circuit.gates) == 1
        assert isinstance(circuit.gates[0], HadamardGate)
        
        # Apply CNOT gate
        circuit.apply_gate(CNOTGate(), (0, 1))
        assert len(circuit.gates) == 2
        assert isinstance(circuit.gates[1], CNOTGate)
    
    def test_state_evolution(self, circuit):
        """Test quantum state evolution"""
        # Prepare initial state
        circuit.apply_gate(HadamardGate(), 0)
        circuit.apply_gate(CNOTGate(), (0, 1))
        
        result = circuit.evolve_state()
        
        assert isinstance(result, SimulationResult)
        assert result.success
        assert isinstance(result.final_state, QuantumState)
        assert result.error_rate < 0.1
    
    def test_measurement(self, circuit):
        """Test quantum measurement"""
        # Prepare state
        circuit.apply_gate(HadamardGate(), 0)
        
        # Perform measurement
        measurement = circuit.measure(0)
        
        assert isinstance(measurement, tuple)
        assert len(measurement) == 2
        assert measurement[0] in [0, 1]  # Measurement outcome
        assert 0 <= measurement[1] <= 1  # Probability
    
    def test_error_correction(self, circuit):
        """Test error correction in circuit"""
        # Introduce error
        circuit.apply_gate(PauliXGate(), 0)  # Bit flip error
        
        result = circuit.correct_errors()
        
        assert isinstance(result, SimulationResult)
        assert result.success
        assert result.error_rate < 0.1
        assert result.correction_applied
    
    def test_circuit_robustness(self, circuit):
        """Test circuit robustness"""
        results = []
        for _ in range(5):
            circuit.apply_gate(HadamardGate(), 0)
            result = circuit.evolve_state()
            results.append(result.final_state)
            circuit.reset()
        
        # Check consistency of results
        for i in range(1, len(results)):
            assert np.isclose(results[i].amplitude, results[0].amplitude, atol=1e-6)
            assert np.isclose(results[i].phase, results[0].phase, atol=1e-6)
    
    def test_error_handling(self, circuit):
        """Test error handling"""
        # Test with invalid qubit index
        with pytest.raises(Exception):
            circuit.apply_gate(HadamardGate(), -1)
        
        # Test with invalid gate
        with pytest.raises(Exception):
            circuit.apply_gate(None, 0)
    
    def test_circuit_quality(self, circuit):
        """Test circuit quality"""
        circuit.apply_gate(HadamardGate(), 0)
        result = circuit.evolve_state()
        
        assert result.error_rate < 0.1  # Should have low error rate
        assert result.final_state.coherence > 0.8  # Should maintain high coherence
        assert result.verification_passed  # Should pass verification
    
    def test_circuit_efficiency(self, circuit):
        """Test circuit efficiency"""
        import time
        
        # Build a simple circuit
        circuit.apply_gate(HadamardGate(), 0)
        circuit.apply_gate(CNOTGate(), (0, 1))
        
        start_time = time.time()
        result = circuit.evolve_state()
        execution_time = time.time() - start_time
        
        assert result.success
        assert execution_time < 1.0  # Should execute within reasonable time
        assert result.error_rate < 0.1
    
    def test_state_preservation(self, circuit):
        """Test state preservation during circuit operations"""
        initial_state = circuit.get_state()
        
        circuit.apply_gate(HadamardGate(), 0)
        result = circuit.evolve_state()
        
        assert result.success
        assert not np.isclose(result.final_state.amplitude, initial_state.amplitude)
        assert not np.isclose(result.final_state.phase, initial_state.phase)
    
    def test_circuit_consistency(self, circuit):
        """Test consistency of circuit results"""
        results = []
        for _ in range(10):
            circuit.apply_gate(HadamardGate(), 0)
            result = circuit.evolve_state()
            results.append(result.final_state)
            circuit.reset()
        
        # Check consistency of results
        for i in range(1, len(results)):
            assert np.isclose(results[i].amplitude, results[0].amplitude, atol=1e-6)
            assert np.isclose(results[i].phase, results[0].phase, atol=1e-6)
    
    def test_gate_sequence(self, circuit):
        """Test application of gate sequence"""
        gates = [
            (HadamardGate(), 0),
            (PauliXGate(), 1),
            (CNOTGate(), (0, 1)),
            (PhaseGate(np.pi/2), 0)
        ]
        
        for gate, qubit in gates:
            circuit.apply_gate(gate, qubit)
        
        assert len(circuit.gates) == len(gates)
        result = circuit.evolve_state()
        assert result.success
        assert result.error_rate < 0.1
    
    def test_circuit_verification(self, circuit):
        """Test circuit verification"""
        circuit.apply_gate(HadamardGate(), 0)
        result = circuit.evolve_state()
        verification = circuit.verify_state(result.final_state)
        
        assert verification.success
        assert verification.verification_passed
        assert verification.error_rate < 0.1
        assert verification.coherence > 0.8
    
    def test_multiple_operations(self, circuit):
        """Test multiple circuit operations"""
        # Perform multiple operations
        circuit.apply_gate(HadamardGate(), 0)
        circuit.apply_gate(CNOTGate(), (0, 1))
        circuit.apply_gate(PhaseGate(np.pi/2), 0)
        
        result = circuit.evolve_state()
        
        # Verify final state quality
        assert result.final_state.coherence > 0.8
        assert result.final_state.error_rate < 0.1
        assert 0 <= result.final_state.phase <= 2 * np.pi
        assert result.final_state.amplitude > 0 
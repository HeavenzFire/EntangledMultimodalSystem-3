import pytest
import numpy as np
from quantum_consciousness_integrator import QuantumConsciousnessIntegrator

def test_consciousness_circuit_creation():
    qci = QuantumConsciousnessIntegrator(num_qubits=7, depth=5)
    input_state = np.array([1] + [0] * (2**7 - 1))
    
    circuit = qci.create_consciousness_circuit(input_state)
    
    # Verify circuit properties
    assert circuit.num_qubits == 7
    assert circuit.num_clbits == 7
    assert len(circuit.data) > 0
    
    # Verify chakra frequency initialization
    for i, freq in enumerate(qci.chakra_frequencies.values()):
        angle = 2 * np.pi * freq / 1000
        assert any(gate.name == 'ry' and gate.params[0] == angle for gate in circuit.data)
        assert any(gate.name == 'rz' and gate.params[0] == angle * qci.golden_ratio for gate in circuit.data)

def test_consciousness_metrics_calculation():
    qci = QuantumConsciousnessIntegrator()
    input_state = np.array([1] + [0] * (2**7 - 1))
    
    metrics = qci.run_consciousness_circuit(input_state)
    
    # Verify metrics are within expected ranges
    assert 0 <= metrics['chakra_alignment'] <= 1
    assert 0 <= metrics['consciousness_coherence'] <= 1
    assert 0 <= metrics['spiritual_resonance'] <= 1
    
    # Verify metrics sum to a reasonable value
    total = sum(metrics.values())
    assert 0 <= total <= 3

def test_chakra_focus():
    qci = QuantumConsciousnessIntegrator()
    input_state = np.array([1] + [0] * (2**7 - 1))
    
    # Test all chakras
    all_metrics = qci.run_consciousness_circuit(input_state, chakra_focus='all')
    
    # Test specific chakra
    crown_metrics = qci.run_consciousness_circuit(input_state, chakra_focus='crown')
    
    # Verify crown chakra metrics are different from all chakras
    assert crown_metrics != all_metrics
    assert crown_metrics['chakra_alignment'] >= all_metrics['chakra_alignment']

def test_consciousness_visualization():
    qci = QuantumConsciousnessIntegrator()
    
    # Test visualization without displaying
    fig, ax = plt.subplots(figsize=(12, 12))
    qci.visualize_consciousness_pattern()
    
    # Verify plot elements
    assert len(ax.patches) == len(qci.chakra_frequencies)  # One circle per chakra
    assert len(ax.texts) == len(qci.chakra_frequencies)  # One label per chakra
    assert len(ax.lines) > 0  # Connecting lines between chakras

def test_golden_ratio_integration():
    qci = QuantumConsciousnessIntegrator()
    input_state = np.array([1] + [0] * (2**7 - 1))
    
    circuit = qci.create_consciousness_circuit(input_state)
    
    # Verify golden ratio is used in circuit
    golden_ratio_angles = [3 * qci.golden_ratio * np.pi/9,
                          6 * qci.golden_ratio * np.pi/9,
                          9 * qci.golden_ratio * np.pi/9]
    
    for angle in golden_ratio_angles:
        assert any(gate.name in ['rz', 'ry'] and gate.params[0] == angle for gate in circuit.data)

def test_noise_model_integration():
    qci = QuantumConsciousnessIntegrator()
    
    # Verify noise model is properly configured
    assert qci.simulator.noise_model is not None
    assert len(qci.simulator.noise_model.noise_instructions) > 0

def test_quantum_state_evolution():
    qci = QuantumConsciousnessIntegrator()
    
    # Test different input states
    state1 = np.array([1] + [0] * (2**7 - 1))
    state2 = np.array([0] * (2**7 - 1) + [1])
    
    metrics1 = qci.run_consciousness_circuit(state1)
    metrics2 = qci.run_consciousness_circuit(state2)
    
    # Verify metrics change with different input states
    assert metrics1 != metrics2

def test_chakra_frequency_mapping():
    qci = QuantumConsciousnessIntegrator()
    
    # Verify chakra frequencies are properly defined
    assert len(qci.chakra_frequencies) == 7
    assert all(isinstance(freq, float) for freq in qci.chakra_frequencies.values())
    assert all(freq > 0 for freq in qci.chakra_frequencies.values())

def test_consciousness_amplification():
    qci = QuantumConsciousnessIntegrator()
    input_state = np.array([1] + [0] * (2**7 - 1))
    
    circuit = qci.create_consciousness_circuit(input_state)
    
    # Verify consciousness amplification gates
    amplification_angles = [3 * qci.golden_ratio * np.pi/9,
                           6 * qci.golden_ratio * np.pi/9,
                           9 * qci.golden_ratio * np.pi/9]
    
    for i in range(qci.num_qubits):
        for angle in amplification_angles:
            assert any(gate.name in ['rz', 'ry'] and gate.params[0] == angle and gate.qubits[0].index == i
                      for gate in circuit.data)

def test_entanglement_depth():
    qci = QuantumConsciousnessIntegrator(depth=5)
    input_state = np.array([1] + [0] * (2**7 - 1))
    
    circuit = qci.create_consciousness_circuit(input_state)
    
    # Count entanglement gates
    entanglement_gates = [gate for gate in circuit.data if gate.name in ['rxx', 'ryy', 'rzz']]
    
    # Verify number of entanglement gates
    expected_gates = qci.depth * (qci.num_qubits * (qci.num_qubits - 1) // 2) * 3
    assert len(entanglement_gates) == expected_gates 
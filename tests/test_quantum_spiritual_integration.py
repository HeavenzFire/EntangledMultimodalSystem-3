import pytest
import numpy as np
from quantum_spiritual_integration import QuantumSpiritualCore, EthicalQuantumFramework

@pytest.fixture
def quantum_spiritual_core():
    return QuantumSpiritualCore(num_qubits=3, shots=1024, noise_level=0.01)

@pytest.fixture
def ethical_framework():
    return EthicalQuantumFramework()

def test_sacred_geometry_circuit_creation(quantum_spiritual_core):
    input_state = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    circuit = quantum_spiritual_core.create_sacred_geometry_circuit(input_state)
    
    # Verify circuit structure
    assert circuit.num_qubits == 3
    assert circuit.num_clbits == 3
    
    # Verify golden ratio rotations
    for instruction in circuit.data:
        if instruction[0].name == 'ry':
            assert abs(instruction[0].params[0] - quantum_spiritual_core.golden_ratio * np.pi) < 1e-10

def test_spiritual_metrics_calculation(quantum_spiritual_core):
    input_state = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    metrics = quantum_spiritual_core.run_spiritual_quantum_circuit(input_state)
    
    # Verify metrics are within expected ranges
    assert 0 <= metrics['coherence'] <= 1
    assert 0 <= metrics['entanglement'] <= 1
    assert 0 <= metrics['harmony'] <= 1
    
    # Verify metrics sum to reasonable value
    total = sum(metrics.values())
    assert 0.5 <= total <= 2.5  # Allow for quantum fluctuations

def test_ethical_framework_evaluation(ethical_framework):
    quantum_state = np.array([0.5, 0.3, 0.2])
    action = "test_action"
    score = ethical_framework.evaluate_ethical_decision(quantum_state, action)
    
    # Verify ethical score is within expected range
    assert 0 <= score <= 1
    
    # Verify principle evaluations
    beneficence = ethical_framework._evaluate_beneficence(quantum_state)
    non_maleficence = ethical_framework._evaluate_non_maleficence(quantum_state)
    autonomy = ethical_framework._evaluate_autonomy(quantum_state)
    justice = ethical_framework._evaluate_justice(quantum_state)
    
    assert 0 <= beneficence <= 1
    assert 0 <= non_maleficence <= 1
    assert 0 <= autonomy <= 1
    assert 0 <= justice <= 1

def test_vortex_mathematics_integration(quantum_spiritual_core):
    input_state = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    circuit = quantum_spiritual_core.create_sacred_geometry_circuit(input_state)
    
    # Verify 3-6-9 pattern in circuit
    three_six_nine_found = False
    for instruction in circuit.data:
        if instruction[0].name in ['rz', 'ry']:
            angle = instruction[0].params[0]
            if abs(angle - 3 * np.pi/9) < 1e-10 or \
               abs(angle - 6 * np.pi/9) < 1e-10 or \
               abs(angle - 9 * np.pi/9) < 1e-10:
                three_six_nine_found = True
                break
    
    assert three_six_nine_found, "3-6-9 vortex pattern not found in circuit"

def test_flower_of_life_pattern(quantum_spiritual_core):
    input_state = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    circuit = quantum_spiritual_core.create_sacred_geometry_circuit(input_state)
    
    # Verify entanglement gates (RXX, RYY, RZZ)
    entanglement_gates = 0
    for instruction in circuit.data:
        if instruction[0].name in ['rxx', 'ryy', 'rzz']:
            entanglement_gates += 1
    
    # For 3 qubits, we expect 3 pairs with 3 gates each = 9 gates
    assert entanglement_gates == 9, "Incorrect number of Flower of Life entanglement gates"

def test_ethical_principle_weights(ethical_framework):
    # Verify Asilomar principle weights
    weights = ethical_framework.asilomar_principles
    assert weights['beneficence'] == 0.8
    assert weights['non_maleficence'] == 0.9
    assert weights['autonomy'] == 0.7
    assert weights['justice'] == 0.85
    
    # Verify weights sum to reasonable value
    total_weight = sum(weights.values())
    assert 3.0 <= total_weight <= 3.5

def test_quantum_spiritual_visualization(quantum_spiritual_core):
    # Test visualization without displaying
    fig, ax = quantum_spiritual_core.visualize_sacred_geometry()
    assert fig is not None
    assert ax is not None
    
    # Verify plot elements
    assert len(ax.patches) >= 7  # 6 outer circles + 1 central circle
    assert len(ax.lines) >= 12  # Platonic solid vertices

def test_noise_model_integration(quantum_spiritual_core):
    # Verify noise model is properly configured
    noise_model = quantum_spiritual_core.simulator.noise_model
    assert noise_model is not None
    
    # Verify depolarizing error is set
    errors = noise_model.to_dict()['errors']
    assert any(error['type'] == 'qerror' for error in errors)
    assert any(error['operations'] == ['u1', 'u2', 'u3'] for error in errors)

def test_quantum_state_evolution(quantum_spiritual_core):
    input_state = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    initial_metrics = quantum_spiritual_core.run_spiritual_quantum_circuit(input_state)
    
    # Modify input state
    modified_state = np.array([0.7, 0.3, 0, 0, 0, 0, 0, 0])
    modified_metrics = quantum_spiritual_core.run_spiritual_quantum_circuit(modified_state)
    
    # Verify metrics change with state modification
    assert not np.allclose(
        list(initial_metrics.values()),
        list(modified_metrics.values()),
        atol=0.1
    ) 
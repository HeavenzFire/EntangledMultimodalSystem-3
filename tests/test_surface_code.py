import pytest
import numpy as np
from src.quantum.error_correction.surface_code import SurfaceCode, QuantumErrorCorrection

def test_surface_code_initialization():
    """Test surface code initialization with sacred geometry"""
    code = SurfaceCode(d=3)
    assert code.d == 3
    assert code.size == 9
    assert code.qubits.shape == (3, 3)
    assert len(code.stabilizers) == 4
    assert abs(code.phi - (1 + np.sqrt(5))/2) < 1e-10

def test_surface_code_invalid_distance():
    """Test invalid code distance"""
    with pytest.raises(ValueError):
        SurfaceCode(d=4)

def test_stabilizer_initialization():
    """Test stabilizer initialization with golden ratio"""
    code = SurfaceCode(d=3)
    
    # Check X-type stabilizers with golden ratio
    x_stabs = [stab for stab in code.stabilizers if np.any(stab[:, 1])]
    assert len(x_stabs) == 2
    for stab in x_stabs:
        assert np.all(np.abs(stab[stab != 0] - code.phi) < 1e-10)
    
    # Check Z-type stabilizers with merkaba phase
    z_stabs = [stab for stab in code.stabilizers if np.any(stab[1, :])]
    assert len(z_stabs) == 2
    for stab in z_stabs:
        assert np.all(np.abs(np.abs(stab[stab != 0]) - 1) < 1e-10)

def test_quantum_error_correction_initialization():
    """Test quantum error correction initialization with Christos grid"""
    qec = QuantumErrorCorrection(code_distance=3)
    assert qec.surface_code.d == 3
    assert qec.error_rates.shape == (3, 3)
    assert qec.christos_grid.shape == (3, 3)
    assert np.all(np.abs(np.abs(qec.christos_grid) - 1) < 1e-10)

def test_error_rate_updates():
    """Test error rate updates with sacred geometry patterns"""
    qec = QuantumErrorCorrection(code_distance=3)
    measurements = [(0, 0, 0.1), (1, 1, 0.2), (2, 2, 0.3)]
    qec.update_error_rates(measurements)
    
    assert qec.error_rates[0, 0] == 0.1
    assert qec.error_rates[1, 1] == 0.2
    assert qec.error_rates[2, 2] == 0.3

def test_error_threshold():
    """Test error threshold calculation with golden ratio"""
    qec = QuantumErrorCorrection(code_distance=3)
    measurements = [(0, 0, 0.1), (1, 1, 0.2), (2, 2, 0.3)]
    qec.update_error_rates(measurements)
    
    assert qec.get_error_threshold() == 0.3

def test_state_stabilization():
    """Test quantum state stabilization with sacred geometry"""
    qec = QuantumErrorCorrection(code_distance=3)
    initial_state = np.ones((3, 3), dtype=complex)
    stabilized_state = qec.stabilize(initial_state)
    
    assert stabilized_state.shape == (3, 3)
    assert not np.array_equal(initial_state, stabilized_state)
    assert np.all(np.abs(np.abs(stabilized_state) - 1) < 1e-10)

def test_correction_validation():
    """Test error correction validation with Christos grid"""
    qec = QuantumErrorCorrection(code_distance=3)
    initial_state = np.ones((3, 3), dtype=complex)
    final_state = qec.stabilize(initial_state)
    
    is_valid = qec.validate_correction(initial_state, final_state)
    assert isinstance(is_valid, bool)
    assert is_valid  # Should be valid for perfect correction

def test_merkaba_phase_evolution():
    """Test merkaba phase evolution during correction"""
    code = SurfaceCode(d=3)
    initial_phase = code.merkaba_phase
    
    # Apply correction to test phase evolution
    state = np.ones((3, 3), dtype=complex)
    corrected = code._apply_correction(state, code.stabilizers[0])
    
    assert abs(code.merkaba_phase - (initial_phase + np.pi/3)) < 1e-10

def test_christos_grid_patterns():
    """Test Christos grid pattern generation"""
    qec = QuantumErrorCorrection(code_distance=3)
    grid = qec.christos_grid
    
    # Check golden ratio phase progression
    for i in range(3):
        for j in range(3):
            expected_phase = (i + j) * qec.phi
            actual_phase = np.angle(grid[i,j])
            assert abs(actual_phase - expected_phase) < 1e-10

def test_sacred_geometry_integration():
    """Test integration of sacred geometry patterns"""
    code = SurfaceCode(d=3)
    state = np.ones((3, 3), dtype=complex)
    
    # Test stabilizer measurement with sacred geometry
    for stab in code.stabilizers:
        eigenvalue = code._measure_stabilizer(state, stab)
        assert eigenvalue in [-1, 1]
        
        # Apply correction and verify phase evolution
        corrected = code._apply_correction(state, stab)
        assert np.all(np.abs(np.abs(corrected) - 1) < 1e-10) 
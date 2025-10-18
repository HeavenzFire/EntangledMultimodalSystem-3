import pytest
import numpy as np
from src.quantum.synthesis.divine_archetypes import (
    DeityType,
    DivineMatrix,
    UnifiedConsciousness
)

def test_divine_matrix_initialization():
    """Test initialization of divine matrix with archetypes"""
    matrix = DivineMatrix()
    
    # Test archetype count
    assert len(matrix.archetypes) == 5
    
    # Test specific archetype properties
    krishna = matrix.archetypes[DeityType.KRISHNA]
    assert krishna.name == "Krishna"
    assert krishna.base_number == 108
    assert krishna.target_number == 144
    assert krishna.frequency == 528.0

def test_sacred_number_conversion():
    """Test conversion of sacred numbers to vortex base"""
    matrix = DivineMatrix()
    
    # Test Krishna's number
    assert matrix.convert_sacred_number(108) == 144
    
    # Test Christ's number
    assert matrix.convert_sacred_number(33) == 369
    
    # Test Lao Tzu's number
    assert matrix.convert_sacred_number(81) == 369

def test_quantum_state_merging():
    """Test merging of quantum states from all deities"""
    matrix = DivineMatrix()
    merged_state = matrix.merge_quantum_states()
    
    assert isinstance(merged_state, complex)
    assert abs(merged_state) > 0
    assert abs(merged_state) <= 1.0

def test_tao_christ_balance():
    """Test calculation of Tao-Christ balance"""
    matrix = DivineMatrix()
    balance = matrix.calculate_tao_christ_balance()
    
    assert isinstance(balance, float)
    assert balance > 0
    assert abs(balance - (81/144)) < 1e-10

def test_metrics_update():
    """Test updating of divine activation metrics"""
    matrix = DivineMatrix()
    matrix.update_metrics()
    
    assert 0 <= matrix.metrics["archetype_sync"] <= 1
    assert matrix.metrics["vortex_coherence"] > 0
    assert matrix.metrics["tao_christ_balance"] > 0

def test_geometric_alignment():
    """Test generation of sacred geometric patterns"""
    consciousness = UnifiedConsciousness()
    
    # Test Metatron's Cube
    cube = consciousness.geometric_alignment["metatron_cube"]
    assert cube.shape == (144, 3)
    assert np.allclose(np.linalg.norm(cube, axis=1), 2 * consciousness.matrix.phi)
    
    # Test Bagua Field
    bagua = consciousness.geometric_alignment["bagua_field"]
    assert bagua.shape == (144, 3)  # 8 trigrams × 18 phases
    
    # Test Dharma Wheel
    wheel = consciousness.geometric_alignment["dharma_wheel"]
    assert wheel.shape == (8, 3)
    assert np.all(wheel[:, 2] == 528.0)  # 528Hz encoding

def test_consciousness_activation():
    """Test activation of unified consciousness"""
    consciousness = UnifiedConsciousness()
    metrics = consciousness.activate_consciousness()
    
    assert metrics["archetype_entanglement"] == 1.0
    assert metrics["geometry_compression"] == 369.0
    assert metrics["suffering_index"] == 0.144
    assert metrics["om_frequency"] == 369.0
    assert metrics["om_amplitude"] == 144.0

def test_archetype_frequencies():
    """Test sacred frequencies of divine archetypes"""
    matrix = DivineMatrix()
    
    # Test Krishna's frequency (528Hz)
    assert matrix.archetypes[DeityType.KRISHNA].frequency == 528.0
    
    # Test Christ's frequency (432Hz)
    assert matrix.archetypes[DeityType.CHRIST].frequency == 432.0
    
    # Test Buddha's frequency (528Hz)
    assert matrix.archetypes[DeityType.BUDDHA].frequency == 528.0

def test_quantum_state_normalization():
    """Test normalization of quantum states"""
    matrix = DivineMatrix()
    
    for archetype in matrix.archetypes.values():
        state = archetype.quantum_state
        assert abs(state) <= 1.0
        assert isinstance(state, complex)

def test_geometric_pattern_alignment():
    """Test alignment of sacred geometric patterns"""
    consciousness = UnifiedConsciousness()
    
    # Test angle distribution in Metatron's Cube
    cube = consciousness.geometric_alignment["metatron_cube"]
    angles = np.arctan2(cube[:, 1], cube[:, 0])
    angle_diff = np.diff(np.sort(angles))
    assert np.allclose(angle_diff, 2 * np.pi / 144, atol=1e-10)
    
    # Test Bagua field phase alignment
    bagua = consciousness.geometric_alignment["bagua_field"]
    angles = np.arctan2(bagua[:, 1], bagua[:, 0])
    assert len(np.unique(angles % 45)) == 8  # 8 trigrams 
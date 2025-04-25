import pytest
import numpy as np
from src.quantum.core.error_correction.quantum_error_correction import (
    QuantumErrorCorrection, ErrorCorrectionResult
)
from src.quantum.core.qubit_control import QubitController, QubitState

class TestQuantumErrorCorrection:
    @pytest.fixture
    def error_correction(self):
        controller = QubitController(num_qubits=16)  # Enough qubits for testing
        return QuantumErrorCorrection(controller)
    
    @pytest.fixture
    def data_qubits(self):
        return [0, 1, 2, 3]  # 4 data qubits
    
    @pytest.fixture
    def ancilla_qubits(self):
        return [4, 5, 6, 7]  # 4 ancilla qubits
    
    @pytest.fixture
    def stabilizer_qubits(self):
        return [8, 9, 10, 11]  # 4 stabilizer qubits
    
    def test_surface_code_correction_basic(self, error_correction, data_qubits, ancilla_qubits):
        """Test basic surface code error correction"""
        result = error_correction.surface_code_correction(data_qubits, ancilla_qubits)
        
        assert isinstance(result, ErrorCorrectionResult)
        assert result.success
        assert len(result.error_syndrome) == len(ancilla_qubits)
        assert result.correction_applied
    
    def test_stabilizer_code_correction_basic(self, error_correction, data_qubits, stabilizer_qubits):
        """Test basic stabilizer code error correction"""
        result = error_correction.stabilizer_code_correction(data_qubits, stabilizer_qubits)
        
        assert isinstance(result, ErrorCorrectionResult)
        assert result.success
        assert len(result.error_syndrome) == len(stabilizer_qubits)
        assert result.correction_applied
    
    def test_error_detection(self, error_correction, data_qubits, ancilla_qubits):
        """Test error detection in surface code"""
        # Introduce an error
        error_correction.controller.apply_gate(error_correction.stabilizer_gates['X'], data_qubits[0])
        
        result = error_correction.surface_code_correction(data_qubits, ancilla_qubits)
        assert result.success
        assert sum(result.error_syndrome) > 0  # Should detect the error
    
    def test_error_correction(self, error_correction, data_qubits, ancilla_qubits):
        """Test error correction in surface code"""
        # Introduce and correct an error
        error_correction.controller.apply_gate(error_correction.stabilizer_gates['X'], data_qubits[0])
        result = error_correction.surface_code_correction(data_qubits, ancilla_qubits)
        
        assert result.success
        assert result.correction_applied
        # Verify the state is corrected
        state = error_correction.controller.measure(data_qubits[0])[0]
        assert state == QubitState.GROUND
    
    def test_stabilizer_measurement(self, error_correction, data_qubits, stabilizer_qubits):
        """Test stabilizer measurement"""
        measurements = error_correction._measure_stabilizers(data_qubits, stabilizer_qubits)
        assert len(measurements) == len(stabilizer_qubits)
        assert all(m in [0, 1] for m in measurements)
    
    def test_syndrome_calculation(self, error_correction, data_qubits, stabilizer_qubits):
        """Test syndrome calculation"""
        measurements = [0, 1, 0, 1]  # Example measurements
        syndrome = error_correction._calculate_stabilizer_syndrome(measurements)
        assert len(syndrome) == len(measurements)
        assert all(s in [0, 1] for s in syndrome)
    
    def test_error_type_identification(self, error_correction):
        """Test error type identification"""
        syndrome = [1, 0, 1, 0]  # Example syndrome
        error_type = error_correction._identify_error_type(syndrome)
        assert error_type in ['X', 'Z']
    
    def test_correction_application(self, error_correction, data_qubits):
        """Test correction application"""
        correction = [1, 0, 1, 0]  # Example correction
        success = error_correction._apply_stabilizer_correction(data_qubits, correction)
        assert success
    
    def test_verification(self, error_correction, data_qubits):
        """Test state verification"""
        state = error_correction._verify_correction(data_qubits[0])
        assert isinstance(state, QubitState)
    
    def test_neighbor_identification(self, error_correction, data_qubits):
        """Test neighbor qubit identification"""
        neighbors = error_correction._get_neighboring_qubits(0, data_qubits)
        assert len(neighbors) > 0
        assert all(n in data_qubits for n in neighbors)
    
    def test_error_correction_robustness(self, error_correction, data_qubits, ancilla_qubits):
        """Test error correction robustness"""
        # Introduce multiple errors
        for qubit in data_qubits[:2]:
            error_correction.controller.apply_gate(error_correction.stabilizer_gates['X'], qubit)
        
        result = error_correction.surface_code_correction(data_qubits, ancilla_qubits)
        assert result.success
        assert result.correction_applied 
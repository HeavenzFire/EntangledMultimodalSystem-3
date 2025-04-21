import os
import time
import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from qiskit import QuantumCircuit
from src.core.quantum_processor import QuantumProcessor, QuantumError
from src.utils.errors import ModelError

class TestQuantumProcessor:
    """Test suite for QuantumProcessor class."""
    
    @pytest.fixture
    def processor(self):
        """Create a QuantumProcessor instance for testing."""
        with patch.dict(os.environ, {
            "QUANTUM_NUM_QUBITS": "3",
            "QUANTUM_SHOTS": "100",
            "QUANTUM_ERROR_CORRECTION": "surface_code",
            "QUANTUM_OPTIMIZATION_LEVEL": "2",
            "QUANTUM_MAX_EXECUTION_TIME": "60.0",
            "QUANTUM_HYBRID_THRESHOLD": "0.5",
            "SURFACE_CODE_DISTANCE": "2",
            "SURFACE_CODE_THRESHOLD": "0.01"
        }):
            return QuantumProcessor(
                backend_name="aer_simulator",
                cloud_provider=None,
                api_token=None
            )
    
    def test_initialization(self, processor):
        """Test successful initialization."""
        assert processor.backend_name == "aer_simulator"
        assert processor.cloud_provider is None
        assert processor.api_token is None
        assert processor.quantum_params["num_qubits"] == 3
        assert processor.quantum_params["shots"] == 100
        assert processor.quantum_params["error_correction"] == "surface_code"
        assert processor.state["status"] == "active"
        assert processor.metrics["fidelity"] == 0.0
    
    def test_initialization_with_cloud(self):
        """Test initialization with cloud provider."""
        with patch.dict(os.environ, {
            "QUANTUM_CLOUD_PROVIDER": "ibmq",
            "QUANTUM_API_TOKEN": "test_token"
        }):
            processor = QuantumProcessor(
                backend_name="ibmq_qasm_simulator",
                cloud_provider="ibmq",
                api_token="test_token"
            )
            assert processor.cloud_provider == "ibmq"
            assert processor.api_token == "test_token"
    
    def test_initialization_with_invalid_cloud(self):
        """Test initialization with invalid cloud provider."""
        with pytest.raises(QuantumError):
            QuantumProcessor(
                backend_name="invalid_backend",
                cloud_provider="invalid_provider"
            )
    
    def test_process_optimization(self, processor):
        """Test optimization task processing."""
        data = {
            "theta_0": 0.5,
            "phi_0": 0.3,
            "theta_1": 0.2,
            "phi_1": 0.4,
            "theta_2": 0.1,
            "phi_2": 0.6
        }
        result = processor.process("optimization", data)
        assert "result" in result
        assert "metrics" in result
        assert "execution_time" in result
        assert result["used_cloud"] is False
        assert processor.state["execution_count"] == 1
        assert processor.state["local_usage"] == 1
    
    def test_process_sampling(self, processor):
        """Test sampling task processing."""
        data = {
            "entangle_0_1": True,
            "entangle_1_2": True
        }
        result = processor.process("sampling", data)
        assert "result" in result
        assert "metrics" in result
        assert "execution_time" in result
        assert result["used_cloud"] is False
        assert processor.state["execution_count"] == 1
        assert processor.state["local_usage"] == 1
    
    def test_process_entanglement(self, processor):
        """Test entanglement task processing."""
        data = {
            "state_0": [1, 0],
            "state_1": [0, 1],
            "state_2": [1, 0],
            "entangle_0_1": True,
            "entangle_1_2": True
        }
        result = processor.process("entanglement", data)
        assert "result" in result
        assert "metrics" in result
        assert "execution_time" in result
        assert result["used_cloud"] is False
        assert processor.state["execution_count"] == 1
        assert processor.state["local_usage"] == 1
    
    def test_process_invalid_task(self, processor):
        """Test processing with invalid task type."""
        with pytest.raises(QuantumError):
            processor.process("invalid_task", {})
    
    def test_should_use_cloud(self, processor):
        """Test cloud usage decision making."""
        # Test with low complexity
        data = {"theta_0": 0.1}
        assert not processor._should_use_cloud("optimization", data)
        
        # Test with high complexity
        data = {f"theta_{i}": 1.0 for i in range(10)}
        assert processor._should_use_cloud("optimization", data)
    
    def test_create_circuit(self, processor):
        """Test quantum circuit creation."""
        # Test optimization circuit
        data = {"theta_0": 0.5, "phi_0": 0.3}
        circuit = processor._create_circuit("optimization", data)
        assert isinstance(circuit, QuantumCircuit)
        assert circuit.num_qubits == 3
        
        # Test sampling circuit
        data = {"entangle_0_1": True}
        circuit = processor._create_circuit("sampling", data)
        assert isinstance(circuit, QuantumCircuit)
        assert circuit.num_qubits == 3
        
        # Test entanglement circuit
        data = {"state_0": [1, 0], "entangle_0_1": True}
        circuit = processor._create_circuit("entanglement", data)
        assert isinstance(circuit, QuantumCircuit)
        assert circuit.num_qubits == 3
    
    def test_error_correction(self, processor):
        """Test error correction application."""
        # Test surface code
        circuit = QuantumCircuit(3)
        corrected_circuit = processor._apply_error_correction(circuit)
        assert isinstance(corrected_circuit, QuantumCircuit)
        assert len(processor.error_correction["logical_qubits"]) > 0
        
        # Test repetition code
        processor.quantum_params["error_correction"] = "repetition_code"
        processor._initialize_error_correction()
        corrected_circuit = processor._apply_error_correction(circuit)
        assert isinstance(corrected_circuit, QuantumCircuit)
        assert len(processor.error_correction["logical_qubits"]) > 0
    
    def test_execute_local(self, processor):
        """Test local circuit execution."""
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure_all()
        
        result = processor._execute_local(circuit)
        assert "counts" in result
        assert "time_taken" in result
        assert "success" in result
        assert result["success"] is True
    
    @patch('qiskit.providers.ibmq.IBMQ')
    def test_execute_cloud(self, mock_ibmq, processor):
        """Test cloud circuit execution."""
        # Mock IBMQ provider and backend
        mock_provider = MagicMock()
        mock_backend = MagicMock()
        mock_job = MagicMock()
        mock_result = MagicMock()
        
        mock_ibmq.enable_account.return_value = None
        mock_ibmq.get_provider.return_value = mock_provider
        mock_provider.get_backend.return_value = mock_backend
        mock_backend.run.return_value = mock_job
        mock_job.result.return_value = mock_result
        mock_result.get_counts.return_value = {"000": 50, "111": 50}
        mock_result.time_taken = 1.0
        mock_result.success = True
        
        processor.cloud_provider = "ibmq"
        processor.api_token = "test_token"
        processor._initialize_backend()
        
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure_all()
        
        result = processor._execute_cloud(circuit)
        assert "counts" in result
        assert "time_taken" in result
        assert "success" in result
        assert result["success"] is True
    
    def test_calculate_complexity(self, processor):
        """Test complexity calculation."""
        # Test optimization complexity
        data = {f"theta_{i}": 0.5 for i in range(3)}
        complexity = processor._calculate_complexity("optimization", data)
        assert 0 <= complexity <= 1
        
        # Test sampling complexity
        data = {f"entangle_{i}_{i+1}": True for i in range(2)}
        complexity = processor._calculate_complexity("sampling", data)
        assert 0 <= complexity <= 1
        
        # Test entanglement complexity
        data = {f"entangle_{i}_{i+1}": True for i in range(2)}
        complexity = processor._calculate_complexity("entanglement", data)
        assert 0 <= complexity <= 1
    
    def test_calculate_resources(self, processor):
        """Test resource calculation."""
        # Test optimization resources
        data = {f"theta_{i}": 0.5 for i in range(3)}
        resources = processor._calculate_resources("optimization", data)
        assert 0 <= resources <= 1
        
        # Test sampling resources
        data = {f"entangle_{i}_{i+1}": True for i in range(2)}
        resources = processor._calculate_resources("sampling", data)
        assert 0 <= resources <= 1
        
        # Test entanglement resources
        data = {f"entangle_{i}_{i+1}": True for i in range(2)}
        resources = processor._calculate_resources("entanglement", data)
        assert 0 <= resources <= 1
    
    def test_update_metrics(self, processor):
        """Test metrics update."""
        result = {
            "counts": {"000": 60, "111": 40},
            "time_taken": 1.0,
            "success": True
        }
        execution_time = 1.0
        
        processor._update_metrics(result, execution_time)
        assert 0 <= processor.metrics["fidelity"] <= 1
        assert 0 <= processor.metrics["gate_performance"] <= 1
        assert 0 <= processor.metrics["error_rate"] <= 1
        assert processor.metrics["execution_time"] == execution_time
        assert 0 <= processor.metrics["resource_usage"] <= 1
    
    def test_get_state(self, processor):
        """Test state retrieval."""
        state = processor.get_state()
        assert "status" in state
        assert "last_execution" in state
        assert "execution_count" in state
        assert "error_count" in state
        assert "cloud_usage" in state
        assert "local_usage" in state
    
    def test_get_metrics(self, processor):
        """Test metrics retrieval."""
        metrics = processor.get_metrics()
        assert "fidelity" in metrics
        assert "gate_performance" in metrics
        assert "error_rate" in metrics
        assert "execution_time" in metrics
        assert "resource_usage" in metrics
    
    def test_reset(self, processor):
        """Test processor reset."""
        # Perform some operations
        processor.process("optimization", {"theta_0": 0.5})
        
        # Reset processor
        processor.reset()
        
        # Verify reset state
        assert processor.state["execution_count"] == 0
        assert processor.state["error_count"] == 0
        assert processor.state["cloud_usage"] == 0
        assert processor.state["local_usage"] == 0
        assert processor.metrics["fidelity"] == 0.0
        assert processor.metrics["gate_performance"] == 0.0
        assert processor.metrics["error_rate"] == 0.0
        assert processor.metrics["execution_time"] == 0.0
        assert processor.metrics["resource_usage"] == 0.0

def test_initialization(quantum_processor):
    """Test quantum processor initialization."""
    assert quantum_processor.num_qubits == 4
    assert quantum_processor.state is not None
    assert isinstance(quantum_processor.state, np.ndarray)
    assert quantum_processor.state.shape == (2**4,)

def test_create_quantum_state(quantum_processor):
    """Test creation of quantum states."""
    state = quantum_processor.create_quantum_state()
    assert state is not None
    assert isinstance(state, np.ndarray)
    assert state.shape == (2**4,)
    assert np.abs(np.sum(np.abs(state)**2) - 1.0) < 1e-10

def test_apply_quantum_gate(quantum_processor):
    """Test application of quantum gates."""
    initial_state = quantum_processor.state.copy()
    
    # Test Hadamard gate
    quantum_processor.apply_gate("H", 0)
    assert not np.array_equal(quantum_processor.state, initial_state)
    
    # Test CNOT gate
    quantum_processor.apply_gate("CNOT", [0, 1])
    assert not np.array_equal(quantum_processor.state, initial_state)

def test_measure_state(quantum_processor):
    """Test quantum state measurement."""
    # Apply Hadamard to create superposition
    quantum_processor.apply_gate("H", 0)
    result = quantum_processor.measure_state()
    assert isinstance(result, dict)
    assert "measurement_outcome" in result
    assert "probability_distribution" in result
    assert isinstance(result["probability_distribution"], np.ndarray)

def test_entangle_qubits(quantum_processor):
    """Test qubit entanglement."""
    quantum_processor.entangle_qubits([0, 1])
    state = quantum_processor.state
    # Verify entanglement by checking reduced density matrix
    density_matrix = np.outer(state, state.conj())
    reduced_matrix = np.trace(density_matrix.reshape(2, 2, 2, 2), axis1=1, axis2=3)
    eigenvalues = np.linalg.eigvals(reduced_matrix)
    # Check if the state is maximally entangled
    assert np.allclose(eigenvalues, [0.5, 0.5], atol=1e-10)

def test_quantum_error_correction(quantum_processor):
    """Test quantum error correction."""
    # Introduce error
    quantum_processor.apply_gate("X", 0)  # Bit flip error
    # Apply error correction
    corrected_state = quantum_processor.apply_error_correction()
    assert corrected_state is not None
    assert isinstance(corrected_state, np.ndarray)

def test_get_quantum_state_info(quantum_processor):
    """Test retrieval of quantum state information."""
    info = quantum_processor.get_quantum_state_info()
    assert isinstance(info, dict)
    assert "state_vector" in info
    assert "num_qubits" in info
    assert "entanglement_measure" in info

def test_error_handling(quantum_processor):
    """Test error handling in quantum operations."""
    # Test invalid number of qubits
    with pytest.raises(ModelError):
        QuantumProcessor(num_qubits=0)
    
    # Test invalid gate
    with pytest.raises(ModelError):
        quantum_processor.apply_gate("INVALID", 0)
    
    # Test invalid qubit index
    with pytest.raises(ModelError):
        quantum_processor.apply_gate("H", quantum_processor.num_qubits + 1)

def test_reset_processor(quantum_processor):
    """Test reset of quantum processor."""
    # Apply some operations
    quantum_processor.apply_gate("H", 0)
    quantum_processor.apply_gate("CNOT", [0, 1])
    
    # Reset
    quantum_processor.reset_processor()
    
    # Verify reset state
    assert np.array_equal(quantum_processor.state, np.array([1] + [0]*(2**4-1))) 
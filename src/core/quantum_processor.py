import os
import time
import logging
from typing import Dict, Any, Optional, List, Tuple, Union
from src.utils.errors import ModelError, QuantumError
from dotenv import load_dotenv
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.providers import Backend
from qiskit.providers.ibmq import IBMQ
from qiskit.providers.aer import AerSimulator
from qiskit.primitives import Sampler, Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Options

class QuantumProcessor:
    """Advanced Quantum Processor with hybrid quantum-classical capabilities."""
    
    def __init__(
        self,
        backend_name: str = "aer_simulator",
        cloud_provider: Optional[str] = None,
        api_token: Optional[str] = None
    ):
        """Initialize Quantum Processor.
        
        Args:
            backend_name: Name of the quantum backend to use
            cloud_provider: Optional cloud provider name (ibmq, aws, azure)
            api_token: Optional API token for cloud provider
        """
        try:
            # Load environment variables
            load_dotenv()
            
            # Initialize parameters
            self.backend_name = backend_name
            self.cloud_provider = cloud_provider or os.getenv("QUANTUM_CLOUD_PROVIDER")
            self.api_token = api_token or os.getenv("QUANTUM_API_TOKEN")
            
            # Initialize quantum parameters
            self.quantum_params = {
                "num_qubits": int(os.getenv("QUANTUM_NUM_QUBITS", "5")),
                "shots": int(os.getenv("QUANTUM_SHOTS", "1024")),
                "error_correction": os.getenv("QUANTUM_ERROR_CORRECTION", "surface_code"),
                "optimization_level": int(os.getenv("QUANTUM_OPTIMIZATION_LEVEL", "3")),
                "max_execution_time": float(os.getenv("QUANTUM_MAX_EXECUTION_TIME", "300.0")),
                "hybrid_threshold": float(os.getenv("QUANTUM_HYBRID_THRESHOLD", "0.5"))
            }
            
            # Initialize state
            self.state = {
                "status": "active",
                "last_execution": None,
                "execution_count": 0,
                "error_count": 0,
                "cloud_usage": 0,
                "local_usage": 0
            }
            
            # Initialize metrics
            self.metrics = {
                "fidelity": 0.0,
                "gate_performance": 0.0,
                "error_rate": 0.0,
                "execution_time": 0.0,
                "resource_usage": 0.0
            }
            
            # Initialize backend and primitives
            self._initialize_backend()
            
            # Initialize error correction
            self._initialize_error_correction()
            
            logging.info("QuantumProcessor initialized")
            
        except Exception as e:
            logging.error(f"Error initializing QuantumProcessor: {str(e)}")
            raise ModelError(f"Failed to initialize QuantumProcessor: {str(e)}")

    def _initialize_backend(self) -> None:
        """Initialize quantum backend."""
        try:
            if self.cloud_provider == "ibmq":
                if not self.api_token:
                    raise QuantumError("IBMQ API token required for cloud backend")
                # Initialize QiskitRuntimeService
                self.service = QiskitRuntimeService(channel="ibm_quantum", token=self.api_token)
                self.backend = self.service.backend(self.backend_name)
                # Initialize primitives with session
                self.session = Session(service=self.service, backend=self.backend_name)
                self.sampler = Sampler(session=self.session)
                self.estimator = Estimator(session=self.session)
            elif self.cloud_provider == "aws":
                # Initialize AWS Braket backend
                from braket.aws import AwsDevice
                self.backend = AwsDevice(self.backend_name)
                self.sampler = Sampler()
                self.estimator = Estimator()
            elif self.cloud_provider == "azure":
                # Initialize Azure Quantum backend
                from azure.quantum import Workspace
                workspace = Workspace(
                    subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID"),
                    resource_group=os.getenv("AZURE_RESOURCE_GROUP"),
                    name=os.getenv("AZURE_WORKSPACE_NAME"),
                    location=os.getenv("AZURE_LOCATION")
                )
                self.backend = workspace.get_backend(self.backend_name)
                self.sampler = Sampler()
                self.estimator = Estimator()
            else:
                # Use local simulator
                self.backend = AerSimulator()
                self.sampler = Sampler()
                self.estimator = Estimator()
                
        except Exception as e:
            logging.error(f"Error initializing backend: {str(e)}")
            raise QuantumError(f"Backend initialization failed: {str(e)}")

    def _initialize_error_correction(self) -> None:
        """Initialize quantum error correction."""
        try:
            if self.quantum_params["error_correction"] == "surface_code":
                # Initialize surface code error correction
                self.error_correction = {
                    "type": "surface_code",
                    "distance": int(os.getenv("SURFACE_CODE_DISTANCE", "3")),
                    "threshold": float(os.getenv("SURFACE_CODE_THRESHOLD", "0.01")),
                    "logical_qubits": []
                }
            elif self.quantum_params["error_correction"] == "repetition_code":
                # Initialize repetition code error correction
                self.error_correction = {
                    "type": "repetition_code",
                    "repetitions": int(os.getenv("REPETITION_CODE_REPETITIONS", "3")),
                    "threshold": float(os.getenv("REPETITION_CODE_THRESHOLD", "0.1")),
                    "logical_qubits": []
                }
            else:
                self.error_correction = None
                
        except Exception as e:
            logging.error(f"Error initializing error correction: {str(e)}")
            raise QuantumError(f"Error correction initialization failed: {str(e)}")

    def process(
        self,
        task_type: str,
        data: Union[np.ndarray, Dict[str, Any]],
        use_cloud: Optional[bool] = None
    ) -> Dict[str, Any]:
        """Process quantum task.
        
        Args:
            task_type: Type of quantum task (optimization, sampling, entanglement)
            data: Input data for the task
            use_cloud: Whether to use cloud backend (optional)
            
        Returns:
            Processing results
        """
        try:
            start_time = time.time()
            
            # Determine whether to use cloud backend
            if use_cloud is None:
                use_cloud = self._should_use_cloud(task_type, data)
            
            # Create quantum circuit
            circuit = self._create_circuit(task_type, data)
            
            # Apply error correction if enabled
            if self.error_correction:
                circuit = self._apply_error_correction(circuit)
            
            # Execute quantum circuit using primitives
            if task_type == "sampling":
                result = self._execute_sampling(circuit)
            elif task_type == "optimization":
                result = self._execute_optimization(circuit, data)
            else:
                result = self._execute_default(circuit)
            
            # Update metrics
            self._update_metrics(result, time.time() - start_time)
            
            # Update state
            self.state["last_execution"] = time.time()
            self.state["execution_count"] += 1
            if use_cloud:
                self.state["cloud_usage"] += 1
            else:
                self.state["local_usage"] += 1
            
            return {
                "result": result,
                "metrics": self.metrics,
                "execution_time": time.time() - start_time,
                "used_cloud": use_cloud
            }
            
        except Exception as e:
            self.state["error_count"] += 1
            logging.error(f"Error in quantum processing: {str(e)}")
            raise QuantumError(f"Quantum processing failed: {str(e)}")

    def _execute_sampling(self, circuit: QuantumCircuit) -> Dict[str, Any]:
        """Execute circuit using Sampler primitive."""
        job = self.sampler.run(circuit, shots=self.quantum_params["shots"])
        result = job.result()
        return {
            "quasi_dists": result.quasi_dists,
            "metadata": result.metadata
        }

    def _execute_optimization(self, circuit: QuantumCircuit, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute optimization task using Estimator primitive."""
        observables = [SparsePauliOp.from_list(obs) for obs in data.get("observables", [])]
        job = self.estimator.run(circuit, observables)
        result = job.result()
        return {
            "values": result.values,
            "metadata": result.metadata
        }

    def _execute_default(self, circuit: QuantumCircuit) -> Dict[str, Any]:
        """Execute circuit using default backend."""
        transpiled_circuit = transpile(circuit, self.backend)
        job = self.backend.run(transpiled_circuit, shots=self.quantum_params["shots"])
        result = job.result()
        return {
            "counts": result.get_counts(),
            "metadata": result.to_dict().get("results", [{}])[0].get("metadata", {})
        }

    def _should_use_cloud(self, task_type: str, data: Union[np.ndarray, Dict[str, Any]]) -> bool:
        """Determine whether to use cloud backend.
        
        Args:
            task_type: Type of quantum task
            data: Input data
            
        Returns:
            Whether to use cloud backend
        """
        try:
            # Check if cloud provider is available
            if not self.cloud_provider:
                return False
            
            # Check task complexity
            complexity = self._calculate_complexity(task_type, data)
            
            # Check resource requirements
            resources = self._calculate_resources(task_type, data)
            
            # Make decision based on thresholds
            return (
                complexity > self.quantum_params["hybrid_threshold"] or
                resources > self.quantum_params["hybrid_threshold"]
            )
            
        except Exception as e:
            logging.error(f"Error determining cloud usage: {str(e)}")
            return False

    def _create_circuit(self, task_type: str, data: Union[np.ndarray, Dict[str, Any]]) -> QuantumCircuit:
        """Create quantum circuit for task.
        
        Args:
            task_type: Type of quantum task
            data: Input data
            
        Returns:
            Quantum circuit
        """
        try:
            # Create base circuit
            circuit = QuantumCircuit(self.quantum_params["num_qubits"])
            
            # Add gates based on task type
            if task_type == "optimization":
                circuit = self._add_optimization_gates(circuit, data)
            elif task_type == "sampling":
                circuit = self._add_sampling_gates(circuit, data)
            elif task_type == "entanglement":
                circuit = self._add_entanglement_gates(circuit, data)
            else:
                raise QuantumError(f"Invalid task type: {task_type}")
            
            # Add measurement
            circuit.measure_all()
            
            return circuit
            
        except Exception as e:
            logging.error(f"Error creating quantum circuit: {str(e)}")
            raise QuantumError(f"Circuit creation failed: {str(e)}")

    def _add_optimization_gates(self, circuit: QuantumCircuit, data: Dict[str, Any]) -> QuantumCircuit:
        """Add optimization gates to circuit.
        
        Args:
            circuit: Quantum circuit
            data: Optimization parameters
            
        Returns:
            Updated quantum circuit
        """
        try:
            # Add parameterized gates
            for qubit in range(self.quantum_params["num_qubits"]):
                circuit.rx(data.get(f"theta_{qubit}", 0.0), qubit)
                circuit.ry(data.get(f"phi_{qubit}", 0.0), qubit)
            
            # Add entangling gates
            for i in range(self.quantum_params["num_qubits"] - 1):
                circuit.cx(i, i + 1)
            
            return circuit
            
        except Exception as e:
            logging.error(f"Error adding optimization gates: {str(e)}")
            raise QuantumError(f"Optimization gate addition failed: {str(e)}")

    def _add_sampling_gates(self, circuit: QuantumCircuit, data: Dict[str, Any]) -> QuantumCircuit:
        """Add sampling gates to circuit.
        
        Args:
            circuit: Quantum circuit
            data: Sampling parameters
            
        Returns:
            Updated quantum circuit
        """
        try:
            # Add Hadamard gates for superposition
            for qubit in range(self.quantum_params["num_qubits"]):
                circuit.h(qubit)
            
            # Add controlled gates based on data
            for i in range(self.quantum_params["num_qubits"]):
                for j in range(i + 1, self.quantum_params["num_qubits"]):
                    if data.get(f"entangle_{i}_{j}", False):
                        circuit.cx(i, j)
            
            return circuit
            
        except Exception as e:
            logging.error(f"Error adding sampling gates: {str(e)}")
            raise QuantumError(f"Sampling gate addition failed: {str(e)}")

    def _add_entanglement_gates(self, circuit: QuantumCircuit, data: Dict[str, Any]) -> QuantumCircuit:
        """Add entanglement gates to circuit.
        
        Args:
            circuit: Quantum circuit
            data: Entanglement parameters
            
        Returns:
            Updated quantum circuit
        """
        try:
            # Add initial state preparation
            for qubit in range(self.quantum_params["num_qubits"]):
                circuit.initialize(data.get(f"state_{qubit}", [1, 0]), qubit)
            
            # Add entanglement gates
            for i in range(self.quantum_params["num_qubits"]):
                for j in range(i + 1, self.quantum_params["num_qubits"]):
                    if data.get(f"entangle_{i}_{j}", False):
                        circuit.cx(i, j)
                        circuit.h(i)
                        circuit.h(j)
            
            return circuit
            
        except Exception as e:
            logging.error(f"Error adding entanglement gates: {str(e)}")
            raise QuantumError(f"Entanglement gate addition failed: {str(e)}")

    def _apply_error_correction(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Apply error correction to circuit.
        
        Args:
            circuit: Quantum circuit
            
        Returns:
            Error-corrected quantum circuit
        """
        try:
            if self.error_correction["type"] == "surface_code":
                # Apply surface code error correction
                return self._apply_surface_code(circuit)
            elif self.error_correction["type"] == "repetition_code":
                # Apply repetition code error correction
                return self._apply_repetition_code(circuit)
            else:
                return circuit
                
        except Exception as e:
            logging.error(f"Error applying error correction: {str(e)}")
            raise QuantumError(f"Error correction failed: {str(e)}")

    def _apply_surface_code(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Apply surface code error correction.
        
        Args:
            circuit: Quantum circuit
            
        Returns:
            Error-corrected quantum circuit
        """
        try:
            # Create logical qubits
            logical_qubits = []
            for i in range(0, self.quantum_params["num_qubits"], 4):
                logical_qubits.append((i, i + 1, i + 2, i + 3))
            
            # Apply stabilizer measurements
            for qubits in logical_qubits:
                circuit.h(qubits[0])
                circuit.cx(qubits[0], qubits[1])
                circuit.cx(qubits[0], qubits[2])
                circuit.cx(qubits[0], qubits[3])
                circuit.h(qubits[0])
            
            self.error_correction["logical_qubits"] = logical_qubits
            return circuit
            
        except Exception as e:
            logging.error(f"Error applying surface code: {str(e)}")
            raise QuantumError(f"Surface code application failed: {str(e)}")

    def _apply_repetition_code(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Apply repetition code error correction.
        
        Args:
            circuit: Quantum circuit
            
        Returns:
            Error-corrected quantum circuit
        """
        try:
            # Create logical qubits
            logical_qubits = []
            for i in range(0, self.quantum_params["num_qubits"], 3):
                logical_qubits.append((i, i + 1, i + 2))
            
            # Apply repetition code
            for qubits in logical_qubits:
                circuit.cx(qubits[0], qubits[1])
                circuit.cx(qubits[0], qubits[2])
            
            self.error_correction["logical_qubits"] = logical_qubits
            return circuit
            
        except Exception as e:
            logging.error(f"Error applying repetition code: {str(e)}")
            raise QuantumError(f"Repetition code application failed: {str(e)}")

    def _calculate_complexity(self, task_type: str, data: Union[np.ndarray, Dict[str, Any]]) -> float:
        """Calculate task complexity.
        
        Args:
            task_type: Type of quantum task
            data: Input data
            
        Returns:
            Complexity score
        """
        try:
            if task_type == "optimization":
                return len(data) / self.quantum_params["num_qubits"]
            elif task_type == "sampling":
                return np.log2(len(data)) / self.quantum_params["num_qubits"]
            elif task_type == "entanglement":
                return sum(1 for k, v in data.items() if k.startswith("entangle_")) / (
                    self.quantum_params["num_qubits"] * (self.quantum_params["num_qubits"] - 1) / 2
                )
            else:
                return 0.0
                
        except Exception as e:
            logging.error(f"Error calculating complexity: {str(e)}")
            return 0.0

    def _calculate_resources(self, task_type: str, data: Union[np.ndarray, Dict[str, Any]]) -> float:
        """Calculate resource requirements.
        
        Args:
            task_type: Type of quantum task
            data: Input data
            
        Returns:
            Resource score
        """
        try:
            if task_type == "optimization":
                return sum(abs(v) for v in data.values()) / len(data)
            elif task_type == "sampling":
                return len(data) / (2 ** self.quantum_params["num_qubits"])
            elif task_type == "entanglement":
                return sum(1 for v in data.values() if v) / len(data)
            else:
                return 0.0
                
        except Exception as e:
            logging.error(f"Error calculating resources: {str(e)}")
            return 0.0

    def _update_metrics(self, result: Dict[str, Any], execution_time: float) -> None:
        """Update processor metrics.
        
        Args:
            result: Execution results
            execution_time: Time taken for execution
        """
        try:
            # Calculate fidelity
            counts = result["counts"]
            total = sum(counts.values())
            max_count = max(counts.values())
            self.metrics["fidelity"] = max_count / total if total > 0 else 0.0
            
            # Calculate gate performance
            self.metrics["gate_performance"] = 1.0 - (len(counts) / (2 ** self.quantum_params["num_qubits"]))
            
            # Calculate error rate
            self.metrics["error_rate"] = 1.0 - self.metrics["fidelity"]
            
            # Update execution time
            self.metrics["execution_time"] = execution_time
            
            # Calculate resource usage
            self.metrics["resource_usage"] = (
                self.metrics["gate_performance"] *
                (1.0 - self.metrics["error_rate"]) *
                (1.0 - execution_time / self.quantum_params["max_execution_time"])
            )
            
        except Exception as e:
            logging.error(f"Error updating metrics: {str(e)}")

    def get_state(self) -> Dict[str, Any]:
        """Get current processor state."""
        return self.state

    def get_metrics(self) -> Dict[str, Any]:
        """Get current processor metrics."""
        return self.metrics

    def reset(self) -> None:
        """Reset processor state."""
        self.state.update({
            "status": "active",
            "last_execution": None,
            "execution_count": 0,
            "error_count": 0,
            "cloud_usage": 0,
            "local_usage": 0
        })
        
        self.metrics.update({
            "fidelity": 0.0,
            "gate_performance": 0.0,
            "error_rate": 0.0,
            "execution_time": 0.0,
            "resource_usage": 0.0
        }) 
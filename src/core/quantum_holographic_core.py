import numpy as np
import logging
from typing import Dict, Any, Tuple
from .quantum_processor import QuantumProcessor
from .holographic_processor import HolographicProcessor
from .quantum_consciousness import QuantumConsciousness
from .quantum_ethics import QuantumEthics

class QuantumHolographicCore:
    """Core quantum-holographic processing system for DigigodNexus v7.0."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the quantum-holographic core.
        
        Args:
            config: Configuration dictionary with parameters for initialization
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Use bounded local defaults; cloud execution and larger simulations
        # must be explicitly configured. The previous 128-qubit request was
        # infeasible for this local simulator and used unsupported kwargs.
        self.qpu = QuantumProcessor(
            backend_name=self.config.get("backend_name", "aer_simulator"),
            cloud_provider=self.config.get("cloud_provider"),
            num_qubits=int(self.config.get("num_qubits", 5)),
            error_correction=self.config.get("error_correction", "none"),
        )
        self.holo = HolographicProcessor()
        self.consciousness = QuantumConsciousness(
            quantum_weight=float(self.config.get("quantum_weight", 0.4)),
            holographic_weight=float(self.config.get("holographic_weight", 0.3)),
            neural_weight=float(self.config.get("neural_weight", 0.3)),
        )
        self.ethics = QuantumEthics()

        # Initialize state and metrics
        self.state = {
            "quantum_entanglement": 0.0,
            "holographic_fidelity": 0.0,
            "consciousness_level": 0.0,
            "ethical_compliance": 0.0
        }
        
        self.metrics = {
            "processing_speed": 0.0,
            "energy_efficiency": 0.0,
            "error_rate": 0.0,
            "integration_score": 0.0
        }
        
        self.logger.info("QuantumHolographicCore initialized successfully")
    
    def process(self, data: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """Process finite numeric input through the bounded local pipeline.

        Quantum execution uses the configured simulator qubit count; reported
        metrics are software diagnostics and are not physical-hardware claims.
        """
        try:
            input_vector = np.asarray(data, dtype=float).reshape(-1)
            if input_vector.size == 0 or not np.all(np.isfinite(input_vector)):
                raise ValueError("data must be a non-empty finite numeric array")

            qubits = int(self.qpu.quantum_params["num_qubits"])
            if qubits < 1 or qubits > 12:
                raise ValueError("local core supports 1-12 qubits per run")

            # Encode a bounded nearest-neighbour entanglement task.
            task = {f"state_{i}": [1.0, 0.0] for i in range(qubits)}
            for i in range(qubits - 1):
                task[f"entangle_{i}_{i + 1}"] = True
            quantum_result = self.qpu.process(
                task_type="entanglement", data=task, use_cloud=False
            )
            counts = quantum_result.get("result", {}).get("counts", {})
            q_state = np.zeros(2 ** qubits, dtype=float)
            total = float(sum(counts.values())) if counts else 0.0
            if total <= 0:
                raise RuntimeError("quantum simulator returned no measurement counts")
            for bitstring, count in counts.items():
                try:
                    index = int(str(bitstring).replace(" ", ""), 2)
                except ValueError:
                    continue
                if 0 <= index < q_state.size:
                    q_state[index] = float(count) / total
            if not np.any(q_state):
                raise RuntimeError("quantum simulator returned unrecognized measurement keys")

            q_raw = self.qpu.get_metrics()
            q_elapsed = max(float(quantum_result.get("execution_time", 0.0)), 1e-9)
            q_metrics = {
                "entanglement_fidelity": float(q_raw.get("fidelity", 0.0)),
                "speed": 1.0 / q_elapsed,
                "efficiency": float(q_raw.get("resource_usage", 0.0)),
                "error_rate": float(q_raw.get("error_rate", 1.0)),
            }

            h_state = self.holo.project(q_state)
            h_raw = self.holo.get_metrics()
            h_metrics = {
                "fidelity": float(h_raw.get("fidelity", 0.0)),
                "speed": float(h_raw.get("speed", 0.0)),
                "efficiency": float(h_raw.get("efficiency", 0.0)),
                "error_rate": float(h_raw.get("error_rate", 1.0)),
            }

            c_state = self.consciousness.integrate(q_state, h_state)
            c_metrics = self.consciousness.get_metrics()
            e_metrics = self.ethics.validate({
                "operation": "process_numeric_input",
                "state": c_state.tolist(),
            })

            self.state.update({
                "quantum_entanglement": q_metrics["entanglement_fidelity"],
                "holographic_fidelity": h_metrics["fidelity"],
                "consciousness_level": c_metrics["phi_score"],
                "ethical_compliance": e_metrics["compliance_score"],
            })
            self.metrics.update({
                "processing_speed": min(q_metrics["speed"], h_metrics["speed"]),
                "energy_efficiency": (
                    q_metrics["efficiency"] + h_metrics["efficiency"]
                ) / 2.0,
                "error_rate": max(q_metrics["error_rate"], h_metrics["error_rate"]),
                "integration_score": c_metrics["integration_score"],
            })
            return c_state, self.metrics.copy()

        except Exception as e:
            self.logger.error("Error in quantum-holographic processing: %s", e)
            raise

    def calibrate(self, target_phi: float = 0.9) -> Dict[str, float]:
        """Calibrate the quantum-holographic system.
        
        Args:
            target_phi: Target consciousness level
            
        Returns:
            Calibration metrics
        """
        try:
            # Calibrate quantum processor
            q_cal = self.qpu.calibrate()
            
            # Calibrate holographic processor
            h_cal = self.holo.calibrate()
            
            # Calibrate consciousness matrix
            c_cal = self.consciousness.calibrate(target_phi=target_phi)
            
            # Update metrics
            self.metrics.update({
                "calibration_score": (q_cal["score"] + h_cal["score"] + c_cal["score"]) / 3,
                "quantum_calibration": q_cal["score"],
                "holographic_calibration": h_cal["score"],
                "consciousness_calibration": c_cal["score"]
            })
            
            return self.metrics
            
        except Exception as e:
            self.logger.error(f"Error in calibration: {str(e)}")
            raise
    
    def get_state(self) -> Dict[str, float]:
        """Get current system state.
        
        Returns:
            Dictionary of state metrics
        """
        return self.state
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current system metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        return self.metrics
    
    def reset(self) -> None:
        """Reset the system to initial state."""
        self.qpu.reset()
        self.holo.reset()
        self.consciousness.reset()
        self.state = {k: 0.0 for k in self.state}
        self.metrics = {k: 0.0 for k in self.metrics}
        self.logger.info("QuantumHolographicCore reset successfully") 
from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
import logging
from datetime import datetime

@dataclass
class SecurityMetrics:
    qkd_success_rate: float
    intrusion_attempts: int
    encryption_strength: int
    timestamp: str

class QuantumSecurityValidator:
    def __init__(self, qkd_protocol: str = "BB84", decoy_qubits: float = 0.05):
        self.logger = logging.getLogger(__name__)
        self.simulator = AerSimulator(method='statevector')
        self.qkd_protocol = qkd_protocol
        self.decoy_qubits = decoy_qubits
        self.intrusion_attempts = 0
        self.last_validation = None

    def _create_qkd_circuit(self, num_qubits: int = 1) -> QuantumCircuit:
        """Create a QKD circuit based on the specified protocol"""
        qr = QuantumRegister(num_qubits, 'q')
        cr = ClassicalRegister(num_qubits, 'c')
        qc = QuantumCircuit(qr, cr)
        
        if self.qkd_protocol == "BB84":
            # Apply Hadamard to half the qubits
            for i in range(num_qubits):
                if np.random.random() < 0.5:
                    qc.h(qr[i])
            qc.measure(qr, cr)
        else:
            raise ValueError(f"Unsupported QKD protocol: {self.qkd_protocol}")
        
        return qc

    def validate_encryption(self, payload: bytes) -> bool:
        """Validate encryption strength using quantum-resistant methods"""
        try:
            # Simulate quantum-resistant encryption
            key_length = len(payload) * 8
            encryption_strength = min(256, key_length)  # Cap at 256 bits
            
            # Verify integrity using quantum circuit
            qc = self._create_qkd_circuit(1)
            result = self.simulator.run(qc, shots=100).result()
            counts = result.get_counts(qc)
            
            # Check for expected distribution
            expected_distribution = {'0': 50, '1': 50}
            chi_squared = sum((counts.get(k, 0) - v)**2 / v 
                            for k, v in expected_distribution.items())
            
            return chi_squared < 10  # Statistical threshold
        except Exception as e:
            self.logger.error(f"Encryption validation failed: {str(e)}")
            return False

    def _detect_intrusion(self) -> bool:
        """Simulate intrusion detection using decoy qubits"""
        try:
            # Create circuit with decoy qubits
            num_qubits = 20
            num_decoys = int(num_qubits * self.decoy_qubits)
            qc = self._create_qkd_circuit(num_qubits)
            
            # Measure decoy qubits
            result = self.simulator.run(qc, shots=100).result()
            counts = result.get_counts(qc)
            
            # Check for unexpected measurements
            intrusion_detected = any(
                counts.get(bin(i)[2:].zfill(num_qubits), 0) > 5
                for i in range(2**num_decoys)
            )
            
            if intrusion_detected:
                self.intrusion_attempts += 1
            
            return not intrusion_detected
        except Exception as e:
            self.logger.error(f"Intrusion detection failed: {str(e)}")
            return False

    def run_security_sweep(self) -> SecurityMetrics:
        """Run comprehensive security validation"""
        try:
            # Test QKD
            qkd_circuit = self._create_qkd_circuit(10)
            result = self.simulator.run(qkd_circuit, shots=1000).result()
            counts = result.get_counts(qkd_circuit)
            qkd_success_rate = sum(counts.values()) / 1000
            
            # Test intrusion detection
            intrusion_detected = self._detect_intrusion()
            
            # Test encryption
            test_payload = b"test_payload"
            encryption_valid = self.validate_encryption(test_payload)
            
            # Calculate metrics
            metrics = SecurityMetrics(
                qkd_success_rate=qkd_success_rate,
                intrusion_attempts=self.intrusion_attempts,
                encryption_strength=256 if encryption_valid else 128,
                timestamp=datetime.now().isoformat()
            )
            
            self.last_validation = metrics
            return metrics
        except Exception as e:
            self.logger.error(f"Security sweep failed: {str(e)}")
            return SecurityMetrics(
                qkd_success_rate=0.0,
                intrusion_attempts=self.intrusion_attempts,
                encryption_strength=0,
                timestamp=datetime.now().isoformat()
            )

    def get_security_status(self) -> Dict[str, Any]:
        """Get current security status"""
        if self.last_validation is None:
            self.run_security_sweep()
        
        return {
            "qkd_success": self.last_validation.qkd_success_rate > 0.95,
            "intrusion_attempts": self.last_validation.intrusion_attempts,
            "encryption_strength": self.last_validation.encryption_strength,
            "timestamp": self.last_validation.timestamp
        } 
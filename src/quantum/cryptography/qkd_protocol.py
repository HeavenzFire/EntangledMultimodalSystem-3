from typing import List, Tuple, Dict, Optional
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import logging
from ..core.qubit_control import Qubit, QubitState, QuantumGate, QubitController
from .quantum_crypto import QuantumKey

logger = logging.getLogger(__name__)

@dataclass
class QKDResult:
    """Results from QKD protocol execution"""
    shared_key: bytes
    error_rate: float
    security_metrics: Dict[str, float]
    timestamp: datetime

class QKDProtocol:
    """Implementation of quantum key distribution protocols"""
    def __init__(self, controller: QubitController):
        self.controller = controller
        self.basis_choices = ['Z', 'X']  # Standard and Hadamard bases
        self.quantum_gates = {
            'Z': QuantumGate('Z', np.array([[1, 0], [0, -1]])),
            'X': QuantumGate('X', np.array([[0, 1], [1, 0]])),
            'H': QuantumGate('H', (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]]))
        }
        
    def bb84_protocol(self, num_qubits: int = 256) -> QKDResult:
        """Implement BB84 protocol for quantum key distribution"""
        # Initialize qubits
        self.controller.initialize_qubits(num_qubits)
        
        # Alice's operations
        alice_bits = np.random.randint(0, 2, num_qubits)
        alice_bases = np.random.choice(self.basis_choices, num_qubits)
        
        # Prepare qubits
        for i in range(num_qubits):
            if alice_bits[i] == 1:
                self.controller.apply_gate(self.quantum_gates['X'], i)
            if alice_bases[i] == 'X':
                self.controller.apply_gate(self.quantum_gates['H'], i)
        
        # Bob's measurements
        bob_bases = np.random.choice(self.basis_choices, num_qubits)
        bob_bits = []
        
        for i in range(num_qubits):
            if bob_bases[i] == 'X':
                self.controller.apply_gate(self.quantum_gates['H'], i)
            state, _ = self.controller.measure(i)
            bob_bits.append(1 if state == QubitState.EXCITED else 0)
        
        # Key sifting
        matching_bases = alice_bases == bob_bases
        sifted_key = [alice_bits[i] for i in range(num_qubits) if matching_bases[i]]
        
        # Error estimation
        error_rate = self._estimate_error_rate(alice_bits, bob_bits, matching_bases)
        
        # Security metrics
        security_metrics = {
            'error_rate': error_rate,
            'key_length': len(sifted_key),
            'eavesdropping_probability': self._calculate_eavesdropping_probability(error_rate)
        }
        
        return QKDResult(
            shared_key=bytes(sifted_key),
            error_rate=error_rate,
            security_metrics=security_metrics,
            timestamp=datetime.now()
        )
    
    def e91_protocol(self, num_qubits: int = 256) -> QKDResult:
        """Implement E91 protocol using entangled qubits"""
        # Initialize entangled qubit pairs
        self.controller.initialize_qubits(num_qubits * 2)
        
        # Create entangled pairs
        for i in range(0, num_qubits * 2, 2):
            self._create_entangled_pair(i, i + 1)
        
        # Measurement bases
        bases = ['Z', 'X', 'Y']
        alice_bases = np.random.choice(bases, num_qubits)
        bob_bases = np.random.choice(bases, num_qubits)
        
        # Perform measurements
        alice_results = []
        bob_results = []
        
        for i in range(num_qubits):
            alice_result = self._measure_in_basis(i * 2, alice_bases[i])
            bob_result = self._measure_in_basis(i * 2 + 1, bob_bases[i])
            alice_results.append(alice_result)
            bob_results.append(bob_result)
        
        # Key generation and error estimation
        shared_key, error_rate = self._process_e91_results(
            alice_results, bob_results, alice_bases, bob_bases
        )
        
        security_metrics = {
            'error_rate': error_rate,
            'entanglement_quality': self._calculate_entanglement_quality(alice_results, bob_results),
            'key_length': len(shared_key)
        }
        
        return QKDResult(
            shared_key=bytes(shared_key),
            error_rate=error_rate,
            security_metrics=security_metrics,
            timestamp=datetime.now()
        )
    
    def _create_entangled_pair(self, qubit1_idx: int, qubit2_idx: int) -> None:
        """Create an entangled Bell state between two qubits"""
        # Apply Hadamard to first qubit
        self.controller.apply_gate(self.quantum_gates['H'], qubit1_idx)
        # Apply CNOT
        self.controller.apply_gate(self.quantum_gates['X'], qubit2_idx)
    
    def _measure_in_basis(self, qubit_idx: int, basis: str) -> int:
        """Measure qubit in specified basis"""
        if basis == 'X':
            self.controller.apply_gate(self.quantum_gates['H'], qubit_idx)
        elif basis == 'Y':
            self.controller.apply_gate(self.quantum_gates['H'], qubit_idx)
            self.controller.apply_gate(self.quantum_gates['Z'], qubit_idx)
        
        state, _ = self.controller.measure(qubit_idx)
        return 1 if state == QubitState.EXCITED else 0
    
    def _estimate_error_rate(self, alice_bits: np.ndarray, bob_bits: List[int], 
                           matching_bases: np.ndarray) -> float:
        """Estimate error rate in the quantum channel"""
        matching_results = [alice_bits[i] == bob_bits[i] for i in range(len(alice_bits)) 
                          if matching_bases[i]]
        return 1 - sum(matching_results) / len(matching_results) if matching_results else 1.0
    
    def _calculate_eavesdropping_probability(self, error_rate: float) -> float:
        """Calculate probability of eavesdropping based on error rate"""
        return min(1.0, 2 * error_rate)
    
    def _process_e91_results(self, alice_results: List[int], bob_results: List[int],
                           alice_bases: np.ndarray, bob_bases: np.ndarray) -> Tuple[List[int], float]:
        """Process E91 measurement results to generate key and estimate errors"""
        matching_bases = alice_bases == bob_bases
        shared_key = []
        errors = 0
        
        for i in range(len(alice_results)):
            if matching_bases[i]:
                if alice_results[i] == bob_results[i]:
                    shared_key.append(0)
                else:
                    shared_key.append(1)
            else:
                if alice_results[i] != bob_results[i]:
                    errors += 1
        
        error_rate = errors / len(alice_results) if alice_results else 1.0
        return shared_key, error_rate
    
    def _calculate_entanglement_quality(self, alice_results: List[int], 
                                      bob_results: List[int]) -> float:
        """Calculate quality of entanglement based on correlation of results"""
        correlations = sum(1 for a, b in zip(alice_results, bob_results) if a == b)
        return correlations / len(alice_results) if alice_results else 0.0 
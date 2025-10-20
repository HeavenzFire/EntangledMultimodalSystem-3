from typing import List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import logging
from ..qubit_control import Qubit, QubitState, QuantumGate, QubitController

logger = logging.getLogger(__name__)

@dataclass
class EntanglementResult:
    """Results from multi-qubit entanglement operations"""
    success: bool
    state_type: str
    fidelity: float
    qubit_states: List[QubitState]
    verification_passed: bool

class MultiQubitEntanglement:
    """Multi-qubit entanglement operations implementation"""
    def __init__(self, controller: QubitController):
        self.controller = controller
        self.entanglement_gates = {
            'H': QuantumGate('H', (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]])),
            'X': QuantumGate('X', np.array([[0, 1], [1, 0]])),
            'CNOT': QuantumGate('CNOT', np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])),
            'Toffoli': QuantumGate('Toffoli', np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                                                       [0, 1, 0, 0, 0, 0, 0, 0],
                                                       [0, 0, 1, 0, 0, 0, 0, 0],
                                                       [0, 0, 0, 1, 0, 0, 0, 0],
                                                       [0, 0, 0, 0, 1, 0, 0, 0],
                                                       [0, 0, 0, 0, 0, 1, 0, 0],
                                                       [0, 0, 0, 0, 0, 0, 0, 1],
                                                       [0, 0, 0, 0, 0, 0, 1, 0]]))
        }
    
    def create_ghz_state(self, qubits: List[int]) -> EntanglementResult:
        """Create a GHZ state among multiple qubits"""
        try:
            # Prepare first qubit in |+> state
            self.controller.apply_gate(self.entanglement_gates['H'], qubits[0])
            
            # Apply CNOT gates to entangle remaining qubits
            for i in range(1, len(qubits)):
                self.controller.apply_gate(self.entanglement_gates['CNOT'], qubits[0], qubits[i])
            
            # Verify GHZ state
            fidelity = self._verify_ghz_state(qubits)
            
            # Get final states
            states = [self.controller.measure(q)[0] for q in qubits]
            
            return EntanglementResult(
                success=True,
                state_type='GHZ',
                fidelity=fidelity,
                qubit_states=states,
                verification_passed=fidelity > 0.95
            )
        except Exception as e:
            logger.error(f"Error creating GHZ state: {str(e)}")
            return EntanglementResult(
                success=False,
                state_type='GHZ',
                fidelity=0.0,
                qubit_states=[QubitState.GROUND] * len(qubits),
                verification_passed=False
            )
    
    def create_w_state(self, qubits: List[int]) -> EntanglementResult:
        """Create a W state among multiple qubits"""
        try:
            # Prepare first qubit in |1> state
            self.controller.apply_gate(self.entanglement_gates['X'], qubits[0])
            
            # Apply controlled operations to create W state
            for i in range(1, len(qubits)):
                self._apply_w_state_operation(qubits[0], qubits[i])
            
            # Verify W state
            fidelity = self._verify_w_state(qubits)
            
            # Get final states
            states = [self.controller.measure(q)[0] for q in qubits]
            
            return EntanglementResult(
                success=True,
                state_type='W',
                fidelity=fidelity,
                qubit_states=states,
                verification_passed=fidelity > 0.95
            )
        except Exception as e:
            logger.error(f"Error creating W state: {str(e)}")
            return EntanglementResult(
                success=False,
                state_type='W',
                fidelity=0.0,
                qubit_states=[QubitState.GROUND] * len(qubits),
                verification_passed=False
            )
    
    def create_cluster_state(self, qubits: List[int], 
                           connections: List[Tuple[int, int]]) -> EntanglementResult:
        """Create a cluster state with specified connections"""
        try:
            # Prepare all qubits in |+> state
            for qubit in qubits:
                self.controller.apply_gate(self.entanglement_gates['H'], qubit)
            
            # Apply CZ gates according to connections
            for q1, q2 in connections:
                self._apply_cz_gate(qubits[q1], qubits[q2])
            
            # Verify cluster state
            fidelity = self._verify_cluster_state(qubits, connections)
            
            # Get final states
            states = [self.controller.measure(q)[0] for q in qubits]
            
            return EntanglementResult(
                success=True,
                state_type='Cluster',
                fidelity=fidelity,
                qubit_states=states,
                verification_passed=fidelity > 0.95
            )
        except Exception as e:
            logger.error(f"Error creating cluster state: {str(e)}")
            return EntanglementResult(
                success=False,
                state_type='Cluster',
                fidelity=0.0,
                qubit_states=[QubitState.GROUND] * len(qubits),
                verification_passed=False
            )
    
    def _apply_w_state_operation(self, control_qubit: int, target_qubit: int) -> None:
        """Apply operation to create W state between two qubits"""
        # Apply controlled rotation
        self.controller.apply_gate(self.entanglement_gates['H'], target_qubit)
        self.controller.apply_gate(self.entanglement_gates['CNOT'], control_qubit, target_qubit)
    
    def _apply_cz_gate(self, qubit1: int, qubit2: int) -> None:
        """Apply controlled-Z gate between two qubits"""
        # Implement CZ using CNOT and Hadamard
        self.controller.apply_gate(self.entanglement_gates['H'], qubit2)
        self.controller.apply_gate(self.entanglement_gates['CNOT'], qubit1, qubit2)
        self.controller.apply_gate(self.entanglement_gates['H'], qubit2)
    
    def _verify_ghz_state(self, qubits: List[int]) -> float:
        """Verify GHZ state by checking correlations"""
        # Measure all qubits
        states = [self.controller.measure(q)[0] for q in qubits]
        
        # Check for perfect correlation
        if all(s == states[0] for s in states):
            return 1.0
        return 0.0
    
    def _verify_w_state(self, qubits: List[int]) -> float:
        """Verify W state by checking single excitation"""
        # Measure all qubits
        states = [self.controller.measure(q)[0] for q in qubits]
        
        # Check for exactly one excitation
        if sum(1 for s in states if s == QubitState.EXCITED) == 1:
            return 1.0
        return 0.0
    
    def _verify_cluster_state(self, qubits: List[int], 
                            connections: List[Tuple[int, int]]) -> float:
        """Verify cluster state by checking stabilizer measurements"""
        # Measure stabilizers for each connection
        stabilizer_results = []
        for q1, q2 in connections:
            # Apply Hadamard to both qubits
            self.controller.apply_gate(self.entanglement_gates['H'], qubits[q1])
            self.controller.apply_gate(self.entanglement_gates['H'], qubits[q2])
            
            # Measure and check correlation
            state1, _ = self.controller.measure(qubits[q1])
            state2, _ = self.controller.measure(qubits[q2])
            stabilizer_results.append(state1 == state2)
        
        # Calculate fidelity based on stabilizer measurements
        return sum(1 for r in stabilizer_results if r) / len(stabilizer_results) 
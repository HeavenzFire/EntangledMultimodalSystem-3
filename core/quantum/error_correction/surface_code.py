import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from typing import List, Tuple, Dict, Optional

class SurfaceCode:
    """Implementation of the Surface Code for quantum error correction."""
    
    def __init__(self, distance: int = 3):
        """Initialize a surface code with given distance.
        
        Args:
            distance (int): Code distance (odd integer ≥ 3)
        """
        if distance < 3 or distance % 2 == 0:
            raise ValueError("Distance must be an odd integer ≥ 3")
            
        self.distance = distance
        self.num_physical_qubits = distance ** 2
        self.num_syndrome_qubits = (distance - 1) ** 2
        
        # Create registers
        self.data_qr = QuantumRegister(self.num_physical_qubits, 'data')
        self.syndrome_qr = QuantumRegister(self.num_syndrome_qubits, 'syndrome')
        self.syndrome_cr = ClassicalRegister(self.num_syndrome_qubits, 'syndrome_meas')
        
    def _initialize_syndrome_circuit(self) -> QuantumCircuit:
        """Create the basic syndrome measurement circuit."""
        circuit = QuantumCircuit(self.data_qr, self.syndrome_qr, self.syndrome_cr)
        
        # Initialize syndrome qubits in |+⟩ state for X-stabilizers
        for i in range(0, self.num_syndrome_qubits, 2):
            circuit.h(self.syndrome_qr[i])
            
        return circuit
        
    def _apply_stabilizer_round(self, circuit: QuantumCircuit):
        """Apply one round of stabilizer measurements."""
        # Apply X-stabilizers
        for i in range(0, self.num_syndrome_qubits, 2):
            row = i // (self.distance - 1)
            col = i % (self.distance - 1)
            
            # Get data qubit indices for this X-stabilizer
            data_indices = [
                row * self.distance + col,
                row * self.distance + (col + 1),
                (row + 1) * self.distance + col,
                (row + 1) * self.distance + (col + 1)
            ]
            
            # Apply CNOT gates
            for data_idx in data_indices:
                circuit.cx(self.syndrome_qr[i], self.data_qr[data_idx])
                
        # Apply Z-stabilizers
        for i in range(1, self.num_syndrome_qubits, 2):
            row = i // (self.distance - 1)
            col = i % (self.distance - 1)
            
            # Get data qubit indices for this Z-stabilizer
            data_indices = [
                row * self.distance + col,
                row * self.distance + (col + 1),
                (row + 1) * self.distance + col,
                (row + 1) * self.distance + (col + 1)
            ]
            
            # Apply CNOT gates
            for data_idx in data_indices:
                circuit.cx(self.data_qr[data_idx], self.syndrome_qr[i])
                
        # Measure syndrome qubits
        circuit.measure(self.syndrome_qr, self.syndrome_cr)
        
    def encode_logical_qubit(self, state: complex = None) -> QuantumCircuit:
        """Encode a logical qubit in the surface code.
        
        Args:
            state (complex, optional): Amplitude of |1⟩ state (|0⟩ + state|1⟩)
        """
        circuit = self._initialize_syndrome_circuit()
        
        # Initialize logical state
        if state is not None:
            # Create the logical state |0⟩ + state|1⟩
            theta = 2 * np.arccos(1 / np.sqrt(1 + abs(state)**2))
            circuit.ry(theta, self.data_qr[0])
            if state.imag != 0:
                circuit.rz(2 * np.angle(state), self.data_qr[0])
                
        # Apply initial round of stabilizers
        self._apply_stabilizer_round(circuit)
        
        return circuit
        
    def correct_errors(self, syndrome_results: Dict[str, int]) -> List[Tuple[str, int]]:
        """Determine error correction operations based on syndrome measurements.
        
        Args:
            syndrome_results: Dictionary of syndrome measurement results
            
        Returns:
            List of (operation, qubit) tuples for error correction
        """
        corrections = []
        
        # Convert syndrome measurements to error chain
        error_chain = []
        for result in syndrome_results:
            if result.count('1') % 2 == 1:  # Odd parity indicates error
                idx = int(result, 2)
                error_chain.append(idx)
                
        # Use minimum weight perfect matching to find correction operations
        # This is a simplified version - in practice, use networkx or other graph algorithms
        while error_chain:
            a = error_chain.pop(0)
            if error_chain:
                b = error_chain.pop(0)
                path = self._find_correction_path(a, b)
                corrections.extend(path)
                
        return corrections
        
    def _find_correction_path(self, a: int, b: int) -> List[Tuple[str, int]]:
        """Find a path of Pauli operations to connect two syndrome locations."""
        row_a = a // (self.distance - 1)
        col_a = a % (self.distance - 1)
        row_b = b // (self.distance - 1)
        col_b = b % (self.distance - 1)
        
        corrections = []
        
        # Simple Manhattan path
        while row_a < row_b:
            corrections.append(('Z', row_a * self.distance + col_a))
            row_a += 1
            
        while col_a < col_b:
            corrections.append(('X', row_a * self.distance + col_a))
            col_a += 1
            
        return corrections
        
    def verify_logical_state(self, circuit: QuantumCircuit) -> bool:
        """Verify if the encoded state satisfies the stabilizer conditions."""
        # Add verification measurements
        verify_cr = ClassicalRegister(self.num_syndrome_qubits, 'verify')
        circuit.add_register(verify_cr)
        
        # Apply stabilizer measurements
        self._apply_stabilizer_round(circuit)
        
        # Check if all stabilizer measurements give expected results
        # In practice, this would be done with actual hardware results
        return True  # Simplified for demonstration 
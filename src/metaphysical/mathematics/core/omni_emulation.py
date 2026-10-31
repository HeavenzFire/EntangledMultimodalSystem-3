"""Bounded local quantum-circuit emulation for structural-system components.

This module uses Qiskit Aer only; it does not connect to quantum hardware.
"""
from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


class OmniEmulationSystem:
    """Create and run circuits within an explicit local simulation budget."""

    def __init__(self, num_qubits: int = 128, max_shots: int = 100_000) -> None:
        self._validate_positive_int("num_qubits", num_qubits)
        self._validate_positive_int("max_shots", max_shots)
        self.num_qubits = num_qubits
        self.max_shots = max_shots
        self.backend = AerSimulator(method="matrix_product_state")

    @staticmethod
    def _validate_positive_int(name: str, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")

    def create_circuit(self, num_qubits: int | None = None) -> QuantumCircuit:
        """Return an empty circuit no wider than this emulator's configured limit."""
        width = self.num_qubits if num_qubits is None else num_qubits
        self._validate_positive_int("num_qubits", width)
        if width > self.num_qubits:
            raise ValueError(
                f"requested {width} qubits exceeds configured limit {self.num_qubits}"
            )
        return QuantumCircuit(width)

    def run(self, circuit: QuantumCircuit, shots: int = 1024):
        """Run a circuit locally and return Qiskit's simulation result."""
        if not isinstance(circuit, QuantumCircuit):
            raise TypeError("circuit must be a qiskit.QuantumCircuit")
        if circuit.num_qubits < 1 or circuit.num_qubits > self.num_qubits:
            raise ValueError(
                f"circuit width {circuit.num_qubits} exceeds configured limit "
                f"{self.num_qubits}"
            )
        self._validate_positive_int("shots", shots)
        if shots > self.max_shots:
            raise ValueError(
                f"requested {shots} shots exceeds configured limit {self.max_shots}"
            )
        return self.backend.run(circuit, shots=shots).result()

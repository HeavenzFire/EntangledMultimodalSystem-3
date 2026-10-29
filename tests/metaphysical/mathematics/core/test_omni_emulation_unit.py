import pytest
from qiskit import QuantumCircuit

from src.metaphysical.mathematics.core.omni_emulation import OmniEmulationSystem


def test_create_circuit_respects_configured_width():
    emulator = OmniEmulationSystem(num_qubits=2)

    circuit = emulator.create_circuit()

    assert isinstance(circuit, QuantumCircuit)
    assert circuit.num_qubits == 2


def test_run_executes_a_bell_pair_circuit_locally():
    emulator = OmniEmulationSystem(num_qubits=2)
    circuit = emulator.create_circuit()
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()

    counts = emulator.run(circuit, shots=128).get_counts()

    assert sum(counts.values()) == 128
    assert set(counts).issubset({"00", "11"})
    assert counts.get("01", 0) == 0
    assert counts.get("10", 0) == 0


def test_create_circuit_rejects_width_above_limit():
    emulator = OmniEmulationSystem(num_qubits=2)

    with pytest.raises(ValueError, match="exceeds configured limit"):
        emulator.create_circuit(3)


def test_run_rejects_circuit_above_limit():
    emulator = OmniEmulationSystem(num_qubits=2)

    with pytest.raises(ValueError, match="exceeds configured limit"):
        emulator.run(QuantumCircuit(3))


def test_run_rejects_shots_above_limit():
    emulator = OmniEmulationSystem(num_qubits=2, max_shots=16)

    with pytest.raises(ValueError, match="exceeds configured limit"):
        emulator.run(QuantumCircuit(2), shots=17)

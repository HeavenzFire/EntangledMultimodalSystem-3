import numpy as np

from src.core.quantum_holographic_core import QuantumHolographicCore


def test_core_process_uses_real_bounded_local_components():
    core = QuantumHolographicCore({"num_qubits": 2, "error_correction": "none"})
    output, metrics = core.process(np.array([0.1, 0.2, 0.3]))

    assert output.size > 0
    assert np.all(np.isfinite(output))
    assert {
        "processing_speed",
        "energy_efficiency",
        "error_rate",
        "integration_score",
    } <= metrics.keys()
    assert core.qpu.quantum_params["num_qubits"] == 2
    assert core.holo.tensor_holography.encoding_matrix is None


def test_core_rejects_empty_or_nonfinite_input():
    core = QuantumHolographicCore({"num_qubits": 2, "error_correction": "none"})

    for invalid in (np.array([]), np.array([1.0, np.nan])):
        try:
            core.process(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid input must be rejected")


def test_entanglement_task_prepares_a_bell_pair():
    from qiskit.quantum_info import Statevector

    from src.core.quantum_processor import QuantumProcessor

    processor = QuantumProcessor(num_qubits=2, error_correction="none")
    circuit = processor._create_circuit(
        "entanglement",
        {
            "state_0": [1.0, 0.0],
            "state_1": [1.0, 0.0],
            "entangle_0_1": True,
        },
    )
    unmeasured = circuit.remove_final_measurements(inplace=False)
    probabilities = Statevector.from_instruction(unmeasured).probabilities()

    np.testing.assert_allclose(probabilities, [0.5, 0.0, 0.0, 0.5], atol=1e-8)

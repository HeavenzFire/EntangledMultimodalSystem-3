import numpy as np
import pytest

from research.reference_quantum_core import (
    evolve_lindblad, evolve_state, measure, qft_matrix,
    unitary_from_hamiltonian, validate_density_matrix,
)

X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def test_hamiltonian_evolution_is_unitary_and_norm_preserving():
    unitary = unitary_from_hamiltonian(X, 0.37)
    assert np.allclose(unitary.conj().T @ unitary, np.eye(2), atol=1e-12)
    final = evolve_state(np.array([1, 0]), X, 0.37)
    assert np.isclose(np.linalg.norm(final), 1.0, atol=1e-12)


def test_invalid_nonhermitian_hamiltonian_is_rejected():
    with pytest.raises(ValueError, match="Hermitian"):
        unitary_from_hamiltonian(np.array([[0, 1], [0, 0]]), 1.0)


def test_born_measurement_is_seed_reproducible_and_balanced():
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    first, probabilities = measure(plus, Z, shots=20000, seed=7)
    second, _ = measure(plus, Z, shots=20000, seed=7)
    assert np.array_equal(first, second)
    assert np.allclose(probabilities, [0.5, 0.5])
    assert abs(np.mean(first == 1) - 0.5) < 0.02


def test_density_matrix_rejects_nonphysical_input():
    with pytest.raises(ValueError, match="positive semidefinite"):
        validate_density_matrix(np.diag([1.2, -0.2]))


def test_amplitude_damping_matches_analytic_solution():
    excited = np.array([[0, 0], [0, 1]], dtype=complex)
    lowering = np.array([[0, 1], [0, 0]], dtype=complex)
    rate = 0.37
    times, states = evolve_lindblad(
        excited, np.zeros((2, 2)), [lowering], [rate],
        t_span=(0.0, 8.0), points=101,
    )
    analytic_excited_population = np.exp(-rate * times)
    assert np.max(np.abs(states[:, 1, 1].real - analytic_excited_population)) < 2e-8
    assert np.max(np.abs(np.trace(states, axis1=1, axis2=2) - 1)) < 2e-8
    assert all(np.allclose(rho, rho.conj().T, atol=2e-8) for rho in states)


@pytest.mark.parametrize("n_qubits", [1, 2, 3, 4])
def test_qft_matrix_is_unitary(n_qubits):
    qft = qft_matrix(n_qubits)
    assert np.allclose(qft.conj().T @ qft, np.eye(2 ** n_qubits), atol=1e-12)


def test_lindblad_rejects_negative_rate():
    rho = np.diag([1.0, 0.0])
    lowering = np.array([[0, 1], [0, 0]], dtype=complex)
    with pytest.raises(ValueError, match="nonnegative"):
        evolve_lindblad(rho, np.zeros((2, 2)), [lowering], [-0.1])

"""Small, dependency-light reference routines for finite-dimensional quantum mechanics.

These routines are intended for numerical verification and small test fixtures,
not as a production quantum SDK or evidence of a physical hardware advantage.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm


def _square(matrix, name):
    matrix = np.asarray(matrix, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    return matrix


def _hermitian(matrix, name, atol=1e-10):
    matrix = _square(matrix, name)
    if not np.allclose(matrix, matrix.conj().T, atol=atol, rtol=0):
        raise ValueError(f"{name} must be Hermitian")
    return matrix


def validate_state(state, atol=1e-10):
    state = np.asarray(state, dtype=np.complex128)
    if state.ndim != 1 or state.size == 0:
        raise ValueError("state must be a non-empty vector")
    norm = np.linalg.norm(state)
    if not np.isfinite(norm) or not np.isclose(norm, 1.0, atol=atol, rtol=0):
        raise ValueError("state must have unit norm")
    return state


def unitary_from_hamiltonian(hamiltonian, time, hbar=1.0):
    hamiltonian = _hermitian(hamiltonian, "Hamiltonian")
    if not np.isfinite(time) or not np.isfinite(hbar) or hbar <= 0:
        raise ValueError("time must be finite and hbar must be positive")
    return expm(-1j * hamiltonian * time / hbar)


def evolve_state(state, hamiltonian, time, hbar=1.0):
    state = validate_state(state)
    hamiltonian = _hermitian(hamiltonian, "Hamiltonian")
    if hamiltonian.shape[0] != state.size:
        raise ValueError("Hamiltonian/state dimension mismatch")
    evolved = unitary_from_hamiltonian(hamiltonian, time, hbar) @ state
    if not np.isclose(np.linalg.norm(evolved), 1.0, atol=1e-9, rtol=0):
        raise ArithmeticError("unitary evolution failed norm-conservation check")
    return evolved


def measure(state, observable, shots=1, seed=0):
    """Sample observable eigenvalues using Born probabilities and a fixed seed."""
    state = validate_state(state)
    observable = _hermitian(observable, "observable")
    if observable.shape[0] != state.size:
        raise ValueError("observable/state dimension mismatch")
    if not isinstance(shots, int) or shots < 1:
        raise ValueError("shots must be a positive integer")
    eigenvalues, eigenvectors = np.linalg.eigh(observable)
    probabilities = np.abs(eigenvectors.conj().T @ state) ** 2
    probabilities = np.maximum(probabilities, 0.0)
    probabilities /= probabilities.sum()
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(eigenvalues), size=shots, p=probabilities)
    return eigenvalues[indices], probabilities


def validate_density_matrix(rho, atol=1e-10):
    rho = _square(rho, "density matrix")
    if not np.allclose(rho, rho.conj().T, atol=atol, rtol=0):
        raise ValueError("density matrix must be Hermitian")
    if not np.isclose(np.trace(rho), 1.0, atol=atol, rtol=0):
        raise ValueError("density matrix must have trace one")
    if np.linalg.eigvalsh(rho).min() < -atol:
        raise ValueError("density matrix must be positive semidefinite")
    return rho


def _lindblad_rhs(_time, flat_rho, hamiltonian, jumps, rates, hbar):
    n = hamiltonian.shape[0]
    rho = flat_rho.reshape((n, n))
    derivative = -1j * (hamiltonian @ rho - rho @ hamiltonian) / hbar
    for jump, rate in zip(jumps, rates):
        jump_dagger_jump = jump.conj().T @ jump
        derivative += rate * (
            jump @ rho @ jump.conj().T
            - 0.5 * (jump_dagger_jump @ rho + rho @ jump_dagger_jump)
        )
    return derivative.reshape(-1)


def evolve_lindblad(
    rho_initial, hamiltonian, jumps=(), rates=(),
    t_span=(0.0, 1.0), points=101, hbar=1.0,
):
    """Integrate a finite-dimensional GKSL/Lindblad master equation."""
    rho_initial = validate_density_matrix(rho_initial)
    hamiltonian = _hermitian(hamiltonian, "Hamiltonian")
    if hamiltonian.shape != rho_initial.shape:
        raise ValueError("Hamiltonian/density-matrix dimension mismatch")
    if len(jumps) != len(rates):
        raise ValueError("each jump operator must have exactly one rate")
    if not np.isfinite(hbar) or hbar <= 0:
        raise ValueError("hbar must be positive")
    if not isinstance(points, int) or points < 2:
        raise ValueError("points must be an integer of at least two")
    if len(t_span) != 2 or not np.all(np.isfinite(t_span)) or t_span[1] < t_span[0]:
        raise ValueError("t_span must be a finite, nondecreasing pair")

    checked_jumps = []
    checked_rates = []
    for jump, rate in zip(jumps, rates):
        jump = _square(jump, "jump operator")
        if jump.shape != rho_initial.shape:
            raise ValueError("jump operator/density-matrix dimension mismatch")
        if not np.isfinite(rate) or rate < 0:
            raise ValueError("Lindblad rates must be finite and nonnegative")
        checked_jumps.append(jump)
        checked_rates.append(float(rate))

    times = np.linspace(t_span[0], t_span[1], points)
    solution = solve_ivp(
        _lindblad_rhs, t_span, rho_initial.reshape(-1), t_eval=times,
        args=(hamiltonian, checked_jumps, checked_rates, hbar),
        rtol=1e-10, atol=1e-12,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    states = solution.y.T.reshape((-1,) + rho_initial.shape)
    for rho in states:
        validate_density_matrix(rho, atol=2e-8)
    return times, states


def qft_matrix(n_qubits):
    """Return the standard positive-exponent QFT matrix on 2**n_qubits states."""
    if not isinstance(n_qubits, int) or n_qubits < 1:
        raise ValueError("n_qubits must be a positive integer")
    dimension = 2 ** n_qubits
    indices = np.arange(dimension)
    return np.exp(2j * np.pi * np.outer(indices, indices) / dimension) / np.sqrt(dimension)

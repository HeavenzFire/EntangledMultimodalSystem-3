"""Regression tests for the deterministic quantum-holographic integration adapter."""
import numpy as np
import pytest

from src.core.quantum_consciousness import QuantumConsciousness


def test_integrate_is_deterministic_and_returns_finite_vector():
    adapter = QuantumConsciousness()
    quantum = np.array([1.0, 0.0, -1.0])
    holographic = np.array([0.5, 0.5, 0.5])

    first = adapter.integrate(quantum, holographic)
    first_metrics = adapter.get_metrics()
    second = adapter.integrate(quantum, holographic)

    np.testing.assert_allclose(first, second)
    assert np.all(np.isfinite(first))
    assert np.isclose(np.linalg.norm(first), 1.0)
    assert first_metrics == adapter.get_metrics()
    assert set(first_metrics) == {"phi_score", "integration_score"}


@pytest.mark.parametrize(
    "weights",
    [
        ({"quantum_weight": -1.0}),
        ({"quantum_weight": 0.0, "holographic_weight": 0.0, "neural_weight": 0.0}),
        ({"quantum_weight": float("nan")}),
    ],
)
def test_invalid_weights_are_rejected(weights):
    with pytest.raises(ValueError):
        QuantumConsciousness(**weights)


@pytest.mark.parametrize("value", [[], [float("nan")], [float("inf")]])
def test_invalid_integration_vectors_are_rejected(value):
    adapter = QuantumConsciousness()
    with pytest.raises(ValueError):
        adapter.integrate(value, [1.0])


def test_calibration_validates_range_and_reset_clears_metrics():
    adapter = QuantumConsciousness()
    adapter.integrate([1.0, 0.0], [0.0, 1.0])

    assert adapter.calibrate(0.9)["score"] == 1.0
    with pytest.raises(ValueError):
        adapter.calibrate(1.1)

    adapter.reset()
    assert adapter.get_metrics() == {"phi_score": 0.0, "integration_score": 0.0}

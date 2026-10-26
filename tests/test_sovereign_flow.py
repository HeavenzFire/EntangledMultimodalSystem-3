"""Contract tests for the implemented sovereign-flow API.

These tests intentionally validate observable behavior in
src.quantum.purification.sovereign_flow rather than assumptions about
unimplemented geometry/consciousness methods.
"""
import numpy as np
import pytest

from src.quantum.purification.sovereign_flow import (
    PurificationConfig,
    QuantumBackend,
    SovereignFlow,
)


@pytest.fixture
def config():
    return PurificationConfig(
        resonance_threshold=0.0,
        prime_numbers=[2, 3, 5, 7],
        max_iterations=12,
        phi=1.618033988749895,
    )


@pytest.fixture
def flow(config):
    return SovereignFlow(config)


def test_config_applies_explicit_values(config):
    assert config.resonance_threshold == 0.0
    assert config.prime_numbers == [2, 3, 5, 7]
    assert config.max_iterations == 12
    assert config.phi == pytest.approx(1.618033988749895)


def test_default_config_has_twelve_primes():
    config = PurificationConfig()
    assert len(config.prime_numbers) == 12
    assert config.max_iterations == 144


def test_detect_artifacts_initializes_state_and_returns_metrics(flow):
    result = flow.detect_ascension_artifacts()
    assert set(result) == {"resonance_level", "artifact_count", "critical_regions"}
    assert 0.0 <= result["resonance_level"] <= 1.0
    assert result["artifact_count"] == len(result["critical_regions"])
    assert flow.current_state.shape == (12, 12)
    assert np.iscomplexobj(flow.current_state)


def test_firewall_mutates_initialized_state_and_increments_counter(flow):
    flow.detect_ascension_artifacts()
    before = flow.current_state.copy()
    flow.activate_toroidal_firewall()
    assert flow.iteration_count == 1
    assert flow.current_state.shape == before.shape
    assert not np.array_equal(flow.current_state, before)


def test_clear_debris_preserves_state_shape(flow):
    flow.detect_ascension_artifacts()
    before_shape = flow.current_state.shape
    flow.clear_ascension_debris()
    assert flow.current_state.shape == before_shape
    assert np.isfinite(flow.current_state).all()


def test_metrics_are_well_formed_before_initialization(flow):
    metrics = flow.get_system_metrics()
    assert metrics == {
        "resonance_level": 0.0,
        "iteration_count": 0,
        "system_integrity": False,
        "state_energy": 0.0,
    }


def test_metrics_report_initialized_state(flow):
    flow.detect_ascension_artifacts()
    metrics = flow.get_system_metrics()
    assert 0.0 <= metrics["resonance_level"] <= 1.0
    assert metrics["iteration_count"] == 0
    assert metrics["state_energy"] > 0.0
    assert isinstance(metrics["system_integrity"], bool)


def test_quantum_backend_enum_values_are_stable():
    assert {backend.value for backend in QuantumBackend} == {
        "merkaba",
        "platonic",
        "vortex",
    }

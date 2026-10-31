import numpy as np

from src.core.holographic_processor import HolographicProcessor, TensorHolographyDB


def test_holography_database_does_not_allocate_dense_matrix():
    database = TensorHolographyDB()

    assert database.encoding_matrix is None

    original = np.array([0.0, 0.25, 0.5, 1.0])
    restored = database.decode(database.encode(original))

    assert restored.shape == original.shape
    np.testing.assert_allclose(restored, original, atol=1e-12)


def test_holographic_processor_project_returns_finite_vector_and_metrics():
    processor = HolographicProcessor()

    projected = processor.project(np.array([0.25, 0.5, 0.75, 1.0]))
    metrics = processor.get_metrics()

    assert projected.shape == (4,)
    assert np.all(np.isfinite(projected))
    assert {"fidelity", "speed", "efficiency", "error_rate"} <= metrics.keys()
    assert 0.0 <= metrics["fidelity"] <= 1.0
    assert processor.tensor_holography.encoding_matrix is None


def test_holographic_reset_remains_memory_bounded():
    processor = HolographicProcessor()
    processor.project(np.array([1.0, 0.0]))
    processor.reset()

    assert processor.tensor_holography.encoding_matrix is None
    assert processor.last_render_time is None

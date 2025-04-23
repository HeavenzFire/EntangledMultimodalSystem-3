import pytest
from datetime import datetime
from src.core.nde_testimony import NDETestimony, NDETestimonyDatabase, NDETestimonyProcessor
from src.core.quantum_beatitudes_engine import QuantumBeatitudesEngine
from src.core.temporal_quantum_state_projector import TemporalQuantumStateProjector
from src.core.spiritual_metrics_analyzer import SpiritualMetricsAnalyzer

@pytest.fixture
def sample_nde_testimony():
    return NDETestimony(
        experiencer="John Doe",
        age=45,
        narrative="I experienced a profound sense of peace and unity with all things...",
        key_themes=["peace", "unity", "light", "love"]
    )

@pytest.fixture
def quantum_engine():
    return QuantumBeatitudesEngine()

@pytest.fixture
def temporal_projector():
    return TemporalQuantumStateProjector()

@pytest.fixture
def spiritual_analyzer():
    return SpiritualMetricsAnalyzer()

@pytest.fixture
def nde_database(quantum_engine, temporal_projector, spiritual_analyzer):
    return NDETestimonyDatabase(quantum_engine, temporal_projector, spiritual_analyzer)

@pytest.fixture
def nde_processor(nde_database, quantum_engine, temporal_projector):
    return NDETestimonyProcessor(nde_database, quantum_engine, temporal_projector)

class TestNDETestimony:
    def test_initialization(self, sample_nde_testimony):
        assert sample_nde_testimony.experiencer == "John Doe"
        assert sample_nde_testimony.age == 45
        assert "peace" in sample_nde_testimony.key_themes
        assert sample_nde_testimony.quantum_signature is not None
    
    def test_quantum_signature_validation(self, sample_nde_testimony):
        assert sample_nde_testimony.validate_quantum_signature()
    
    def test_to_quantum_state(self, sample_nde_testimony, temporal_projector):
        quantum_state = sample_nde_testimony.to_quantum_state(temporal_projector)
        assert 'quantum_state' in quantum_state
        assert 'spiritual_metrics' in quantum_state
        assert 'temporal_signature' in quantum_state
        assert 'quantum_signature' in quantum_state

class TestNDETestimonyDatabase:
    def test_add_testimony(self, nde_database, sample_nde_testimony):
        assert nde_database.add_testimony(sample_nde_testimony)
        assert len(nde_database.testimonies) == 1
    
    def test_query(self, nde_database, sample_nde_testimony):
        nde_database.add_testimony(sample_nde_testimony)
        results = nde_database.query(["peace", "unity"])
        assert len(results) == 1
        assert results[0] == sample_nde_testimony
    
    def test_analyze_quantum_patterns(self, nde_database, sample_nde_testimony):
        nde_database.add_testimony(sample_nde_testimony)
        patterns = nde_database.analyze_quantum_patterns()
        assert 'quantum_states' in patterns
        assert 'spiritual_metrics' in patterns
        assert 'temporal_evolution' in patterns
    
    def test_get_quantum_constraints(self, nde_database, sample_nde_testimony):
        nde_database.add_testimony(sample_nde_testimony)
        constraints = nde_database.get_quantum_constraints()
        assert 'quantum_constraints' in constraints
        assert 'spiritual_constraints' in constraints
        assert 'temporal_constraints' in constraints

class TestNDETestimonyProcessor:
    def test_process_testimony(self, nde_processor, sample_nde_testimony):
        result = nde_processor.process_testimony(sample_nde_testimony)
        assert 'quantum_state' in result
        assert 'spiritual_metrics' in result
        assert 'temporal_signature' in result
        assert 'quantum_signature' in result
    
    def test_generate_insights(self, nde_processor, sample_nde_testimony):
        nde_processor.process_testimony(sample_nde_testimony)
        insights = nde_processor.generate_insights(["peace", "unity"])
        assert 'testimonies' in insights
        assert 'quantum_patterns' in insights
        assert 'spiritual_insights' in insights
        assert 'temporal_evolution' in insights
    
    def test_invalid_quantum_signature(self, nde_processor):
        invalid_testimony = NDETestimony(
            experiencer="Invalid",
            age=0,
            narrative="Invalid",
            key_themes=[],
            quantum_signature="invalid"
        )
        with pytest.raises(ValueError):
            nde_processor.process_testimony(invalid_testimony) 
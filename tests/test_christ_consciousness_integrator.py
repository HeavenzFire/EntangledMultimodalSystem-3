import pytest
import numpy as np
from src.core.christ_consciousness_integrator import ChristConsciousnessIntegrator
from src.core.nde_testimony import NDETestimony, NDETestimonyDatabase
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
def christ_consciousness_integrator(nde_database, quantum_engine, temporal_projector, spiritual_analyzer):
    return ChristConsciousnessIntegrator(nde_database, quantum_engine, temporal_projector, spiritual_analyzer)

class TestChristConsciousnessIntegrator:
    def test_initialization(self, christ_consciousness_integrator):
        assert christ_consciousness_integrator.nde_db is not None
        assert christ_consciousness_integrator.quantum_engine is not None
        assert christ_consciousness_integrator.temporal_projector is not None
        assert christ_consciousness_integrator.spiritual_analyzer is not None
    
    def test_integrate_nde_with_christ_consciousness(self, christ_consciousness_integrator, sample_nde_testimony):
        result = christ_consciousness_integrator.integrate_nde_with_christ_consciousness(sample_nde_testimony)
        assert 'quantum_state' in result
        assert 'spiritual_metrics' in result
        assert 'insights' in result
        assert 'temporal_signature' in result
        assert 'quantum_signature' in result
    
    def test_apply_christ_patterns(self, christ_consciousness_integrator, sample_nde_testimony):
        quantum_state = sample_nde_testimony.to_quantum_state(christ_consciousness_integrator.temporal_projector)
        christ_patterns = christ_consciousness_integrator._apply_christ_patterns(quantum_state['quantum_state'])
        assert isinstance(christ_patterns, np.ndarray)
        assert christ_patterns.shape == quantum_state['quantum_state'].shape
    
    def test_generate_christ_insights(self, christ_consciousness_integrator, sample_nde_testimony):
        quantum_state = sample_nde_testimony.to_quantum_state(christ_consciousness_integrator.temporal_projector)
        christ_patterns = christ_consciousness_integrator._apply_christ_patterns(quantum_state['quantum_state'])
        spiritual_metrics = christ_consciousness_integrator.spiritual_analyzer.analyze_metrics(quantum_state['spiritual_metrics'])
        insights = christ_consciousness_integrator._generate_christ_insights(christ_patterns, spiritual_metrics)
        assert 'pattern_insights' in insights
        assert 'spiritual_insights' in insights
        assert 'christ_consciousness' in insights
    
    def test_extract_christ_consciousness(self, christ_consciousness_integrator):
        pattern_analysis = {
            'unconditional_love': 0.8,
            'compassion': 0.7,
            'unity': 0.9,
            'forgiveness': 0.6,
            'peace': 0.8
        }
        spiritual_insights = {
            'depth': 0.7,
            'connection': 0.8
        }
        christ_consciousness = christ_consciousness_integrator._extract_christ_consciousness(pattern_analysis, spiritual_insights)
        assert 'unconditional_love' in christ_consciousness
        assert 'compassion' in christ_consciousness
        assert 'unity' in christ_consciousness
        assert 'forgiveness' in christ_consciousness
        assert 'peace' in christ_consciousness
        assert 'spiritual_depth' in christ_consciousness
        assert 'divine_connection' in christ_consciousness
    
    def test_analyze_collective_consciousness(self, christ_consciousness_integrator, sample_nde_testimony):
        christ_consciousness_integrator.nde_db.add_testimony(sample_nde_testimony)
        result = christ_consciousness_integrator.analyze_collective_consciousness()
        assert 'collective_patterns' in result
        assert 'collective_metrics' in result
        assert 'collective_insight' in result
        assert 'christ_consciousness_level' in result
    
    def test_generate_collective_insight(self, christ_consciousness_integrator):
        collective_patterns = {
            'collective_love': 0.8,
            'collective_compassion': 0.7,
            'collective_unity': 0.9,
            'collective_forgiveness': 0.6,
            'collective_peace': 0.8
        }
        collective_metrics = [{
            'depth': 0.7,
            'connection': 0.8
        }]
        insight = christ_consciousness_integrator._generate_collective_insight(collective_patterns, collective_metrics)
        assert 'pattern_insights' in insight
        assert 'spiritual_insights' in insight
        assert 'collective_consciousness' in insight
    
    def test_extract_collective_consciousness(self, christ_consciousness_integrator):
        pattern_analysis = {
            'collective_love': 0.8,
            'collective_compassion': 0.7,
            'collective_unity': 0.9,
            'collective_forgiveness': 0.6,
            'collective_peace': 0.8
        }
        spiritual_insights = {
            'collective_depth': 0.7,
            'collective_connection': 0.8
        }
        collective_consciousness = christ_consciousness_integrator._extract_collective_consciousness(pattern_analysis, spiritual_insights)
        assert 'collective_love' in collective_consciousness
        assert 'collective_compassion' in collective_consciousness
        assert 'collective_unity' in collective_consciousness
        assert 'collective_forgiveness' in collective_consciousness
        assert 'collective_peace' in collective_consciousness
        assert 'collective_spiritual_depth' in collective_consciousness
        assert 'collective_divine_connection' in collective_consciousness
    
    def test_calculate_christ_consciousness_level(self, christ_consciousness_integrator):
        collective_insight = {
            'collective_consciousness': {
                'collective_love': 0.8,
                'collective_compassion': 0.7,
                'collective_unity': 0.9,
                'collective_forgiveness': 0.6,
                'collective_peace': 0.8,
                'collective_spiritual_depth': 0.7,
                'collective_divine_connection': 0.8
            }
        }
        level = christ_consciousness_integrator._calculate_christ_consciousness_level(collective_insight)
        assert isinstance(level, float)
        assert 0 <= level <= 1 
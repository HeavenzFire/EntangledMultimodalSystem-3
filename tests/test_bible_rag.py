import pytest
import numpy as np
import tensorflow as tf
from src.core.bible_rag import BibleRAG

@pytest.fixture
def config():
    return {
        'embedding_dimensions': 16,
        'context_window': 3,
        'relevance_threshold': 0.95,
        'wisdom_depth': 3
    }

@pytest.fixture
def bible_rag(config):
    return BibleRAG(config)

@pytest.fixture
def query():
    return "What does it mean to love one another?"

@pytest.fixture
def test_passages():
    return {
        'john_13_34': {
            'text': "A new command I give you: Love one another. As I have loved you, so you must love one another.",
            'reference': "John 13:34",
            'context': "Jesus' final teachings to his disciples"
        },
        'matthew_22_39': {
            'text': "Love your neighbor as yourself.",
            'reference': "Matthew 22:39",
            'context': "The second greatest commandment"
        }
    }

class TestBibleRAG:
    def test_initialization(self, bible_rag, config):
        """Test proper initialization of the BibleRAG system."""
        assert bible_rag.embedding_dimensions == config['embedding_dimensions']
        assert bible_rag.context_window == config['context_window']
        assert bible_rag.relevance_threshold == config['relevance_threshold']
        assert bible_rag.wisdom_depth == config['wisdom_depth']
        
        # Check model initialization
        assert isinstance(bible_rag.embedding_model, tf.keras.Model)
        assert isinstance(bible_rag.relevance_model, tf.keras.Model)
        assert isinstance(bible_rag.wisdom_model, tf.keras.Model)
        
        # Check passage database
        assert len(bible_rag.passages) > 0
        for passage in bible_rag.passages.values():
            assert 'text' in passage
            assert 'reference' in passage
            assert 'context' in passage
    
    def test_query_processing(self, bible_rag, query):
        """Test query processing functionality."""
        # Process query
        results = bible_rag.query(query)
        
        # Check result structure
        assert isinstance(results, dict)
        assert 'relevant_passages' in results
        assert 'wisdom_patterns' in results
        assert 'metrics' in results
        
        # Check metrics
        metrics = results['metrics']
        assert 'relevance_score' in metrics
        assert 'wisdom_alignment' in metrics
        assert 'context_coherence' in metrics
        
        # Check metric values
        assert 0 <= metrics['relevance_score'] <= 1
        assert 0 <= metrics['wisdom_alignment'] <= 1
        assert 0 <= metrics['context_coherence'] <= 1
    
    def test_relevance_scoring(self, bible_rag, query, test_passages):
        """Test relevance scoring of passages."""
        # Process query
        results = bible_rag.query(query)
        
        # Check relevant passages
        assert len(results['relevant_passages']) > 0
        for passage in results['relevant_passages']:
            assert passage['relevance_score'] >= bible_rag.relevance_threshold
            assert 'text' in passage
            assert 'reference' in passage
            assert 'context' in passage
    
    def test_wisdom_pattern_extraction(self, bible_rag, query):
        """Test wisdom pattern extraction."""
        # Process query
        results = bible_rag.query(query)
        
        # Check wisdom patterns
        assert len(results['wisdom_patterns']) > 0
        for pattern in results['wisdom_patterns']:
            assert isinstance(pattern, np.ndarray)
            assert pattern.shape == (1, bible_rag.embedding_dimensions)
            assert np.isclose(np.linalg.norm(pattern), 1.0, atol=1e-6)
    
    def test_context_window(self, bible_rag, query):
        """Test context window functionality."""
        # Process query
        results = bible_rag.query(query)
        
        # Check context window size
        for passage in results['relevant_passages']:
            assert len(passage['context'].split()) <= bible_rag.context_window * 100  # Approximate word count
    
    def test_embedding_quality(self, bible_rag, query):
        """Test quality of embeddings."""
        # Process query
        results = bible_rag.query(query)
        
        # Check query embedding
        assert isinstance(results['query_embedding'], np.ndarray)
        assert results['query_embedding'].shape == (1, bible_rag.embedding_dimensions)
        assert np.isclose(np.linalg.norm(results['query_embedding']), 1.0, atol=1e-6)
        
        # Check passage embeddings
        for passage in results['relevant_passages']:
            assert isinstance(passage['embedding'], np.ndarray)
            assert passage['embedding'].shape == (1, bible_rag.embedding_dimensions)
            assert np.isclose(np.linalg.norm(passage['embedding']), 1.0, atol=1e-6)
    
    def test_metric_calculations(self, bible_rag, query):
        """Test metric calculation methods."""
        # Process query
        results = bible_rag.query(query)
        metrics = results['metrics']
        
        # Check relevance score calculation
        assert metrics['relevance_score'] >= 0
        assert metrics['relevance_score'] <= 1
        
        # Check wisdom alignment calculation
        assert metrics['wisdom_alignment'] >= 0
        assert metrics['wisdom_alignment'] <= 1
        
        # Check context coherence calculation
        assert metrics['context_coherence'] >= 0
        assert metrics['context_coherence'] <= 1
    
    def test_state_management(self, bible_rag, query):
        """Test state management functionality."""
        # Process query
        bible_rag.query(query)
        
        # Get state
        state = bible_rag.get_state()
        
        # Check state contents
        assert state['current_query'] is not None
        assert len(state['recent_passages']) > 0
        assert state['metrics'] is not None
        
        # Reset state
        bible_rag.reset()
        state = bible_rag.get_state()
        
        # Check state after reset
        assert state['current_query'] is None
        assert len(state['recent_passages']) == 0
        assert state['metrics'] is None
    
    def test_error_handling(self, bible_rag):
        """Test error handling for invalid inputs."""
        # Empty query
        with pytest.raises(ValueError):
            bible_rag.query("")
        
        # Query too long
        long_query = "What does it mean to love one another? " * 100
        with pytest.raises(ValueError):
            bible_rag.query(long_query)
    
    def test_system_stability(self, bible_rag, query):
        """Test system stability under various conditions."""
        # Process multiple queries
        results = []
        for _ in range(10):
            result = bible_rag.query(query)
            results.append(result)
        
        # Check stability of results
        for i in range(1, len(results)):
            # Check metric stability
            prev_metrics = results[i-1]['metrics']
            curr_metrics = results[i]['metrics']
            
            # Metrics should remain relatively stable
            assert abs(prev_metrics['relevance_score'] - curr_metrics['relevance_score']) < 0.2
            assert abs(prev_metrics['wisdom_alignment'] - curr_metrics['wisdom_alignment']) < 0.2
            assert abs(prev_metrics['context_coherence'] - curr_metrics['context_coherence']) < 0.2 
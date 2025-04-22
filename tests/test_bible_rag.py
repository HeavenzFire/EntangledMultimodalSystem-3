import unittest
import numpy as np
from src.core.bible_rag import BibleRAG
import time

class TestBibleRAG(unittest.TestCase):
    def setUp(self):
        """Set up test configuration and data."""
        self.config = {
            'embedding_dimensions': 768,
            'context_window': 512,
            'relevance_threshold': 0.85,
            'wisdom_depth': 12
        }
        
        # Initialize RAG system
        self.rag = BibleRAG(self.config)
        
        # Sample queries for testing
        self.queries = [
            "What does the Bible say about love?",
            "How should we treat our neighbors?",
            "What is the meaning of life according to scripture?"
        ]
    
    def test_initialization(self):
        """Test RAG system initialization."""
        self.assertEqual(self.rag.config['embedding_dimensions'], self.config['embedding_dimensions'])
        self.assertEqual(self.rag.config['context_window'], self.config['context_window'])
        self.assertEqual(self.rag.config['relevance_threshold'], self.config['relevance_threshold'])
        self.assertEqual(self.rag.config['wisdom_depth'], self.config['wisdom_depth'])
        
        # Check model initialization
        self.assertIsNotNone(self.rag.embedding_model)
        self.assertIsNotNone(self.rag.relevance_model)
        self.assertIsNotNone(self.rag.wisdom_model)
        
        # Check initial state
        initial_state = self.rag.get_state()
        self.assertIsNone(initial_state['query'])
        self.assertIsNone(initial_state['relevant_passages'])
        self.assertIsNone(initial_state['wisdom_patterns'])
        self.assertIsNone(initial_state['metrics'])
    
    def test_query_processing(self):
        """Test query processing functionality."""
        for query in self.queries:
            result = self.rag.query(query)
            
            # Check result structure
            self.assertIn('relevant_passages', result)
            self.assertIn('wisdom_patterns', result)
            self.assertIn('metrics', result)
            
            # Check state
            state = self.rag.get_state()
            self.assertIsNotNone(state['query'])
            self.assertIsNotNone(state['relevant_passages'])
            self.assertIsNotNone(state['wisdom_patterns'])
            self.assertIsNotNone(state['metrics'])
            
            # Check metrics
            metrics = self.rag.get_metrics()
            self.assertGreaterEqual(metrics['relevance_score'], 0.0)
            self.assertLessEqual(metrics['relevance_score'], 1.0)
            self.assertGreaterEqual(metrics['wisdom_alignment'], 0.0)
            self.assertLessEqual(metrics['wisdom_alignment'], 1.0)
            self.assertGreaterEqual(metrics['context_coherence'], 0.0)
            self.assertLessEqual(metrics['context_coherence'], 1.0)
    
    def test_state_management(self):
        """Test state management functionality."""
        # Process query
        self.rag.query(self.queries[0])
        
        # Get state
        state = self.rag.get_state()
        self.assertIsNotNone(state['query'])
        self.assertIsNotNone(state['relevant_passages'])
        self.assertIsNotNone(state['wisdom_patterns'])
        self.assertIsNotNone(state['metrics'])
        
        # Get metrics
        metrics = self.rag.get_metrics()
        self.assertGreaterEqual(metrics['relevance_score'], 0.0)
        self.assertLessEqual(metrics['relevance_score'], 1.0)
        self.assertGreaterEqual(metrics['wisdom_alignment'], 0.0)
        self.assertLessEqual(metrics['wisdom_alignment'], 1.0)
        self.assertGreaterEqual(metrics['context_coherence'], 0.0)
        self.assertLessEqual(metrics['context_coherence'], 1.0)
        
        # Reset state
        self.rag.reset()
        state = self.rag.get_state()
        self.assertIsNone(state['query'])
        self.assertIsNone(state['relevant_passages'])
        self.assertIsNone(state['wisdom_patterns'])
        self.assertIsNone(state['metrics'])
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test empty query
        with self.assertRaises(ValueError):
            self.rag.query("")
        
        # Test None query
        with self.assertRaises(ValueError):
            self.rag.query(None)
        
        # Test query with only whitespace
        with self.assertRaises(ValueError):
            self.rag.query("   ")
    
    def test_relevance_scoring(self):
        """Test relevance scoring functionality."""
        # Process query
        self.rag.query(self.queries[0])
        
        # Get relevant passages
        state = self.rag.get_state()
        relevant_passages = state['relevant_passages']
        
        # Check passage properties
        for passage in relevant_passages:
            self.assertIn('text', passage)
            self.assertIn('reference', passage)
            self.assertIn('relevance_score', passage)
            self.assertGreaterEqual(passage['relevance_score'], self.config['relevance_threshold'])
    
    def test_wisdom_extraction(self):
        """Test wisdom pattern extraction."""
        # Process query
        self.rag.query(self.queries[0])
        
        # Get wisdom patterns
        state = self.rag.get_state()
        wisdom_patterns = state['wisdom_patterns']
        
        # Check pattern properties
        for pattern in wisdom_patterns:
            self.assertIn('pattern', pattern)
            self.assertIn('alignment_score', pattern)
            self.assertGreaterEqual(pattern['alignment_score'], 0.0)
            self.assertLessEqual(pattern['alignment_score'], 1.0)
    
    def test_metric_calculations(self):
        """Test metric calculation methods."""
        # Process query
        self.rag.query(self.queries[0])
        
        # Get state
        state = self.rag.get_state()
        
        # Test relevance score calculation
        relevance_score = self.rag._calculate_relevance_score(
            state['relevant_passages']
        )
        self.assertGreaterEqual(relevance_score, 0.0)
        self.assertLessEqual(relevance_score, 1.0)
        
        # Test wisdom alignment calculation
        wisdom_alignment = self.rag._calculate_wisdom_alignment(
            state['wisdom_patterns']
        )
        self.assertGreaterEqual(wisdom_alignment, 0.0)
        self.assertLessEqual(wisdom_alignment, 1.0)
        
        # Test context coherence calculation
        context_coherence = self.rag._calculate_context_coherence(
            state['relevant_passages'],
            state['wisdom_patterns']
        )
        self.assertGreaterEqual(context_coherence, 0.0)
        self.assertLessEqual(context_coherence, 1.0)
    
    def test_model_architecture(self):
        """Test model architectures."""
        # Check embedding model
        self.assertEqual(
            self.rag.embedding_model.input_shape[1],
            self.config['context_window']
        )
        self.assertEqual(
            self.rag.embedding_model.output_shape[1],
            self.config['embedding_dimensions']
        )
        
        # Check relevance model
        self.assertEqual(
            self.rag.relevance_model.input_shape[1],
            self.config['embedding_dimensions']
        )
        self.assertEqual(
            self.rag.relevance_model.output_shape[1],
            1
        )
        
        # Check wisdom model
        self.assertEqual(
            self.rag.wisdom_model.input_shape[1],
            self.config['embedding_dimensions']
        )
        self.assertEqual(
            self.rag.wisdom_model.output_shape[1],
            self.config['wisdom_depth']
        )
    
    def test_performance(self):
        """Test processing performance."""
        # Measure query processing time
        start_time = time.time()
        self.rag.query(self.queries[0])
        processing_time = time.time() - start_time
        
        # Check processing time is reasonable
        self.assertLess(processing_time, 1.0)  # Should process within 1 second
        
        # Test performance with different query lengths
        for length in [50, 100, 200]:
            query = " ".join(["test"] * length)
            start_time = time.time()
            self.rag.query(query)
            processing_time = time.time() - start_time
            self.assertLess(processing_time, 2.0)  # Should process within 2 seconds for longer queries
        
        # Test performance with multiple queries
        start_time = time.time()
        for query in self.queries:
            self.rag.query(query)
        total_time = time.time() - start_time
        self.assertLess(total_time, 5.0)  # Should process multiple queries within 5 seconds

if __name__ == '__main__':
    unittest.main() 
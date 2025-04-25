import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple
import logging

class SpiritualMetricsAnalyzer:
    """
    Analyzes and interprets spiritual metrics from the Christ Consciousness system.
    Provides insights into the spiritual dimensions of quantum and holographic processing.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the SpiritualMetricsAnalyzer.
        
        Args:
            config: Configuration dictionary containing:
                - metric_dimensions: Number of dimensions for metric analysis
                - temporal_depth: Depth of temporal analysis
                - spiritual_threshold: Threshold for spiritual significance
                - insight_depth: Depth of spiritual insight analysis
        """
        self.metric_dimensions = config.get('metric_dimensions', 16)
        self.temporal_depth = config.get('temporal_depth', 3)
        self.spiritual_threshold = config.get('spiritual_threshold', 0.85)
        self.insight_depth = config.get('insight_depth', 3)
        
        # Initialize models
        self._build_metric_analysis_model()
        self._build_spiritual_insight_model()
        
        # Initialize state
        self.state = {
            'current_metrics': None,
            'temporal_metrics': [],
            'spiritual_insights': [],
            'analysis_results': None
        }
        
        logging.info("SpiritualMetricsAnalyzer initialized successfully")
    
    def _build_metric_analysis_model(self) -> None:
        """Build the model for analyzing spiritual metrics."""
        input_layer = tf.keras.layers.Input(shape=(self.metric_dimensions,))
        
        # Metric analysis layers
        x = tf.keras.layers.Dense(self.metric_dimensions * 2, activation='relu')(input_layer)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Dense(self.metric_dimensions, activation='relu')(x)
        
        # Output layers for different metric aspects
        agape_output = tf.keras.layers.Dense(1, activation='sigmoid', name='agape_analysis')(x)
        kenosis_output = tf.keras.layers.Dense(1, activation='sigmoid', name='kenosis_analysis')(x)
        koinonia_output = tf.keras.layers.Dense(1, activation='sigmoid', name='koinonia_analysis')(x)
        
        self.metric_analysis_model = tf.keras.Model(
            inputs=input_layer,
            outputs=[agape_output, kenosis_output, koinonia_output]
        )
        
        logging.info("Metric analysis model built successfully")
    
    def _build_spiritual_insight_model(self) -> None:
        """Build the model for generating spiritual insights."""
        input_layer = tf.keras.layers.Input(shape=(self.metric_dimensions,))
        
        # Spiritual insight layers
        x = tf.keras.layers.Dense(self.metric_dimensions * 2, activation='relu')(input_layer)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Dense(self.metric_dimensions, activation='relu')(x)
        
        # Output layer for spiritual insights
        insight_output = tf.keras.layers.Dense(self.insight_depth, activation='sigmoid')(x)
        
        self.spiritual_insight_model = tf.keras.Model(
            inputs=input_layer,
            outputs=insight_output
        )
        
        logging.info("Spiritual insight model built successfully")
    
    def analyze_metrics(self, metrics: Dict) -> Dict:
        """
        Analyze spiritual metrics and generate insights.
        
        Args:
            metrics: Dictionary containing spiritual metrics
            
        Returns:
            Dictionary containing analysis results and insights
        """
        # Validate input metrics
        self._validate_metrics(metrics)
        
        # Prepare metric vector
        metric_vector = np.array([
            metrics['agape_score'],
            metrics['kenosis_factor'],
            metrics['koinonia_coherence']
        ]).reshape(1, -1)
        
        # Analyze metrics
        agape_analysis, kenosis_analysis, koinonia_analysis = self.metric_analysis_model.predict(metric_vector)
        
        # Generate spiritual insights
        insights = self.spiritual_insight_model.predict(metric_vector)
        
        # Update state
        self.state['current_metrics'] = metrics
        self.state['temporal_metrics'].append(metrics)
        if len(self.state['temporal_metrics']) > self.temporal_depth:
            self.state['temporal_metrics'].pop(0)
        self.state['spiritual_insights'].append(insights)
        if len(self.state['spiritual_insights']) > self.temporal_depth:
            self.state['spiritual_insights'].pop(0)
        
        # Prepare results
        results = {
            'metric_analysis': {
                'agape_analysis': float(agape_analysis[0][0]),
                'kenosis_analysis': float(kenosis_analysis[0][0]),
                'koinonia_analysis': float(koinonia_analysis[0][0])
            },
            'spiritual_insights': insights.tolist(),
            'temporal_evolution': self._analyze_temporal_evolution(),
            'spiritual_significance': self._calculate_spiritual_significance()
        }
        
        self.state['analysis_results'] = results
        return results
    
    def _validate_metrics(self, metrics: Dict) -> None:
        """Validate input metrics."""
        required_metrics = ['agape_score', 'kenosis_factor', 'koinonia_coherence']
        for metric in required_metrics:
            if metric not in metrics:
                raise ValueError(f"Missing required metric: {metric}")
            if not 0 <= metrics[metric] <= 1:
                raise ValueError(f"Metric {metric} must be between 0 and 1")
    
    def _analyze_temporal_evolution(self) -> Dict:
        """Analyze the temporal evolution of metrics."""
        if len(self.state['temporal_metrics']) < 2:
            return {'stability': 1.0, 'trend': 'stable'}
        
        # Calculate metric differences
        differences = []
        for i in range(1, len(self.state['temporal_metrics'])):
            diff = np.abs(np.array([
                self.state['temporal_metrics'][i]['agape_score'] - self.state['temporal_metrics'][i-1]['agape_score'],
                self.state['temporal_metrics'][i]['kenosis_factor'] - self.state['temporal_metrics'][i-1]['kenosis_factor'],
                self.state['temporal_metrics'][i]['koinonia_coherence'] - self.state['temporal_metrics'][i-1]['koinonia_coherence']
            ]))
            differences.append(np.mean(diff))
        
        # Calculate stability and trend
        stability = 1.0 - np.mean(differences)
        trend = 'increasing' if np.mean(differences) > 0 else 'decreasing' if np.mean(differences) < 0 else 'stable'
        
        return {'stability': float(stability), 'trend': trend}
    
    def _calculate_spiritual_significance(self) -> Dict:
        """Calculate the spiritual significance of current metrics."""
        if self.state['current_metrics'] is None:
            return {'significance': 0.0, 'level': 'low'}
        
        # Calculate overall significance
        significance = np.mean([
            self.state['current_metrics']['agape_score'],
            self.state['current_metrics']['kenosis_factor'],
            self.state['current_metrics']['koinonia_coherence']
        ])
        
        # Determine significance level
        if significance >= self.spiritual_threshold:
            level = 'high'
        elif significance >= self.spiritual_threshold * 0.7:
            level = 'medium'
        else:
            level = 'low'
        
        return {'significance': float(significance), 'level': level}
    
    def get_state(self) -> Dict:
        """Get the current state of the analyzer."""
        return self.state
    
    def reset(self) -> None:
        """Reset the analyzer state."""
        self.state = {
            'current_metrics': None,
            'temporal_metrics': [],
            'spiritual_insights': [],
            'analysis_results': None
        }
        logging.info("SpiritualMetricsAnalyzer state reset") 
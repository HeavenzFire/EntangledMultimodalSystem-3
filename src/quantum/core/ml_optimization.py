from typing import Dict, Any, List, Tuple
import numpy as np
from datetime import datetime
import logging
from dataclasses import dataclass

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class ModelMetrics:
    accuracy: float
    loss: float
    inference_time: float
    training_iterations: int

class MLModelOptimizer:
    def __init__(self):
        self.model_weights = {}
        self.training_history = []
        self.inference_history = []
        self.error_predictions = []
        self.trend_window = 10
        self.learning_rate = 0.01
        logger.info("Initialized MLModelOptimizer")

    def predict_errors(self, current_metrics: Dict[str, float]) -> float:
        """
        Enhanced error prediction using historical data and current metrics
        
        Args:
            current_metrics: Current system metrics
            
        Returns:
            Predicted error rate
        """
        if len(self.error_predictions) >= self.trend_window:
            # Calculate moving average with exponential weighting
            weights = np.exp(np.linspace(0, 1, self.trend_window))
            weights /= weights.sum()
            recent_errors = [p['actual'] for p in self.error_predictions[-self.trend_window:]]
            predicted = np.average(recent_errors, weights=weights)
            
            # Adjust prediction based on current metrics
            if current_metrics.get('ml_accuracy', 0) < 0.95:
                predicted *= 1.15  # Increase prediction if ML accuracy is low
            return predicted
        return current_metrics.get('error_rate', 0.01)

    def optimize_model(self, training_data: List[Dict[str, float]]) -> ModelMetrics:
        """
        Optimize ML model with adaptive learning
        
        Args:
            training_data: List of training examples
            
        Returns:
            ModelMetrics object with optimization results
        """
        start_time = datetime.now()
        
        # Adaptive learning rate adjustment
        if len(self.training_history) > 0:
            last_accuracy = self.training_history[-1].accuracy
            if last_accuracy < 0.95:
                self.learning_rate *= 1.1  # Increase learning rate if accuracy is low
            else:
                self.learning_rate *= 0.9  # Decrease learning rate if accuracy is good
        
        # Simulate model training
        accuracy = min(0.99, np.random.normal(0.95, 0.02))
        loss = 1 - accuracy
        inference_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Update model weights
        self.model_weights = {
            'accuracy': accuracy,
            'learning_rate': self.learning_rate,
            'last_updated': datetime.now()
        }
        
        # Track training progress
        metrics = ModelMetrics(
            accuracy=accuracy,
            loss=loss,
            inference_time=inference_time,
            training_iterations=len(self.training_history) + 1
        )
        self.training_history.append(metrics)
        
        logger.info(f"Model optimized: accuracy={accuracy:.4f}, loss={loss:.4f}")
        return metrics

    def detect_anomalies(self) -> bool:
        """
        Detect anomalies in system behavior
        
        Returns:
            True if anomaly detected, False otherwise
        """
        if len(self.error_predictions) < self.trend_window:
            return False
            
        # Calculate error rate trend
        recent_errors = [p['actual'] for p in self.error_predictions[-self.trend_window:]]
        avg_error = np.mean(recent_errors)
        std_error = np.std(recent_errors)
        
        # Detect significant deviation
        current_error = self.error_predictions[-1]['actual']
        threshold = avg_error + 2 * std_error
        
        if current_error > threshold:
            logger.warning(f"Anomaly detected: error rate {current_error:.4f} > threshold {threshold:.4f}")
            return True
        return False

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive ML model performance metrics
        
        Returns:
            Dictionary of performance metrics
        """
        if not self.training_history:
            return {
                'accuracy': 0.0,
                'loss': 1.0,
                'inference_time': 0.0,
                'training_iterations': 0
            }
            
        latest = self.training_history[-1]
        return {
            'accuracy': latest.accuracy,
            'loss': latest.loss,
            'inference_time': latest.inference_time,
            'training_iterations': latest.training_iterations,
            'learning_rate': self.learning_rate,
            'model_weights': self.model_weights
        }

    def track_prediction(self, job_id: str, predicted: float, actual: float) -> None:
        """
        Track prediction accuracy for a job
        
        Args:
            job_id: Job identifier
            predicted: Predicted error rate
            actual: Actual error rate
        """
        self.error_predictions.append({
            'timestamp': datetime.now(),
            'job_id': job_id,
            'predicted': predicted,
            'actual': actual
        })
        logger.debug(f"Tracked prediction for {job_id}: predicted={predicted:.4f}, actual={actual:.4f}") 
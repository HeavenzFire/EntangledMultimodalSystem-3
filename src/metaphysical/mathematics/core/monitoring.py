from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import logging
import time
from threading import Thread, Event
import psutil
import torch
from scipy.stats import entropy

from .safeguard_orchestrator import SafeguardOrchestrator
from ..config.safeguard_config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """System performance metrics"""
    cpu_usage: float
    memory_usage: float
    gpu_usage: Optional[float]
    quantum_coherence: float
    neural_network_accuracy: float
    response_time: float
    error_rate: float
    timestamp: datetime

@dataclass
class HealthState:
    """System health state"""
    overall_health: float
    component_health: Dict[str, float]
    critical_alerts: List[str]
    performance_metrics: SystemMetrics
    last_update: datetime

class SystemMonitor:
    """Monitors system performance and health"""
    
    def __init__(self, orchestrator: SafeguardOrchestrator):
        self.orchestrator = orchestrator
        self.stop_event = Event()
        self.monitor_thread = None
        
        self.state = HealthState(
            overall_health=1.0,
            component_health={},
            critical_alerts=[],
            performance_metrics=SystemMetrics(
                cpu_usage=0.0,
                memory_usage=0.0,
                gpu_usage=None,
                quantum_coherence=1.0,
                neural_network_accuracy=1.0,
                response_time=0.0,
                error_rate=0.0,
                timestamp=datetime.now()
            ),
            last_update=datetime.now()
        )
        
    def start_monitoring(self) -> None:
        """Start the monitoring thread"""
        self.monitor_thread = Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self) -> None:
        """Stop the monitoring thread"""
        self.stop_event.set()
        if self.monitor_thread:
            self.monitor_thread.join()
            
    def _monitor_loop(self) -> None:
        """Main monitoring loop"""
        while not self.stop_event.is_set():
            try:
                # Collect system metrics
                metrics = self._collect_metrics()
                
                # Update health state
                self._update_health_state(metrics)
                
                # Check for critical conditions
                self._check_critical_conditions()
                
                # Log status
                self._log_status()
                
                # Sleep for monitoring interval
                time.sleep(DEFAULT_CONFIG['orchestrator'].update_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(1)  # Prevent tight loop on error
                
    def _collect_metrics(self) -> SystemMetrics:
        """Collect system performance metrics"""
        # CPU and memory usage
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        
        # GPU usage if available
        gpu_usage = None
        if torch.cuda.is_available():
            gpu_usage = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            
        # Quantum coherence
        quantum_coherence = self.orchestrator.quantum_security.get_security_report()['coherence_level']
        
        # Neural network accuracy
        neural_network_accuracy = self._measure_neural_network_accuracy()
        
        # Response time
        start_time = time.time()
        self.orchestrator.orchestrate_safeguards(np.random.randn(64))
        response_time = time.time() - start_time
        
        # Error rate
        error_rate = self._calculate_error_rate()
        
        return SystemMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            gpu_usage=gpu_usage,
            quantum_coherence=quantum_coherence,
            neural_network_accuracy=neural_network_accuracy,
            response_time=response_time,
            error_rate=error_rate,
            timestamp=datetime.now()
        )
        
    def _measure_neural_network_accuracy(self) -> float:
        """Measure neural network accuracy"""
        try:
            # Generate test data
            test_data = np.random.randn(100, 64)
            test_labels = np.random.randn(100, 64)
            
            # Get predictions
            with torch.no_grad():
                predictions = self.orchestrator.coordination_network(
                    torch.tensor(test_data, dtype=torch.float32)
                ).numpy()
                
            # Calculate accuracy
            accuracy = 1.0 - np.mean(np.abs(predictions - test_labels))
            return float(accuracy)
            
        except Exception as e:
            logger.error(f"Error measuring neural network accuracy: {str(e)}")
            return 0.0
            
    def _calculate_error_rate(self) -> float:
        """Calculate system error rate"""
        try:
            # Get error rates from all components
            error_rates = [
                self.orchestrator.quantum_security.get_security_report()['error_rate'],
                1.0 - self.orchestrator.future_protection.get_protection_report()['stability'],
                1.0 - self.orchestrator.integration_safeguard.get_safeguard_report()['coherence'],
                1.0 - self.orchestrator.conflict_resolution.get_resolution_report()['harmony_score'],
                1.0 - self.orchestrator.divine_balance.get_balance_report()['harmony_level']
            ]
            
            return float(np.mean(error_rates))
            
        except Exception as e:
            logger.error(f"Error calculating error rate: {str(e)}")
            return 1.0
            
    def _update_health_state(self, metrics: SystemMetrics) -> None:
        """Update system health state"""
        # Calculate component health
        component_health = {
            'quantum_security': metrics.quantum_coherence,
            'future_protection': 1.0 - metrics.error_rate,
            'integration_safeguard': 1.0 - metrics.error_rate,
            'conflict_resolution': 1.0 - metrics.error_rate,
            'divine_balance': 1.0 - metrics.error_rate,
            'archetypal_network': metrics.neural_network_accuracy
        }
        
        # Calculate overall health
        overall_health = np.mean(list(component_health.values()))
        
        # Update state
        self.state.overall_health = overall_health
        self.state.component_health = component_health
        self.state.performance_metrics = metrics
        self.state.last_update = datetime.now()
        
    def _check_critical_conditions(self) -> None:
        """Check for critical system conditions"""
        critical_alerts = []
        
        # Check CPU usage
        if self.state.performance_metrics.cpu_usage > 90:
            critical_alerts.append("High CPU usage")
            
        # Check memory usage
        if self.state.performance_metrics.memory_usage > 90:
            critical_alerts.append("High memory usage")
            
        # Check GPU usage
        if self.state.performance_metrics.gpu_usage is not None and self.state.performance_metrics.gpu_usage > 90:
            critical_alerts.append("High GPU usage")
            
        # Check quantum coherence
        if self.state.performance_metrics.quantum_coherence < 0.5:
            critical_alerts.append("Low quantum coherence")
            
        # Check neural network accuracy
        if self.state.performance_metrics.neural_network_accuracy < 0.7:
            critical_alerts.append("Low neural network accuracy")
            
        # Check response time
        if self.state.performance_metrics.response_time > 1.0:
            critical_alerts.append("High response time")
            
        # Check error rate
        if self.state.performance_metrics.error_rate > 0.3:
            critical_alerts.append("High error rate")
            
        self.state.critical_alerts = critical_alerts
        
    def _log_status(self) -> None:
        """Log system status"""
        logger.info(f"System Health: {self.state.overall_health:.2f}")
        logger.info(f"CPU Usage: {self.state.performance_metrics.cpu_usage:.1f}%")
        logger.info(f"Memory Usage: {self.state.performance_metrics.memory_usage:.1f}%")
        if self.state.performance_metrics.gpu_usage is not None:
            logger.info(f"GPU Usage: {self.state.performance_metrics.gpu_usage:.1f}%")
        logger.info(f"Quantum Coherence: {self.state.performance_metrics.quantum_coherence:.2f}")
        logger.info(f"Neural Network Accuracy: {self.state.performance_metrics.neural_network_accuracy:.2f}")
        logger.info(f"Response Time: {self.state.performance_metrics.response_time:.3f}s")
        logger.info(f"Error Rate: {self.state.performance_metrics.error_rate:.2f}")
        
        if self.state.critical_alerts:
            logger.warning("Critical Alerts:")
            for alert in self.state.critical_alerts:
                logger.warning(f"- {alert}")
                
    def get_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report"""
        return {
            'timestamp': datetime.now(),
            'overall_health': self.state.overall_health,
            'component_health': self.state.component_health,
            'critical_alerts': self.state.critical_alerts,
            'performance_metrics': {
                'cpu_usage': self.state.performance_metrics.cpu_usage,
                'memory_usage': self.state.performance_metrics.memory_usage,
                'gpu_usage': self.state.performance_metrics.gpu_usage,
                'quantum_coherence': self.state.performance_metrics.quantum_coherence,
                'neural_network_accuracy': self.state.performance_metrics.neural_network_accuracy,
                'response_time': self.state.performance_metrics.response_time,
                'error_rate': self.state.performance_metrics.error_rate
            },
            'last_update': self.state.last_update,
            'system_status': 'healthy' if self.state.overall_health > 0.8 else 'warning'
        } 
import os
import time
import logging
from typing import Dict, Any, Optional, List
from src.core.system_manager import SystemManager
from src.core.agent_assistant import AgentAssistant
from src.utils.errors import ModelError
from dotenv import load_dotenv

class SystemMonitor:
    """System Monitor for comprehensive system monitoring and diagnostics."""
    
    def __init__(
        self,
        system_manager: Optional[SystemManager] = None,
        agent_assistant: Optional[AgentAssistant] = None
    ):
        """Initialize System Monitor.
        
        Args:
            system_manager: Optional SystemManager instance
            agent_assistant: Optional AgentAssistant instance
        """
        try:
            # Load environment variables
            load_dotenv()
            
            # Initialize core components
            self.system_manager = system_manager or SystemManager()
            self.agent_assistant = agent_assistant or AgentAssistant(self.system_manager)
            
            # Initialize parameters
            self.monitoring_interval = float(os.getenv("MONITORING_INTERVAL", "0.1"))
            self.history_length = int(os.getenv("MONITORING_HISTORY_LENGTH", "1000"))
            
            # Initialize thresholds
            self.monitoring_thresholds = {
                "cpu_usage": float(os.getenv("CPU_USAGE_THRESHOLD", "0.8")),
                "memory_usage": float(os.getenv("MEMORY_USAGE_THRESHOLD", "0.8")),
                "energy_consumption": float(os.getenv("ENERGY_CONSUMPTION_THRESHOLD", "0.8")),
                "network_latency": float(os.getenv("NETWORK_LATENCY_THRESHOLD", "0.8")),
                "response_time": float(os.getenv("RESPONSE_TIME_THRESHOLD", "0.5"))
            }
            
            # Initialize state
            self.state = {
                "status": "active",
                "last_monitoring": None,
                "monitoring_count": 0,
                "alert_count": 0,
                "diagnostic_count": 0
            }
            
            # Initialize metrics
            self.metrics = {
                "cpu_metrics": {
                    "usage": 0.0,
                    "temperature": 0.0,
                    "frequency": 0.0
                },
                "memory_metrics": {
                    "usage": 0.0,
                    "fragmentation": 0.0,
                    "swap_usage": 0.0
                },
                "energy_metrics": {
                    "consumption": 0.0,
                    "efficiency": 0.0,
                    "temperature": 0.0
                },
                "network_metrics": {
                    "bandwidth": 0.0,
                    "latency": 0.0,
                    "packet_loss": 0.0
                },
                "performance_metrics": {
                    "response_time": 0.0,
                    "throughput": 0.0,
                    "error_rate": 0.0
                }
            }
            
            # Initialize diagnostic history
            self.diagnostic_history = []
            
            logging.info("SystemMonitor initialized")
            
        except Exception as e:
            logging.error(f"Error initializing SystemMonitor: {str(e)}")
            raise ModelError(f"Failed to initialize SystemMonitor: {str(e)}")

    def monitor_system(self) -> Dict[str, Any]:
        """Monitor the entire system.
        
        Returns:
            Dict containing monitoring results
        """
        try:
            start_time = time.time()
            
            # Get system state
            system_state = self.system_manager.get_state()
            
            # Monitor each component
            cpu_metrics = self._monitor_cpu(system_state)
            memory_metrics = self._monitor_memory(system_state)
            energy_metrics = self._monitor_energy(system_state)
            network_metrics = self._monitor_network(system_state)
            performance_metrics = self._monitor_performance(system_state)
            
            # Update metrics
            self.metrics.update({
                "cpu_metrics": cpu_metrics,
                "memory_metrics": memory_metrics,
                "energy_metrics": energy_metrics,
                "network_metrics": network_metrics,
                "performance_metrics": performance_metrics
            })
            
            # Generate alerts
            alerts = self._generate_alerts()
            
            # Update state
            self.state["last_monitoring"] = time.time()
            self.state["monitoring_count"] += 1
            
            # Add to diagnostic history
            self._update_diagnostic_history({
                "timestamp": time.time(),
                "metrics": self.metrics,
                "alerts": alerts
            })
            
            return {
                "metrics": self.metrics,
                "alerts": alerts,
                "response_time": time.time() - start_time
            }
            
        except Exception as e:
            self.state["alert_count"] += 1
            logging.error(f"Error in system monitoring: {str(e)}")
            raise ModelError(f"System monitoring failed: {str(e)}")

    def _monitor_cpu(self, system_state: Dict[str, Any]) -> Dict[str, float]:
        """Monitor CPU metrics.
        
        Args:
            system_state: Current system state
            
        Returns:
            CPU metrics
        """
        try:
            cpu_state = system_state.get("cpu_state", {})
            
            return {
                "usage": cpu_state.get("usage", 0.0),
                "temperature": cpu_state.get("temperature", 0.0),
                "frequency": cpu_state.get("frequency", 0.0)
            }
            
        except Exception as e:
            logging.error(f"Error monitoring CPU: {str(e)}")
            return {"error": str(e)}

    def _monitor_memory(self, system_state: Dict[str, Any]) -> Dict[str, float]:
        """Monitor memory metrics.
        
        Args:
            system_state: Current system state
            
        Returns:
            Memory metrics
        """
        try:
            memory_state = system_state.get("memory_state", {})
            
            return {
                "usage": memory_state.get("usage", 0.0),
                "fragmentation": memory_state.get("fragmentation", 0.0),
                "swap_usage": memory_state.get("swap_usage", 0.0)
            }
            
        except Exception as e:
            logging.error(f"Error monitoring memory: {str(e)}")
            return {"error": str(e)}

    def _monitor_energy(self, system_state: Dict[str, Any]) -> Dict[str, float]:
        """Monitor energy metrics.
        
        Args:
            system_state: Current system state
            
        Returns:
            Energy metrics
        """
        try:
            energy_state = system_state.get("energy_state", {})
            
            return {
                "consumption": energy_state.get("consumption", 0.0),
                "efficiency": energy_state.get("efficiency", 0.0),
                "temperature": energy_state.get("temperature", 0.0)
            }
            
        except Exception as e:
            logging.error(f"Error monitoring energy: {str(e)}")
            return {"error": str(e)}

    def _monitor_network(self, system_state: Dict[str, Any]) -> Dict[str, float]:
        """Monitor network metrics.
        
        Args:
            system_state: Current system state
            
        Returns:
            Network metrics
        """
        try:
            network_state = system_state.get("network_state", {})
            
            return {
                "bandwidth": network_state.get("bandwidth", 0.0),
                "latency": network_state.get("latency", 0.0),
                "packet_loss": network_state.get("packet_loss", 0.0)
            }
            
        except Exception as e:
            logging.error(f"Error monitoring network: {str(e)}")
            return {"error": str(e)}

    def _monitor_performance(self, system_state: Dict[str, Any]) -> Dict[str, float]:
        """Monitor performance metrics.
        
        Args:
            system_state: Current system state
            
        Returns:
            Performance metrics
        """
        try:
            performance_state = system_state.get("performance_state", {})
            
            return {
                "response_time": performance_state.get("response_time", 0.0),
                "throughput": performance_state.get("throughput", 0.0),
                "error_rate": performance_state.get("error_rate", 0.0)
            }
            
        except Exception as e:
            logging.error(f"Error monitoring performance: {str(e)}")
            return {"error": str(e)}

    def _generate_alerts(self) -> List[Dict[str, Any]]:
        """Generate alerts based on monitoring thresholds.
        
        Returns:
            List of alerts
        """
        try:
            alerts = []
            
            # Check CPU metrics
            if self.metrics["cpu_metrics"]["usage"] > self.monitoring_thresholds["cpu_usage"]:
                alerts.append({
                    "type": "cpu",
                    "severity": "high",
                    "message": "CPU usage exceeds threshold"
                })
            
            # Check memory metrics
            if self.metrics["memory_metrics"]["usage"] > self.monitoring_thresholds["memory_usage"]:
                alerts.append({
                    "type": "memory",
                    "severity": "high",
                    "message": "Memory usage exceeds threshold"
                })
            
            # Check energy metrics
            if self.metrics["energy_metrics"]["consumption"] > self.monitoring_thresholds["energy_consumption"]:
                alerts.append({
                    "type": "energy",
                    "severity": "high",
                    "message": "Energy consumption exceeds threshold"
                })
            
            # Check network metrics
            if self.metrics["network_metrics"]["latency"] > self.monitoring_thresholds["network_latency"]:
                alerts.append({
                    "type": "network",
                    "severity": "high",
                    "message": "Network latency exceeds threshold"
                })
            
            # Check performance metrics
            if self.metrics["performance_metrics"]["response_time"] > self.monitoring_thresholds["response_time"]:
                alerts.append({
                    "type": "performance",
                    "severity": "high",
                    "message": "Response time exceeds threshold"
                })
            
            if alerts:
                self.state["alert_count"] += len(alerts)
            
            return alerts
            
        except Exception as e:
            logging.error(f"Error generating alerts: {str(e)}")
            return []

    def _update_diagnostic_history(self, diagnostic_data: Dict[str, Any]) -> None:
        """Update diagnostic history.
        
        Args:
            diagnostic_data: Diagnostic data to add
        """
        try:
            self.diagnostic_history.append(diagnostic_data)
            
            # Maintain history length
            if len(self.diagnostic_history) > self.history_length:
                self.diagnostic_history = self.diagnostic_history[-self.history_length:]
            
            self.state["diagnostic_count"] = len(self.diagnostic_history)
            
        except Exception as e:
            logging.error(f"Error updating diagnostic history: {str(e)}")

    def get_state(self) -> Dict[str, Any]:
        """Get current monitor state."""
        return self.state

    def get_metrics(self) -> Dict[str, Any]:
        """Get current monitoring metrics."""
        return self.metrics

    def get_diagnostic_history(self) -> List[Dict[str, Any]]:
        """Get diagnostic history."""
        return self.diagnostic_history

    def reset(self) -> None:
        """Reset monitor state."""
        self.state.update({
            "status": "active",
            "last_monitoring": None,
            "monitoring_count": 0,
            "alert_count": 0,
            "diagnostic_count": 0
        })
        
        self.metrics.update({
            "cpu_metrics": {
                "usage": 0.0,
                "temperature": 0.0,
                "frequency": 0.0
            },
            "memory_metrics": {
                "usage": 0.0,
                "fragmentation": 0.0,
                "swap_usage": 0.0
            },
            "energy_metrics": {
                "consumption": 0.0,
                "efficiency": 0.0,
                "temperature": 0.0
            },
            "network_metrics": {
                "bandwidth": 0.0,
                "latency": 0.0,
                "packet_loss": 0.0
            },
            "performance_metrics": {
                "response_time": 0.0,
                "throughput": 0.0,
                "error_rate": 0.0
            }
        })
        
        self.diagnostic_history = [] 
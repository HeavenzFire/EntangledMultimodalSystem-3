from typing import List, Dict, Optional, Any, Callable
import numpy as np
from .quantum_threading import QuantumThreadingBridge
from .error_correction import QuantumErrorCorrection
import logging
from datetime import datetime

class QuantumAPI:
    """High-level API for quantum threading operations."""
    
    def __init__(self, name: str = "default", config: Optional[Dict] = None):
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"QuantumAPI.{name}")
        self.bridge = QuantumThreadingBridge(
            num_threads=self.config.get("num_threads", 8),
            thread_capacity=self.config.get("thread_capacity", 4)
        )
        self.threads = {}
        self.pipelines = {}
        
    def create_thread(self, name: str, capacity: int = 4) -> str:
        """Create a new quantum thread with a friendly name."""
        thread = self.bridge.create_thread(name)
        self.threads[name] = thread
        self.logger.info(f"Created quantum thread: {name}")
        return name
        
    def apply_quantum_transform(self, thread_name: str, transform: str, **kwargs) -> None:
        """Apply a quantum transform to a thread."""
        if thread_name not in self.threads:
            raise ValueError(f"Thread {thread_name} not found")
            
        thread = self.threads[thread_name]
        
        # Define common transforms
        transforms = {
            "superposition": lambda: np.array([[1, 1], [1, -1]]) / np.sqrt(2),
            "phase_shift": lambda angle: np.array([[1, 0], [0, np.exp(1j * angle)]]),
            "swap": lambda: np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]),
            "controlled_not": lambda: np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
        }
        
        if transform not in transforms:
            raise ValueError(f"Unknown transform: {transform}")
            
        gate = transforms[transform](**kwargs)
        thread.apply_gate(gate)
        self.logger.info(f"Applied {transform} to thread {thread_name}")
        
    def create_quantum_pipeline(self, name: str, operations: List[Dict]) -> str:
        """Create a reusable quantum operation pipeline."""
        self.pipelines[name] = operations
        self.logger.info(f"Created quantum pipeline: {name}")
        return name
        
    def execute_pipeline(self, thread_name: str, pipeline_name: str) -> Any:
        """Execute a quantum pipeline on a thread."""
        if thread_name not in self.threads:
            raise ValueError(f"Thread {thread_name} not found")
        if pipeline_name not in self.pipelines:
            raise ValueError(f"Pipeline {pipeline_name} not found")
            
        thread = self.threads[thread_name]
        operations = self.pipelines[pipeline_name]
        
        for op in operations:
            self.apply_quantum_transform(thread_name, op["transform"], **op.get("params", {}))
            
        return thread.get_state()
        
    def create_quantum_network(self, name: str, thread_names: List[str]) -> str:
        """Create a network of quantum threads."""
        for i in range(len(thread_names)-1):
            self.bridge.apply_coupling(thread_names[i], thread_names[i+1])
            
        self.logger.info(f"Created quantum network: {name}")
        return name
        
    def create_interference_pattern(self, thread_names: List[str]) -> None:
        """Create interference between multiple threads."""
        self.bridge.create_interference(thread_names)
        self.logger.info(f"Created interference pattern between {len(thread_names)} threads")
        
    def measure_thread(self, thread_name: str) -> int:
        """Measure a quantum thread's state."""
        if thread_name not in self.threads:
            raise ValueError(f"Thread {thread_name} not found")
            
        result = self.threads[thread_name].measure()
        self.logger.info(f"Measured thread {thread_name}: {result}")
        return result
        
    def get_thread_state(self, thread_name: str) -> Dict:
        """Get the current state of a quantum thread."""
        if thread_name not in self.threads:
            raise ValueError(f"Thread {thread_name} not found")
            
        thread = self.threads[thread_name]
        state = thread.get_state()
        error_info = thread.get_error_info()
        
        return {
            "thread_name": thread_name,
            "state": state.tolist(),
            "error_info": error_info,
            "timestamp": datetime.now().isoformat()
        }
        
    def get_network_status(self) -> Dict:
        """Get the status of all quantum threads and networks."""
        return {
            "threads": {
                name: self.get_thread_state(name)
                for name in self.threads
            },
            "pipelines": list(self.pipelines.keys()),
            "bridge_status": self.bridge.get_bridge_status()
        }
        
    def save_state(self, filename: str) -> None:
        """Save the current quantum state to a file."""
        state = {
            "threads": {
                name: {
                    "state": thread.get_state().tolist(),
                    "error_info": thread.get_error_info()
                }
                for name, thread in self.threads.items()
            },
            "pipelines": self.pipelines,
            "timestamp": datetime.now().isoformat()
        }
        
        import json
        with open(filename, 'w') as f:
            json.dump(state, f)
            
        self.logger.info(f"Saved quantum state to {filename}")
        
    def load_state(self, filename: str) -> None:
        """Load a quantum state from a file."""
        import json
        with open(filename, 'r') as f:
            state = json.load(f)
            
        # Recreate threads
        self.threads = {}
        for name, data in state["threads"].items():
            thread = self.bridge.create_thread(name)
            thread.state = np.array(data["state"], dtype=complex)
            self.threads[name] = thread
            
        # Restore pipelines
        self.pipelines = state["pipelines"]
        
        self.logger.info(f"Loaded quantum state from {filename}")
        
    def create_quantum_algorithm(self, name: str, algorithm: Callable) -> str:
        """Register a quantum algorithm for reuse."""
        self.pipelines[name] = algorithm
        self.logger.info(f"Registered quantum algorithm: {name}")
        return name
        
    def execute_algorithm(self, thread_name: str, algorithm_name: str, **kwargs) -> Any:
        """Execute a registered quantum algorithm on a thread."""
        if thread_name not in self.threads:
            raise ValueError(f"Thread {thread_name} not found")
        if algorithm_name not in self.pipelines:
            raise ValueError(f"Algorithm {algorithm_name} not found")
            
        thread = self.threads[thread_name]
        algorithm = self.pipelines[algorithm_name]
        
        return algorithm(thread, **kwargs) 
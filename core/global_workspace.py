"""
Global Workspace Core — orchestrates multimodal integration, attention, and information broadcast.
"""

from typing import Dict, Any, Optional, List
import numpy as np
import threading
import logging
from dataclasses import dataclass
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class InformationChunk:
    """Represents a chunk of information in the global workspace"""
    content: Any
    modality: str
    timestamp: float
    importance: float
    source: str
    metadata: Dict[str, Any] = None

class GlobalWorkspace:
    """Implements the global workspace architecture"""
    
    def __init__(self):
        self.information_chunks: List[InformationChunk] = []
        self.attention_weights: Optional[np.ndarray] = None
        self.integration_matrix: Optional[np.ndarray] = None
        self.attention_focus: Optional[str] = None
        self.lock = threading.Lock()
        self._initialize_workspace()
    
    def _initialize_workspace(self) -> None:
        """Initialize workspace state"""
        self.information_chunks = []
        self.attention_weights = np.zeros(0)
        self.integration_matrix = np.zeros((0, 0))
        self.attention_focus = None
    
    def integrate(self, chunk: InformationChunk) -> np.ndarray:
        """
        Integrate new information into the global workspace
        
        Args:
            chunk: Information chunk to integrate
            
        Returns:
            np.ndarray: Integrated state after processing
        """
        with self.lock:
            # Add new chunk
            self.information_chunks.append(chunk)
            
            # Update attention weights
            self._update_attention_weights()
            
            # Update integration matrix
            self._update_integration_matrix()
            
            # Update attention focus
            self._update_attention_focus()
            
            # Return integrated state
            return self._get_integrated_state()
    
    def _update_attention_weights(self) -> None:
        """Update attention weights based on chunk importance"""
        weights = np.array([chunk.importance for chunk in self.information_chunks])
        if len(weights) > 0:
            self.attention_weights = weights / np.sum(weights)
    
    def _update_integration_matrix(self) -> None:
        """Update integration matrix based on chunk relationships"""
        n = len(self.information_chunks)
        if n > 0:
            self.integration_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if i != j:
                        # Simple integration based on temporal proximity
                        time_diff = abs(self.information_chunks[i].timestamp - 
                                      self.information_chunks[j].timestamp)
                        self.integration_matrix[i, j] = np.exp(-time_diff)
    
    def _update_attention_focus(self) -> None:
        """Update attention focus based on current state"""
        if len(self.information_chunks) > 0:
            max_idx = np.argmax(self.attention_weights)
            self.attention_focus = self.information_chunks[max_idx].modality
    
    def _get_integrated_state(self) -> np.ndarray:
        """Get current integrated state"""
        if len(self.information_chunks) == 0:
            return np.zeros(0)
        
        # Combine chunks weighted by attention
        state = np.zeros_like(self.information_chunks[0].content)
        for chunk, weight in zip(self.information_chunks, self.attention_weights):
            state += weight * chunk.content
        return state
    
    def broadcast(self) -> Dict[str, Any]:
        """
        Broadcast current workspace state
        
        Returns:
            Dict[str, Any]: Current workspace state
        """
        with self.lock:
            return {
                "focus": self.attention_focus,
                "chunks": self.information_chunks,
                "attention_weights": self.attention_weights,
                "integration_matrix": self.integration_matrix,
                "integrated_state": self._get_integrated_state()
            }
    
    def clear(self) -> None:
        """Clear the workspace"""
        with self.lock:
            self._initialize_workspace()
            logger.info("Workspace cleared")

# Example usage
if __name__ == "__main__":
    # Create workspace
    workspace = GlobalWorkspace()
    
    # Create some test chunks
    chunk1 = InformationChunk(
        content=np.array([1, 2, 3]),
        modality="vision",
        timestamp=datetime.now().timestamp(),
        importance=0.8,
        source="camera"
    )
    
    chunk2 = InformationChunk(
        content=np.array([4, 5, 6]),
        modality="audio",
        timestamp=datetime.now().timestamp(),
        importance=0.6,
        source="microphone"
    )
    
    # Integrate chunks
    workspace.integrate(chunk1)
    workspace.integrate(chunk2)
    
    # Broadcast state
    state = workspace.broadcast()
    print("Workspace state:", state)
    
    # Clear workspace
    workspace.clear()
    print("After clear:", workspace.broadcast()) 
import numpy as np
import tensorflow as tf
from typing import Dict, Any, Tuple, Optional
from src.utils.logger import logger
from src.utils.errors import ModelError

class FractalIntelligenceEngine:
    """Advanced fractal intelligence engine for generating and analyzing fractals with embedded semantic meaning."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the Fractal Intelligence Engine.
        
        Args:
            config: Configuration dictionary with parameters
        """
        self.config = config or {}
        self.logger = logger
        
        # Initialize fractal parameters
        self.max_iterations = self.config.get("max_iterations", 100)
        self.resolution = self.config.get("resolution", 1024)
        self.escape_radius = self.config.get("escape_radius", 2.0)
        
        # Initialize semantic parameters
        self.semantic_dimensions = self.config.get("semantic_dimensions", 128)
        self.embedding_depth = self.config.get("embedding_depth", 3)
        
        # Initialize neural network
        self.semantic_network = self._build_semantic_network()
        
        # Initialize state
        self.state = {
            "fractal_data": None,
            "semantic_embedding": None,
            "analysis_results": None,
            "generation_metrics": None
        }
        
        self.metrics = {
            "fractal_complexity": 0.0,
            "semantic_coherence": 0.0,
            "pattern_richness": 0.0,
            "generation_quality": 0.0
        }
    
    def _build_semantic_network(self) -> tf.keras.Model:
        """Build semantic embedding network."""
        # Input layer for fractal data
        fractal_input = tf.keras.layers.Input(shape=(self.resolution, self.resolution, 3))
        
        # Convolutional layers for feature extraction
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(fractal_input)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        
        # Semantic embedding layers
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        
        # Output layer for semantic embedding
        semantic_output = tf.keras.layers.Dense(
            self.semantic_dimensions,
            activation='tanh'
        )(x)
        
        return tf.keras.Model(inputs=fractal_input, outputs=semantic_output)
    
    def generate_mandelbrot(self, center: Tuple[float, float], 
                          zoom: float = 1.0) -> np.ndarray:
        """Generate Mandelbrot set fractal.
        
        Args:
            center: Center point of the fractal (real, imaginary)
            zoom: Zoom level of the fractal
            
        Returns:
            Generated fractal as numpy array
        """
        try:
            # Create coordinate grid
            x = np.linspace(center[0] - 2/zoom, center[0] + 2/zoom, self.resolution)
            y = np.linspace(center[1] - 2/zoom, center[1] + 2/zoom, self.resolution)
            X, Y = np.meshgrid(x, y)
            Z = X + 1j * Y
            
            # Initialize fractal array
            fractal = np.zeros((self.resolution, self.resolution))
            c = Z
            
            # Generate fractal
            for i in range(self.max_iterations):
                Z = Z**2 + c
                mask = np.abs(Z) < self.escape_radius
                fractal[mask] = i
            
            # Normalize and colorize
            fractal = fractal / self.max_iterations
            fractal = np.stack([fractal, fractal, fractal], axis=-1)
            
            return fractal
            
        except Exception as e:
            self.logger.error(f"Error generating Mandelbrot set: {str(e)}")
            raise ModelError(f"Fractal generation failed: {str(e)}")
    
    def generate_julia(self, c: complex, 
                      center: Tuple[float, float] = (0, 0),
                      zoom: float = 1.0) -> np.ndarray:
        """Generate Julia set fractal.
        
        Args:
            c: Complex parameter for Julia set
            center: Center point of the fractal (real, imaginary)
            zoom: Zoom level of the fractal
            
        Returns:
            Generated fractal as numpy array
        """
        try:
            # Create coordinate grid
            x = np.linspace(center[0] - 2/zoom, center[0] + 2/zoom, self.resolution)
            y = np.linspace(center[1] - 2/zoom, center[1] + 2/zoom, self.resolution)
            X, Y = np.meshgrid(x, y)
            Z = X + 1j * Y
            
            # Initialize fractal array
            fractal = np.zeros((self.resolution, self.resolution))
            
            # Generate fractal
            for i in range(self.max_iterations):
                Z = Z**2 + c
                mask = np.abs(Z) < self.escape_radius
                fractal[mask] = i
            
            # Normalize and colorize
            fractal = fractal / self.max_iterations
            fractal = np.stack([fractal, fractal, fractal], axis=-1)
            
            return fractal
            
        except Exception as e:
            self.logger.error(f"Error generating Julia set: {str(e)}")
            raise ModelError(f"Fractal generation failed: {str(e)}")
    
    def embed_semantic_meaning(self, fractal: np.ndarray) -> Dict[str, Any]:
        """Embed semantic meaning into fractal.
        
        Args:
            fractal: Input fractal data
            
        Returns:
            Dictionary containing semantic embedding and analysis
        """
        try:
            # Validate fractal data
            self._validate_fractal_data(fractal)
            
            # Process through semantic network
            semantic_embedding = self.semantic_network(
                np.expand_dims(fractal, axis=0)
            )
            
            # Update state and metrics
            self._update_state(fractal, semantic_embedding)
            self._update_metrics(fractal, semantic_embedding)
            
            return {
                "semantic_embedding": semantic_embedding.numpy()[0],
                "analysis": self.state["analysis_results"],
                "metrics": self.metrics,
                "state": self.state
            }
            
        except Exception as e:
            self.logger.error(f"Error embedding semantic meaning: {str(e)}")
            raise ModelError(f"Semantic embedding failed: {str(e)}")
    
    def _validate_fractal_data(self, fractal: np.ndarray) -> None:
        """Validate fractal data.
        
        Args:
            fractal: Input fractal data
        """
        if fractal.shape != (self.resolution, self.resolution, 3):
            raise ModelError("Invalid fractal data shape")
        
        if not np.all(0 <= fractal) or not np.all(fractal <= 1):
            raise ModelError("Fractal data values out of range [0, 1]")
    
    def _update_state(self, fractal: np.ndarray,
                     semantic_embedding: tf.Tensor) -> None:
        """Update engine state.
        
        Args:
            fractal: Input fractal data
            semantic_embedding: Semantic embedding
        """
        self.state["fractal_data"] = fractal
        self.state["semantic_embedding"] = semantic_embedding.numpy()[0]
        
        # Perform fractal analysis
        self.state["analysis_results"] = {
            "pattern_density": np.mean(fractal),
            "edge_complexity": self._calculate_edge_complexity(fractal),
            "symmetry_score": self._calculate_symmetry_score(fractal)
        }
        
        self.state["generation_metrics"] = {
            "resolution": self.resolution,
            "iterations": self.max_iterations,
            "zoom_level": self.config.get("zoom", 1.0)
        }
    
    def _update_metrics(self, fractal: np.ndarray,
                       semantic_embedding: tf.Tensor) -> None:
        """Update engine metrics.
        
        Args:
            fractal: Input fractal data
            semantic_embedding: Semantic embedding
        """
        # Calculate fractal complexity
        self.metrics["fractal_complexity"] = np.mean(
            np.abs(np.gradient(fractal))
        )
        
        # Calculate semantic coherence
        self.metrics["semantic_coherence"] = np.mean(
            np.abs(semantic_embedding)
        )
        
        # Calculate pattern richness
        self.metrics["pattern_richness"] = len(
            np.unique(fractal.reshape(-1, 3), axis=0)
        ) / (self.resolution * self.resolution)
        
        # Calculate generation quality
        self.metrics["generation_quality"] = (
            self.metrics["fractal_complexity"] *
            self.metrics["semantic_coherence"] *
            self.metrics["pattern_richness"]
        )
    
    def _calculate_edge_complexity(self, fractal: np.ndarray) -> float:
        """Calculate edge complexity of fractal.
        
        Args:
            fractal: Input fractal data
            
        Returns:
            Edge complexity score
        """
        # Calculate gradients
        dx = np.gradient(fractal, axis=0)
        dy = np.gradient(fractal, axis=1)
        
        # Calculate edge magnitude
        edge_magnitude = np.sqrt(dx**2 + dy**2)
        
        return np.mean(edge_magnitude)
    
    def _calculate_symmetry_score(self, fractal: np.ndarray) -> float:
        """Calculate symmetry score of fractal.
        
        Args:
            fractal: Input fractal data
            
        Returns:
            Symmetry score
        """
        # Check horizontal symmetry
        horizontal_symmetry = np.mean(
            np.abs(fractal - np.flip(fractal, axis=0))
        )
        
        # Check vertical symmetry
        vertical_symmetry = np.mean(
            np.abs(fractal - np.flip(fractal, axis=1))
        )
        
        # Check diagonal symmetry
        diagonal_symmetry = np.mean(
            np.abs(fractal - np.transpose(fractal, (1, 0, 2)))
        )
        
        return 1 - (horizontal_symmetry + vertical_symmetry + diagonal_symmetry) / 3
    
    def get_state(self) -> Dict[str, Any]:
        """Get current engine state."""
        return self.state
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current engine metrics."""
        return self.metrics
    
    def reset(self) -> None:
        """Reset engine state."""
        self.state = {
            "fractal_data": None,
            "semantic_embedding": None,
            "analysis_results": None,
            "generation_metrics": None
        }
        self.metrics = {
            "fractal_complexity": 0.0,
            "semantic_coherence": 0.0,
            "pattern_richness": 0.0,
            "generation_quality": 0.0
        } 
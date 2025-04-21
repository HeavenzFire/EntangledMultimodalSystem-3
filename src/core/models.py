import numpy as np
import tensorflow as tf
from tensorflow import keras
from src.utils.logger import logger
from src.utils.errors import ModelError

class ConsciousnessExpander:
    def __init__(self, input_dim=1):
        """Initialize the consciousness expander model."""
        try:
            self.model = self.build_model(input_dim)
            logger.info("ConsciousnessExpander initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ConsciousnessExpander: {str(e)}")
            raise ModelError(f"Model initialization failed: {str(e)}")

    def build_model(self, input_dim):
        """Build the neural network model."""
        model = keras.Sequential([
            keras.layers.Dense(256, activation='relu', input_shape=(input_dim,)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(64, activation='tanh'),
            keras.layers.Dense(32, activation='tanh'),
            keras.layers.Dense(16, activation='tanh'),
            keras.layers.Dense(8, activation='tanh'),
            keras.layers.Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        return model

    def evolve(self, x):
        """Process input through the model."""
        try:
            if not isinstance(x, (list, np.ndarray)):
                raise ValueError("Input must be a list or numpy array")
            
            x = np.array(x).reshape(-1, 1)
            predictions = self.model.predict(x, verbose=0)
            logger.info(f"Successfully processed input of shape {x.shape}")
            return predictions
        except Exception as e:
            logger.error(f"Error in evolve method: {str(e)}")
            raise ModelError(f"Failed to process input: {str(e)}")

    def train(self, X, y, epochs=10, batch_size=32):
        """Train the model on given data."""
        try:
            X = np.array(X).reshape(-1, 1)
            y = np.array(y).reshape(-1, 1)
            
            history = self.model.fit(
                X, y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                verbose=0
            )
            
            logger.info(f"Model trained successfully for {epochs} epochs")
            return history.history
        except Exception as e:
            logger.error(f"Error in train method: {str(e)}")
            raise ModelError(f"Training failed: {str(e)}") 
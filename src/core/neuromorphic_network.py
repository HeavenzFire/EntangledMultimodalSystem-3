import numpy as np
import torch
import torch.nn as nn
from src.utils.logger import logger
from src.utils.errors import ModelError
import matplotlib.pyplot as plt

class SpikingNeuron(nn.Module):
    def __init__(self, threshold=1.0, decay=0.9):
        super(SpikingNeuron, self).__init__()
        self.threshold = threshold
        self.decay = decay
        self.membrane_potential = 0.0
        self.spike_history = []

    def forward(self, x):
        self.membrane_potential = self.decay * self.membrane_potential + x
        spike = (self.membrane_potential >= self.threshold).float()
        self.membrane_potential = self.membrane_potential * (1 - spike)
        self.spike_history.append(spike.item())
        return spike

class NeuromorphicNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        """Initialize the neuromorphic network."""
        try:
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.output_size = output_size
            
            # Initialize layers
            self.input_layer = [SpikingNeuron() for _ in range(input_size)]
            self.hidden_layer = [SpikingNeuron() for _ in range(hidden_size)]
            self.output_layer = [SpikingNeuron() for _ in range(output_size)]
            
            # Initialize weights
            self.weights_input_hidden = torch.randn(input_size, hidden_size) * 0.1
            self.weights_hidden_output = torch.randn(hidden_size, output_size) * 0.1
            
            # Initialize synaptic plasticity parameters
            self.learning_rate = 0.01
            self.stdp_window = 20  # Spike-timing-dependent plasticity window
            
            logger.info("NeuromorphicNetwork initialized")
        except Exception as e:
            logger.error(f"Failed to initialize NeuromorphicNetwork: {str(e)}")
            raise ModelError(f"Neuromorphic network initialization failed: {str(e)}")

    def forward(self, x):
        """Forward pass through the network."""
        try:
            # Input layer processing
            input_spikes = torch.zeros(self.input_size)
            for i, neuron in enumerate(self.input_layer):
                input_spikes[i] = neuron(x[i])
            
            # Hidden layer processing
            hidden_input = torch.matmul(input_spikes, self.weights_input_hidden)
            hidden_spikes = torch.zeros(self.hidden_size)
            for i, neuron in enumerate(self.hidden_layer):
                hidden_spikes[i] = neuron(hidden_input[i])
            
            # Output layer processing
            output_input = torch.matmul(hidden_spikes, self.weights_hidden_output)
            output_spikes = torch.zeros(self.output_size)
            for i, neuron in enumerate(self.output_layer):
                output_spikes[i] = neuron(output_input[i])
            
            return output_spikes
        except Exception as e:
            logger.error(f"Forward pass failed: {str(e)}")
            raise ModelError(f"Forward pass failed: {str(e)}")

    def update_weights(self, input_spikes, hidden_spikes, output_spikes):
        """Update weights using spike-timing-dependent plasticity."""
        try:
            # STDP for input-hidden weights
            for i in range(self.input_size):
                for j in range(self.hidden_size):
                    if input_spikes[i] and hidden_spikes[j]:
                        self.weights_input_hidden[i, j] += self.learning_rate
                    elif input_spikes[i] and not hidden_spikes[j]:
                        self.weights_input_hidden[i, j] -= self.learning_rate * 0.5
            
            # STDP for hidden-output weights
            for i in range(self.hidden_size):
                for j in range(self.output_size):
                    if hidden_spikes[i] and output_spikes[j]:
                        self.weights_hidden_output[i, j] += self.learning_rate
                    elif hidden_spikes[i] and not output_spikes[j]:
                        self.weights_hidden_output[i, j] -= self.learning_rate * 0.5
        except Exception as e:
            logger.error(f"Weight update failed: {str(e)}")
            raise ModelError(f"Weight update failed: {str(e)}")

    def train(self, x, y, epochs=100):
        """Train the network on input-output pairs."""
        try:
            losses = []
            for epoch in range(epochs):
                # Forward pass
                output = self.forward(x)
                
                # Calculate loss
                loss = torch.mean((output - y) ** 2)
                losses.append(loss.item())
                
                # Update weights
                self.update_weights(
                    torch.tensor([neuron.spike_history[-1] for neuron in self.input_layer]),
                    torch.tensor([neuron.spike_history[-1] for neuron in self.hidden_layer]),
                    torch.tensor([neuron.spike_history[-1] for neuron in self.output_layer])
                )
                
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}, Loss: {loss.item()}")
            
            return losses
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise ModelError(f"Training failed: {str(e)}")

    def visualize_activity(self, save_path=None):
        """Visualize neural activity patterns."""
        try:
            plt.figure(figsize=(12, 6))
            
            # Plot input layer activity
            plt.subplot(3, 1, 1)
            for i, neuron in enumerate(self.input_layer):
                plt.plot(neuron.spike_history, label=f'Input {i}')
            plt.title('Input Layer Activity')
            plt.legend()
            
            # Plot hidden layer activity
            plt.subplot(3, 1, 2)
            for i, neuron in enumerate(self.hidden_layer):
                plt.plot(neuron.spike_history, label=f'Hidden {i}')
            plt.title('Hidden Layer Activity')
            plt.legend()
            
            # Plot output layer activity
            plt.subplot(3, 1, 3)
            for i, neuron in enumerate(self.output_layer):
                plt.plot(neuron.spike_history, label=f'Output {i}')
            plt.title('Output Layer Activity')
            plt.legend()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
            else:
                plt.show()
            
            plt.close()
        except Exception as e:
            logger.error(f"Activity visualization failed: {str(e)}")
            raise ModelError(f"Activity visualization failed: {str(e)}")

    def get_network_state(self):
        """Get the current state of the network."""
        try:
            return {
                "input_layer": {
                    "membrane_potentials": [neuron.membrane_potential for neuron in self.input_layer],
                    "spike_counts": [len(neuron.spike_history) for neuron in self.input_layer]
                },
                "hidden_layer": {
                    "membrane_potentials": [neuron.membrane_potential for neuron in self.hidden_layer],
                    "spike_counts": [len(neuron.spike_history) for neuron in self.hidden_layer]
                },
                "output_layer": {
                    "membrane_potentials": [neuron.membrane_potential for neuron in self.output_layer],
                    "spike_counts": [len(neuron.spike_history) for neuron in self.output_layer]
                },
                "weights": {
                    "input_hidden": self.weights_input_hidden.tolist(),
                    "hidden_output": self.weights_hidden_output.tolist()
                }
            }
        except Exception as e:
            logger.error(f"Network state retrieval failed: {str(e)}")
            raise ModelError(f"Network state retrieval failed: {str(e)}") 
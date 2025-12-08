import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import numpy as np

class NeuralIntegrator(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Define neural network architecture
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.processor = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        # Encode input
        encoded = self.encoder(x)
        
        # Process features
        processed = self.processor(encoded)
        
        # Decode output
        decoded = self.decoder(processed)
        
        return decoded
    
    def process_multimodal_data(self, modalities: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Process multimodal data using neural network"""
        # Stack modalities along channel dimension
        stacked = torch.cat(list(modalities.values()), dim=1)
        
        # Process through network
        processed = self.forward(stacked)
        
        return processed
    
    def extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract features at different network layers"""
        features = {}
        
        # Get encoder features
        x = self.encoder[0](x)
        features['encoder_conv1'] = x
        x = self.encoder[1](x)
        x = self.encoder[2](x)
        features['encoder_conv2'] = x
        x = self.encoder[3](x)
        x = self.encoder[4](x)
        features['encoder_pool'] = x
        
        # Get processor features
        x = self.processor[0](x)
        features['processor_conv1'] = x
        x = self.processor[1](x)
        x = self.processor[2](x)
        features['processor_conv2'] = x
        x = self.processor[3](x)
        x = self.processor[4](x)
        features['processor_conv3'] = x
        
        return features
    
    def compute_loss(self, output: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute various loss metrics"""
        losses = {}
        
        # Reconstruction loss
        losses['reconstruction'] = F.mse_loss(output, target)
        
        # Feature loss (using intermediate features)
        output_features = self.extract_features(output)
        target_features = self.extract_features(target)
        
        for layer in output_features:
            losses[f'feature_{layer}'] = F.mse_loss(
                output_features[layer],
                target_features[layer]
            )
        
        return losses 
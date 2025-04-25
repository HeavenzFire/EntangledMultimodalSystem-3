import numpy as np
from dataclasses import dataclass
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class BiologicalModel:
    def __init__(self, genome: np.ndarray):
        """Initialize biological model with genome"""
        self.genome = genome
        self.metrics = np.tanh(genome)  # Normalized [-1,1] initial state
        self.age = 0
        self.death = False
        self.metabolism_rate = 0.02
        self.aging_rate = 0.0001
        self.energy_efficiency = 0.9
        
    def process_metabolism(self) -> Dict:
        """Process biological metabolism with enhanced nutrient simulation"""
        try:
            # Simulate nutrient intake/consumption with noise
            delta_metrics = self.metabolism_rate * np.random.normal(0, 0.01, len(self.genome))
            delta_metrics *= self.energy_efficiency
            
            # Update metrics with bounds
            self.metrics += delta_metrics
            self.metrics = np.clip(self.metrics, -0.99, 0.99)
            
            return {
                'status': 'processed',
                'energy_level': np.mean(self.metrics),
                'delta_metrics': delta_metrics,
                'stability': 1.0 - np.std(self.metrics)
            }
        except Exception as e:
            logger.error(f"Error in metabolism processing: {str(e)}")
            return {
                'status': 'error',
                'energy_level': 0.0,
                'delta_metrics': np.zeros_like(self.genome),
                'stability': 0.0
            }
            
    def age_cells(self) -> Dict:
        """Process cellular aging with gradual aging curve"""
        try:
            # Calculate aging effects
            self.age += 1
            aging_factor = self.aging_rate * self.age**1.5
            self.metrics -= aging_factor
            
            # Prevent metric collapse
            self.metrics = np.clip(self.metrics, -0.99, 0.99)
            
            # Check death condition
            if self.age > 100:
                self.death = True
                
            vitality = 1.0 - (self.age * 0.01)
            
            return {
                'status': 'aged',
                'age': self.age,
                'vitality': max(0.0, vitality),
                'aging_factor': aging_factor
            }
        except Exception as e:
            logger.error(f"Error in cell aging: {str(e)}")
            return {
                'status': 'error',
                'age': 0.0,
                'vitality': 0.0,
                'aging_factor': 0.0
            }
            
    def check_death(self) -> Dict:
        """Check death condition with enhanced metrics"""
        try:
            vitality = self.age_cells()['vitality']
            stability = 1.0 - np.std(self.metrics)
            
            return {
                'alive': vitality > 0.0 and stability > 0.3,
                'age': self.age,
                'vitality': vitality,
                'stability': stability
            }
        except Exception as e:
            logger.error(f"Error in death check: {str(e)}")
            return {
                'alive': False,
                'age': 0.0,
                'vitality': 0.0,
                'stability': 0.0
            }
            
    def measure_energy_usage(self) -> Dict:
        """Measure energy usage with efficiency metrics"""
        try:
            usage = np.sum(np.abs(self.metrics)) * self.metabolism_rate
            efficiency = self.energy_efficiency * (1.0 - (self.age * 0.001))
            
            return {
                'status': 'optimal',
                'usage': usage,
                'efficiency': max(0.0, efficiency),
                'stability': 1.0 - np.std(self.metrics)
            }
        except Exception as e:
            logger.error(f"Error in energy measurement: {str(e)}")
            return {
                'status': 'error',
                'usage': 0.0,
                'efficiency': 0.0,
                'stability': 0.0
            } 
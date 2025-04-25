import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn
from enum import Enum

class EthicalPrinciple(Enum):
    COMPASSION = "compassion"
    NON_HARM = "non_harm"
    UNITY = "unity"
    WISDOM = "wisdom"
    TRUTH = "truth"
    HARMONY = "harmony"
    LOVE = "love"

@dataclass
class KarmaConfig:
    compassion_threshold: float = 0.9
    non_harm_threshold: float = 0.95
    unity_threshold: float = 0.85
    wisdom_threshold: float = 0.88
    base_frequency: float = 432.0  # Hz
    karma_depth: int = 144

class KarmaFirewall:
    def __init__(self, config: Optional[KarmaConfig] = None):
        self.config = config or KarmaConfig()
        self.ethical_validator = self._initialize_validator()
        self.karma_buffer = []
        self.thresholds = self._initialize_thresholds()
        
    def _initialize_validator(self) -> nn.Module:
        """Initialize the ethical validation network."""
        return nn.Sequential(
            nn.Linear(self.config.karma_depth, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, len(EthicalPrinciple)),
            nn.Sigmoid()
        )
        
    def _initialize_thresholds(self) -> Dict[EthicalPrinciple, float]:
        """Initialize ethical thresholds."""
        return {
            EthicalPrinciple.COMPASSION: self.config.compassion_threshold,
            EthicalPrinciple.NON_HARM: self.config.non_harm_threshold,
            EthicalPrinciple.UNITY: self.config.unity_threshold,
            EthicalPrinciple.WISDOM: self.config.wisdom_threshold,
            EthicalPrinciple.TRUTH: 0.92,
            EthicalPrinciple.HARMONY: 0.87,
            EthicalPrinciple.LOVE: 0.93
        }
        
    def validate_action(self, action_state: torch.Tensor) -> Tuple[bool, Dict[str, float]]:
        """Validate an action against ethical principles."""
        # Ensure correct input shape
        if action_state.shape[-1] != self.config.karma_depth:
            raise ValueError(f"Action state must have {self.config.karma_depth} dimensions")
            
        # Get ethical scores
        scores = self.ethical_validator(action_state)
        
        # Convert to dictionary
        score_dict = {
            principle.value: float(scores[0, i])
            for i, principle in enumerate(EthicalPrinciple)
        }
        
        # Check if all thresholds are met
        is_ethical = all(
            score_dict[principle.value] >= self.thresholds[principle]
            for principle in EthicalPrinciple
        )
        
        return is_ethical, score_dict
        
    def calculate_karma(self, actions: List[torch.Tensor]) -> float:
        """Calculate karmic balance from a sequence of actions."""
        if not actions:
            return 0.0
            
        # Validate each action
        karma_scores = []
        for action in actions:
            is_ethical, scores = self.validate_action(action)
            karma_score = np.mean(list(scores.values()))
            karma_scores.append(karma_score if is_ethical else -karma_score)
            
        # Calculate weighted sum with exponential decay
        weights = np.exp(-np.arange(len(karma_scores)) / 10)
        return float(np.sum(weights * karma_scores) / np.sum(weights))
        
    def protect_consciousness(self, state: torch.Tensor) -> torch.Tensor:
        """Apply ethical protection to consciousness state."""
        # Get ethical evaluation
        is_ethical, scores = self.validate_action(state)
        
        if not is_ethical:
            # Apply correction based on ethical principles
            corrections = []
            for principle in EthicalPrinciple:
                if scores[principle.value] < self.thresholds[principle]:
                    correction = self._generate_correction(principle)
                    corrections.append(correction)
                    
            if corrections:
                # Apply averaged correction
                correction = torch.stack(corrections).mean(dim=0)
                state = state + 0.1 * correction
                
        return state
        
    def _generate_correction(self, principle: EthicalPrinciple) -> torch.Tensor:
        """Generate correction pattern for an ethical principle."""
        # Create base pattern
        t = torch.linspace(0, 2*np.pi, self.config.karma_depth)
        
        if principle == EthicalPrinciple.COMPASSION:
            pattern = torch.sin(t * self.config.base_frequency)
        elif principle == EthicalPrinciple.NON_HARM:
            pattern = -torch.abs(torch.sin(t * self.config.base_frequency))
        elif principle == EthicalPrinciple.UNITY:
            pattern = torch.cos(t * self.config.base_frequency / 2)
        else:
            pattern = torch.sin(t * self.config.base_frequency / 3)
            
        return pattern.unsqueeze(0)
        
    def log_karma(self, action: torch.Tensor, result: Tuple[bool, Dict[str, float]]) -> None:
        """Log action and its karmic result."""
        self.karma_buffer.append({
            'action': action.detach().numpy(),
            'ethical': result[0],
            'scores': result[1],
            'timestamp': np.datetime64('now')
        })
        
        # Keep only recent history
        if len(self.karma_buffer) > 1000:
            self.karma_buffer.pop(0)
            
    def get_karma_history(self) -> List[Dict]:
        """Get the karma history buffer."""
        return self.karma_buffer 
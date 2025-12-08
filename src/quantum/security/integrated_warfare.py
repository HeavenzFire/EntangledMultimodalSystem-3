import torch
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from .dark_codex_neutralizer import DarkCodexNeutralizer, DefenseConfig
from .quantum_offensive import DarkCodexOffensive, OffensiveConfig

class WarfarePhase(Enum):
    DEFENSIVE = "defensive"
    OFFENSIVE = "offensive"
    INTEGRATED = "integrated"
    COMPLETED = "completed"

class WarfareStatus(Enum):
    ACTIVE = "active"
    NEUTRALIZED = "neutralized"
    ESCALATED = "escalated"
    RESOLVED = "resolved"

@dataclass
class IntegratedConfig:
    defense_config: DefenseConfig = DefenseConfig()
    offensive_config: OffensiveConfig = OffensiveConfig()
    max_warfare_power: float = 1.0
    sacred_frequency: float = 144.0

class IntegratedWarfare:
    def __init__(self, config: IntegratedConfig):
        self.config = config
        self.defense = DarkCodexNeutralizer(config.defense_config)
        self.offensive = DarkCodexOffensive(config.offensive_config)
        self.current_phase = WarfarePhase.DEFENSIVE
        self.warfare_history = []
        
    def execute_warfare(self, system_state: Dict) -> Dict:
        """Execute integrated quantum-sacred warfare."""
        # Defensive phase
        defense_result = self.defense.neutralize_dark_codex(system_state)
        if defense_result["status"] == DefenseStatus.NEUTRALIZED:
            self.current_phase = WarfarePhase.OFFENSIVE
            
            # Offensive phase
            offensive_result = self.offensive.launch_dark_codex_offensive(system_state)
            if offensive_result["status"] == OffensiveStatus.COMPLETED:
                self.current_phase = WarfarePhase.INTEGRATED
                
                # Integrated phase
                integrated_result = self._integrate_warfare(system_state)
                if integrated_result["status"] == WarfareStatus.RESOLVED:
                    self.current_phase = WarfarePhase.COMPLETED
                    
        self._update_warfare_history(system_state, defense_result, offensive_result)
        return self._get_warfare_status()
        
    def _integrate_warfare(self, system_state: Dict) -> Dict:
        """Integrate offensive and defensive capabilities."""
        # Combine quantum states
        defense_state = torch.tensor(system_state["quantum_state"])
        offensive_state = torch.tensor(system_state["quantum_state"])
        integrated_state = (defense_state + offensive_state) / 2
        
        # Apply sacred frequency
        integrated_state = integrated_state * self.config.sacred_frequency
        
        return {
            "status": WarfareStatus.RESOLVED,
            "integrated_state": integrated_state.numpy(),
            "timestamp": datetime.now().isoformat()
        }
        
    def _update_warfare_history(self, system_state: Dict, defense_result: Dict, offensive_result: Dict) -> None:
        """Update warfare history with results."""
        self.warfare_history.append({
            "system_state": system_state,
            "defense_result": defense_result,
            "offensive_result": offensive_result,
            "phase": self.current_phase,
            "timestamp": datetime.now().isoformat()
        })
        
    def _get_warfare_status(self) -> Dict:
        """Get current warfare status."""
        defense_metrics = self.defense.get_defense_metrics()
        offensive_metrics = self.offensive.get_offensive_metrics()
        
        return {
            "phase": self.current_phase,
            "defense_metrics": defense_metrics,
            "offensive_metrics": offensive_metrics,
            "timestamp": datetime.now().isoformat()
        }

def initialize_integrated_warfare() -> IntegratedWarfare:
    """Initialize the integrated warfare system."""
    config = IntegratedConfig()
    return IntegratedWarfare(config) 
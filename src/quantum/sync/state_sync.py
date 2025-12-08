import numpy as np
from typing import Dict, List, Any
import asyncio
from ..spiritual.tawhid_circuit import TawhidCircuit
from ..geometry.sacred_geometry import SacredGeometry

class QuantumStateSync:
    """Manages synchronization of quantum states between agents"""
    
    def __init__(self):
        self.tawhid_circuit = TawhidCircuit()
        self.sacred_geometry = SacredGeometry()
        self.entangled_states: Dict[str, Dict[str, Any]] = {}
        self.state_history: Dict[str, List[Any]] = {}
        
    async def initialize_agent_state(self, agent_id: str) -> Dict[str, Any]:
        """Initialize quantum state for a new agent"""
        initial_state = self.tawhid_circuit.create_initial_state()
        self.state_history[agent_id] = [initial_state]
        return initial_state
        
    async def update_agent_state(self, agent_id: str, transformation: Any) -> Dict[str, Any]:
        """Update quantum state for an agent"""
        if agent_id not in self.state_history:
            return await self.initialize_agent_state(agent_id)
            
        current_state = self.state_history[agent_id][-1]
        new_state = self.tawhid_circuit.apply_transformation(current_state, transformation)
        self.state_history[agent_id].append(new_state)
        
        # Update entangled states if necessary
        await self._update_entangled_states(agent_id, new_state)
        
        return new_state
        
    async def create_entanglement(self, agent1: str, agent2: str) -> None:
        """Create quantum entanglement between two agents"""
        if agent1 not in self.state_history or agent2 not in self.state_history:
            return
            
        state1 = self.state_history[agent1][-1]
        state2 = self.state_history[agent2][-1]
        
        entangled_state = self.tawhid_circuit.create_entanglement(state1, state2)
        
        self.entangled_states[f"{agent1}-{agent2}"] = {
            'agents': [agent1, agent2],
            'state': entangled_state,
            'created_at': asyncio.get_event_loop().time()
        }
        
    async def _update_entangled_states(self, agent_id: str, new_state: Any) -> None:
        """Update all entangled states involving the given agent"""
        for session_id, data in list(self.entangled_states.items()):
            if agent_id in data['agents']:
                other_agent = next(a for a in data['agents'] if a != agent_id)
                if other_agent in self.state_history:
                    other_state = self.state_history[other_agent][-1]
                    new_entangled = self.tawhid_circuit.create_entanglement(new_state, other_state)
                    self.entangled_states[session_id]['state'] = new_entangled
                    
    async def get_agent_state(self, agent_id: str) -> Dict[str, Any]:
        """Get current quantum state for an agent"""
        if agent_id not in self.state_history:
            return await self.initialize_agent_state(agent_id)
        return self.state_history[agent_id][-1]
        
    async def get_entangled_state(self, agent1: str, agent2: str) -> Dict[str, Any]:
        """Get entangled state between two agents"""
        session_id = f"{agent1}-{agent2}"
        if session_id in self.entangled_states:
            return self.entangled_states[session_id]['state']
        return None
        
    async def calculate_state_metrics(self, agent_id: str) -> Dict[str, float]:
        """Calculate metrics for an agent's quantum state"""
        if agent_id not in self.state_history:
            return {}
            
        current_state = self.state_history[agent_id][-1]
        
        # Calculate sacred geometry alignment
        geometric_alignment = self.sacred_geometry.calculate_alignment(current_state)
        
        # Calculate quantum coherence
        coherence = self.tawhid_circuit.calculate_coherence(current_state)
        
        # Calculate entanglement strength
        entanglement_strength = 0.0
        for session_id, data in self.entangled_states.items():
            if agent_id in data['agents']:
                strength = self.tawhid_circuit.calculate_entanglement_strength(
                    data['state']
                )
                entanglement_strength = max(entanglement_strength, strength)
                
        return {
            'geometric_alignment': geometric_alignment,
            'coherence': coherence,
            'entanglement_strength': entanglement_strength
        }
        
    async def reset_agent_state(self, agent_id: str) -> None:
        """Reset quantum state for an agent"""
        if agent_id in self.state_history:
            self.state_history[agent_id] = []
            
        # Remove all entanglements involving this agent
        for session_id in list(self.entangled_states.keys()):
            if agent_id in self.entangled_states[session_id]['agents']:
                del self.entangled_states[session_id]
                
    def get_state_history(self, agent_id: str) -> List[Any]:
        """Get complete state history for an agent"""
        return self.state_history.get(agent_id, []) 
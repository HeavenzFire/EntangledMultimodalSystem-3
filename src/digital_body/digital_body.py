from typing import Dict, List, Optional, Any
import asyncio
import logging
from pathlib import Path
import json
import numpy as np
from datetime import datetime

class DigitalBody:
    """Framework for agent embodiment and interaction with the digital world."""
    
    def __init__(self, name: str, capabilities: List[str], config: Optional[Dict] = None):
        self.name = name
        self.capabilities = capabilities
        self.config = config or {}
        self.logger = logging.getLogger(f"DigitalBody.{name}")
        self.sensors = {}
        self.actuators = {}
        self.memory = {}
        self.initialize_body()
        
    def initialize_body(self) -> None:
        """Initialize the digital body with its capabilities."""
        self.logger.info(f"Initializing digital body {self.name}")
        self.memory["start_time"] = datetime.now()
        self.memory["interactions"] = []
        
        # Initialize sensors based on capabilities
        if "vision" in self.capabilities:
            self.sensors["vision"] = self._initialize_vision()
        if "audio" in self.capabilities:
            self.sensors["audio"] = self._initialize_audio()
        if "text" in self.capabilities:
            self.sensors["text"] = self._initialize_text_processing()
            
        # Initialize actuators
        if "movement" in self.capabilities:
            self.actuators["movement"] = self._initialize_movement()
        if "speech" in self.capabilities:
            self.actuators["speech"] = self._initialize_speech()
        if "gesture" in self.capabilities:
            self.actuators["gesture"] = self._initialize_gestures()
    
    def _initialize_vision(self) -> Dict:
        """Initialize vision capabilities."""
        return {
            "active": False,
            "resolution": self.config.get("vision_resolution", (1920, 1080)),
            "fps": self.config.get("vision_fps", 30),
            "processing_pipeline": []
        }
    
    def _initialize_audio(self) -> Dict:
        """Initialize audio capabilities."""
        return {
            "active": False,
            "sample_rate": self.config.get("audio_sample_rate", 44100),
            "channels": self.config.get("audio_channels", 2),
            "processing_pipeline": []
        }
    
    def _initialize_text_processing(self) -> Dict:
        """Initialize text processing capabilities."""
        return {
            "active": False,
            "languages": self.config.get("languages", ["en"]),
            "processing_pipeline": []
        }
    
    def _initialize_movement(self) -> Dict:
        """Initialize movement capabilities."""
        return {
            "active": False,
            "speed": self.config.get("movement_speed", 1.0),
            "range": self.config.get("movement_range", (0, 100)),
            "current_position": (0, 0, 0)
        }
    
    def _initialize_speech(self) -> Dict:
        """Initialize speech capabilities."""
        return {
            "active": False,
            "voice": self.config.get("voice", "default"),
            "rate": self.config.get("speech_rate", 1.0),
            "volume": self.config.get("speech_volume", 1.0)
        }
    
    def _initialize_gestures(self) -> Dict:
        """Initialize gesture capabilities."""
        return {
            "active": False,
            "gestures": self.config.get("available_gestures", []),
            "current_gesture": None
        }
    
    async def process_sensory_input(self, input_type: str, data: Any) -> Dict:
        """Process sensory input from the environment."""
        if input_type not in self.sensors:
            raise ValueError(f"Unknown sensor type: {input_type}")
            
        self.logger.info(f"Processing {input_type} input")
        processed_data = await self._process_sensor_data(input_type, data)
        return processed_data
    
    async def _process_sensor_data(self, sensor_type: str, data: Any) -> Dict:
        """Process data from a specific sensor."""
        processing_pipeline = self.sensors[sensor_type]["processing_pipeline"]
        processed_data = data
        
        for processor in processing_pipeline:
            processed_data = await processor(processed_data)
            
        return {
            "sensor_type": sensor_type,
            "timestamp": datetime.now().isoformat(),
            "data": processed_data
        }
    
    async def execute_action(self, action_type: str, parameters: Dict) -> Dict:
        """Execute an action using the digital body's actuators."""
        if action_type not in self.actuators:
            raise ValueError(f"Unknown actuator type: {action_type}")
            
        self.logger.info(f"Executing {action_type} action")
        result = await self._execute_actuator_action(action_type, parameters)
        return result
    
    async def _execute_actuator_action(self, actuator_type: str, parameters: Dict) -> Dict:
        """Execute an action using a specific actuator."""
        actuator = self.actuators[actuator_type]
        
        if not actuator["active"]:
            self.logger.warning(f"Actuator {actuator_type} is not active")
            return {"status": "error", "message": "Actuator not active"}
            
        # Execute the action based on actuator type
        if actuator_type == "movement":
            return await self._execute_movement(parameters)
        elif actuator_type == "speech":
            return await self._execute_speech(parameters)
        elif actuator_type == "gesture":
            return await self._execute_gesture(parameters)
            
        return {"status": "error", "message": "Unknown action type"}
    
    async def _execute_movement(self, parameters: Dict) -> Dict:
        """Execute movement action."""
        target_position = parameters.get("position")
        speed = parameters.get("speed", self.actuators["movement"]["speed"])
        
        # Simulate movement
        current_pos = self.actuators["movement"]["current_position"]
        new_pos = tuple(np.array(current_pos) + np.array(target_position))
        
        self.actuators["movement"]["current_position"] = new_pos
        
        return {
            "status": "success",
            "new_position": new_pos,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _execute_speech(self, parameters: Dict) -> Dict:
        """Execute speech action."""
        text = parameters.get("text")
        voice = parameters.get("voice", self.actuators["speech"]["voice"])
        
        # Simulate speech
        return {
            "status": "success",
            "text": text,
            "voice": voice,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _execute_gesture(self, parameters: Dict) -> Dict:
        """Execute gesture action."""
        gesture = parameters.get("gesture")
        
        if gesture not in self.actuators["gesture"]["gestures"]:
            return {"status": "error", "message": "Unknown gesture"}
            
        self.actuators["gesture"]["current_gesture"] = gesture
        
        return {
            "status": "success",
            "gesture": gesture,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_status(self) -> Dict:
        """Get current status of the digital body."""
        return {
            "name": self.name,
            "capabilities": self.capabilities,
            "sensors": {k: v["active"] for k, v in self.sensors.items()},
            "actuators": {k: v["active"] for k, v in self.actuators.items()},
            "uptime": (datetime.now() - self.memory["start_time"]).total_seconds()
        }
    
    def activate_capability(self, capability: str) -> None:
        """Activate a specific capability."""
        if capability in self.sensors:
            self.sensors[capability]["active"] = True
        elif capability in self.actuators:
            self.actuators[capability]["active"] = True
        else:
            raise ValueError(f"Unknown capability: {capability}")
            
        self.logger.info(f"Activated capability: {capability}")
    
    def deactivate_capability(self, capability: str) -> None:
        """Deactivate a specific capability."""
        if capability in self.sensors:
            self.sensors[capability]["active"] = False
        elif capability in self.actuators:
            self.actuators[capability]["active"] = False
        else:
            raise ValueError(f"Unknown capability: {capability}")
            
        self.logger.info(f"Deactivated capability: {capability}") 
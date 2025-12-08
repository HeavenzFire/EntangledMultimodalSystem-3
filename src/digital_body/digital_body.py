from typing import Dict, List, Optional, Any
import asyncio
import logging
from pathlib import Path
import json
import numpy as np
from datetime import datetime
from .spatial_sensors import (
    SpatialSensors, PointCloud, SonarData, RadarData,
    EnvironmentalConditions, SensorCalibration,
    ThermalData, HyperspectralData, MagneticData, GravitationalData,
    QuantumData, BioelectricData, GeomagneticData, AtmosphericData
)

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
        self.spatial_sensors = SpatialSensors(config.get('spatial_sensors', {}))
        self.environmental_data = None
        self.environmental_update_interval = 300  # 5 minutes
        self.current_position = None
        self.quantum_state = None
        self.bioelectric_field = None
        self.geomagnetic_field = None
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
        if "lidar" in self.capabilities:
            self.sensors["lidar"] = self._initialize_lidar()
        if "sonar" in self.capabilities:
            self.sensors["sonar"] = self._initialize_sonar()
        if "radar" in self.capabilities:
            self.sensors["radar"] = self._initialize_radar()
        if "thermal" in self.capabilities:
            self.sensors["thermal"] = self._initialize_thermal()
        if "hyperspectral" in self.capabilities:
            self.sensors["hyperspectral"] = self._initialize_hyperspectral()
        if "magnetic" in self.capabilities:
            self.sensors["magnetic"] = self._initialize_magnetic()
        if "gravitational" in self.capabilities:
            self.sensors["gravitational"] = self._initialize_gravitational()
        if "quantum" in self.capabilities:
            self.sensors["quantum"] = self._initialize_quantum()
        if "bioelectric" in self.capabilities:
            self.sensors["bioelectric"] = self._initialize_bioelectric()
        if "geomagnetic" in self.capabilities:
            self.sensors["geomagnetic"] = self._initialize_geomagnetic()
        if "environmental" in self.capabilities:
            self.sensors["environmental"] = self._initialize_environmental()
            
        # Initialize actuators
        if "movement" in self.capabilities:
            self.actuators["movement"] = self._initialize_movement()
        if "speech" in self.capabilities:
            self.actuators["speech"] = self._initialize_speech()
        if "gesture" in self.capabilities:
            self.actuators["gesture"] = self._initialize_gestures()
            
        # Start environmental update task if environmental capability is enabled
        if "environmental" in self.capabilities:
            asyncio.create_task(self._environmental_update_task())
            
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
    
    def _initialize_lidar(self) -> Dict:
        """Initialize LIDAR sensor processing pipeline."""
        return {
            "processing_pipeline": [
                self._process_lidar_data
            ]
        }
        
    def _initialize_sonar(self) -> Dict:
        """Initialize SONAR sensor processing pipeline."""
        return {
            "processing_pipeline": [
                self._process_sonar_data
            ]
        }
        
    def _initialize_radar(self) -> Dict:
        """Initialize RADAR sensor processing pipeline."""
        return {
            "processing_pipeline": [
                self._process_radar_data
            ]
        }
    
    def _initialize_thermal(self) -> Dict:
        """Initialize thermal imaging capabilities."""
        return {
            "processing_pipeline": [
                self._process_thermal_data
            ]
        }
        
    def _initialize_hyperspectral(self) -> Dict:
        """Initialize hyperspectral imaging capabilities."""
        return {
            "processing_pipeline": [
                self._process_hyperspectral_data
            ]
        }
        
    def _initialize_magnetic(self) -> Dict:
        """Initialize magnetic field sensing capabilities."""
        return {
            "processing_pipeline": [
                self._process_magnetic_data
            ]
        }
        
    def _initialize_gravitational(self) -> Dict:
        """Initialize gravitational field sensing capabilities."""
        return {
            "processing_pipeline": [
                self._process_gravitational_data
            ]
        }
    
    def _initialize_quantum(self) -> Dict:
        """Initialize quantum sensing capabilities."""
        return {
            "processing_pipeline": [
                self._process_quantum_data
            ]
        }
        
    def _initialize_bioelectric(self) -> Dict:
        """Initialize bioelectric field sensing capabilities."""
        return {
            "processing_pipeline": [
                self._process_bioelectric_data
            ]
        }
        
    def _initialize_geomagnetic(self) -> Dict:
        """Initialize geomagnetic field sensing capabilities."""
        return {
            "processing_pipeline": [
                self._process_geomagnetic_data
            ]
        }
    
    def _initialize_environmental(self) -> Dict:
        """Initialize environmental data processing."""
        return {
            "processing_pipeline": [
                self._process_environmental_data
            ]
        }
    
    async def _process_lidar_data(self, data: np.ndarray) -> Dict:
        """Process LIDAR data using the spatial sensors module."""
        point_cloud = self.spatial_sensors.process_lidar(data)
        return {
            "type": "lidar",
            "point_cloud": point_cloud,
            "timestamp": datetime.now().isoformat()
        }
        
    async def _process_sonar_data(self, data: Dict) -> Dict:
        """Process SONAR data using the spatial sensors module."""
        sonar_data = self.spatial_sensors.process_sonar(data)
        return {
            "type": "sonar",
            "data": sonar_data,
            "timestamp": datetime.now().isoformat()
        }
        
    async def _process_radar_data(self, data: Dict) -> Dict:
        """Process RADAR data using the spatial sensors module."""
        radar_data = self.spatial_sensors.process_radar(data)
        return {
            "type": "radar",
            "data": radar_data,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _process_thermal_data(self, data: np.ndarray) -> Dict:
        """Process thermal imaging data."""
        thermal_data = self.spatial_sensors.process_thermal(data)
        return {
            "type": "thermal",
            "data": thermal_data,
            "timestamp": datetime.now().isoformat()
        }
        
    async def _process_hyperspectral_data(self, data: np.ndarray) -> Dict:
        """Process hyperspectral imaging data."""
        hyperspectral_data = self.spatial_sensors.process_hyperspectral(data)
        return {
            "type": "hyperspectral",
            "data": hyperspectral_data,
            "timestamp": datetime.now().isoformat()
        }
        
    async def _process_magnetic_data(self, data: Dict) -> Dict:
        """Process magnetic field measurements."""
        magnetic_data = self.spatial_sensors.process_magnetic(data)
        return {
            "type": "magnetic",
            "data": magnetic_data,
            "timestamp": datetime.now().isoformat()
        }
        
    async def _process_gravitational_data(self, data: Dict) -> Dict:
        """Process gravitational field measurements."""
        gravitational_data = self.spatial_sensors.process_gravitational(data)
        return {
            "type": "gravitational",
            "data": gravitational_data,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _process_quantum_data(self, data: Dict) -> Dict:
        """Process quantum sensor data."""
        quantum_data = self.spatial_sensors.process_quantum(data)
        self.quantum_state = quantum_data
        return {
            "type": "quantum",
            "data": quantum_data,
            "timestamp": datetime.now().isoformat()
        }
        
    async def _process_bioelectric_data(self, data: Dict) -> Dict:
        """Process bioelectric field data."""
        bioelectric_data = self.spatial_sensors.process_bioelectric(data)
        self.bioelectric_field = bioelectric_data
        return {
            "type": "bioelectric",
            "data": bioelectric_data,
            "timestamp": datetime.now().isoformat()
        }
        
    async def _process_geomagnetic_data(self, data: Dict) -> Dict:
        """Process geomagnetic field data."""
        geomagnetic_data = self.spatial_sensors.process_geomagnetic(data)
        self.geomagnetic_field = geomagnetic_data
        return {
            "type": "geomagnetic",
            "data": geomagnetic_data,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _process_environmental_data(self, data: Dict) -> Dict:
        """Process environmental data."""
        if self.environmental_data is None:
            await self.update_environmental_data()
            
        return {
            "type": "environmental",
            "data": self.environmental_data,
            "trends": self.spatial_sensors.get_environmental_trends(),
            "timestamp": datetime.now().isoformat()
        }
    
    async def _environmental_update_task(self) -> None:
        """Background task to periodically update environmental data."""
        while True:
            try:
                await self.update_environmental_data()
                await asyncio.sleep(self.environmental_update_interval)
            except Exception as e:
                self.logger.error(f"Error in environmental update task: {str(e)}")
                await asyncio.sleep(60)  # Wait a minute before retrying
                
    async def update_environmental_data(self) -> None:
        """Update environmental data for the current location."""
        if self.current_position is None:
            self.logger.warning("Cannot update environmental data: position not set")
            return
            
        try:
            self.environmental_data = await self.spatial_sensors.fetch_environmental_data(
                self.current_position[0],  # latitude
                self.current_position[1]   # longitude
            )
            
            # Calibrate sensors based on environmental conditions
            self.spatial_sensors.calibrate_sensors(self.environmental_data)
            
            # Update sensor parameters
            adjustments = self.spatial_sensors.adjust_sensor_parameters(self.environmental_data)
            self._apply_sensor_adjustments(adjustments)
            
            # Log environmental trends
            trends = self.spatial_sensors.get_environmental_trends()
            self.logger.info(f"Environmental trends: {trends}")
            
        except Exception as e:
            self.logger.error(f"Error updating environmental data: {str(e)}")
            raise
            
    def _apply_sensor_adjustments(self, adjustments: Dict) -> None:
        """Apply environmental-based adjustments to sensor parameters."""
        for sensor_type, params in adjustments.items():
            if sensor_type in self.sensors:
                self.sensors[sensor_type].update(params)
                
    def set_position(self, latitude: float, longitude: float) -> None:
        """Set the current position of the digital body."""
        self.current_position = (latitude, longitude)
        self.logger.info(f"Position updated to: {latitude}, {longitude}")
        
    async def process_sensory_input(self, input_type: str, data: Any) -> Dict:
        """Process sensory input from the environment."""
        if input_type not in self.sensors:
            raise ValueError(f"Unknown sensor type: {input_type}")
            
        self.logger.info(f"Processing {input_type} input")
        
        # Update environmental data if needed for spatial sensors
        if input_type in ["lidar", "sonar", "radar", "thermal", "hyperspectral", 
                         "magnetic", "gravitational", "quantum", "bioelectric", "geomagnetic"]:
            if self.environmental_data is None:
                await self.update_environmental_data()
            
        processed_data = await self._process_sensor_data(input_type, data)
        
        # If we have multiple spatial sensors, fuse their data
        if input_type in ["lidar", "sonar", "radar", "thermal", "hyperspectral", 
                         "magnetic", "gravitational", "quantum", "bioelectric", "geomagnetic"]:
            fused_data = await self._fuse_spatial_data(processed_data)
            return fused_data
            
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
    
    async def _fuse_spatial_data(self, processed_data: Dict) -> Dict:
        """Fuse data from multiple spatial sensors."""
        lidar_data = None
        sonar_data = None
        radar_data = None
        
        if processed_data["type"] == "lidar":
            lidar_data = processed_data["point_cloud"]
        elif processed_data["type"] == "sonar":
            sonar_data = processed_data["data"]
        elif processed_data["type"] == "radar":
            radar_data = processed_data["data"]
            
        fused_data = self.spatial_sensors.fuse_sensor_data(
            lidar_data=lidar_data,
            sonar_data=sonar_data,
            radar_data=radar_data
        )
        
        return {
            "type": "fused_spatial",
            "data": fused_data,
            "timestamp": datetime.now().isoformat()
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
        
    def get_environmental_status(self) -> Dict:
        """Get current environmental status and sensor calibration."""
        if self.environmental_data is None:
            return {"status": "no_data"}
            
        return {
            "environmental_data": self.environmental_data,
            "trends": self.spatial_sensors.get_environmental_trends(),
            "sensor_calibration": {
                sensor_type: {
                    "range": cal.range,
                    "sensitivity": cal.sensitivity,
                    "confidence": cal.confidence
                }
                for sensor_type, cal in self.spatial_sensors.sensor_calibration.items()
            }
        }

    def get_sensor_status(self) -> Dict:
        """Get current status of all sensors."""
        status = {
            "environmental": self.get_environmental_status(),
            "sensors": {}
        }
        
        for sensor_type, sensor in self.sensors.items():
            if sensor_type != "environmental":
                status["sensors"][sensor_type] = {
                    "active": sensor.get("active", False),
                    "calibration": self.spatial_sensors.sensor_calibration.get(sensor_type, {}),
                    "last_update": sensor.get("last_update", None)
                }
                
        # Add quantum state information
        if self.quantum_state is not None:
            status["quantum_state"] = {
                "coherence_time": self.quantum_state.coherence_time,
                "entanglement_measure": self.quantum_state.entanglement_measure,
                "decoherence_rate": self.quantum_state.decoherence_rate
            }
            
        # Add bioelectric field information
        if self.bioelectric_field is not None:
            status["bioelectric_field"] = {
                "field_strength": self.bioelectric_field.field_strength,
                "frequency_spectrum": self.bioelectric_field.frequency_spectrum.tolist(),
                "coherence_matrix": self.bioelectric_field.coherence_matrix.tolist()
            }
            
        # Add geomagnetic field information
        if self.geomagnetic_field is not None:
            status["geomagnetic_field"] = {
                "field_vector": self.geomagnetic_field.field_vector,
                "inclination": self.geomagnetic_field.inclination,
                "declination": self.geomagnetic_field.declination,
                "intensity": self.geomagnetic_field.intensity,
                "secular_variation": self.geomagnetic_field.secular_variation
            }
                
        return status 
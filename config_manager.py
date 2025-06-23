#!/usr/bin/env python3
"""
Configuration Manager for Trans-Dimensional Computational Framework
================================================================
This module provides configuration management for the Trans-Dimensional
Computational Framework, allowing for flexible and extensible configuration.
"""

import os
import json
import yaml
import logging
from typing import Dict, Any, Optional, Union, List

logger = logging.getLogger("TransFramework.Config")

class ConfigManager:
    """
    Configuration manager for the Trans-Dimensional Computational Framework.
    Handles loading, saving, and accessing configuration settings.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file, or None to use default
        """
        self.config_path = config_path or os.path.join(os.path.dirname(__file__), "config.json")
        self.config = self._load_default_config()
        
        # Load configuration from file if it exists
        if os.path.exists(self.config_path):
            self.load_config()
        
        logger.info(f"Configuration manager initialized with config path: {self.config_path}")
    
    def _load_default_config(self) -> Dict[str, Any]:
        """
        Load the default configuration.
        
        Returns:
            Default configuration dictionary
        """
        return {
            "framework": {
                "name": "Trans-Dimensional Computational Framework",
                "version": "1.0.0",
                "dimensions": 1000,
                "log_level": "INFO",
                "log_file": "trans_framework.log"
            },
            "quantum": {
                "backend": "numpy",
                "precision": "double",
                "max_qubits": 32,
                "optimization_level": 2
            },
            "memory": {
                "type": "hyper_dimensional",
                "max_dimensions": 1000,
                "compression": True
            },
            "processing": {
                "type": "quantum",
                "max_threads": 8,
                "use_gpu": False
            },
            "algorithms": {
                "shor": {
                    "enabled": True,
                    "max_bits": 16
                },
                "grover": {
                    "enabled": True,
                    "max_iterations": 100
                }
            },
            "consciousness": {
                "enabled": True,
                "dimensions": 11,
                "integration_level": "advanced"
            },
            "time_dilation": {
                "enabled": True,
                "max_factor": 10.0
            },
            "unity": {
                "enabled": True,
                "pulse_frequency": 1.0,
                "max_intensity": 5.0
            },
            "future_states": {
                "enabled": True,
                "max_steps": 100,
                "exploration_depth": 5
            },
            "multiagent": {
                "enabled": True,
                "max_agents": 10,
                "collaboration_mode": "quantum"
            }
        }
    
    def load_config(self, path: Optional[str] = None) -> None:
        """
        Load configuration from a file.
        
        Args:
            path: Path to the configuration file, or None to use the default
        """
        config_path = path or self.config_path
        
        try:
            if config_path.endswith(".json"):
                with open(config_path, "r") as f:
                    loaded_config = json.load(f)
            elif config_path.endswith(".yaml") or config_path.endswith(".yml"):
                with open(config_path, "r") as f:
                    loaded_config = yaml.safe_load(f)
            else:
                logger.error(f"Unsupported configuration file format: {config_path}")
                return
            
            # Merge loaded config with default config
            self._merge_configs(self.config, loaded_config)
            logger.info(f"Configuration loaded from {config_path}")
        
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {e}")
    
    def _merge_configs(self, default: Dict[str, Any], loaded: Dict[str, Any]) -> None:
        """
        Merge loaded configuration with default configuration.
        
        Args:
            default: Default configuration dictionary
            loaded: Loaded configuration dictionary
        """
        for key, value in loaded.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self._merge_configs(default[key], value)
            else:
                default[key] = value
    
    def save_config(self, path: Optional[str] = None) -> None:
        """
        Save configuration to a file.
        
        Args:
            path: Path to the configuration file, or None to use the default
        """
        config_path = path or self.config_path
        
        try:
            if config_path.endswith(".json"):
                with open(config_path, "w") as f:
                    json.dump(self.config, f, indent=4)
            elif config_path.endswith(".yaml") or config_path.endswith(".yml"):
                with open(config_path, "w") as f:
                    yaml.dump(self.config, f, default_flow_style=False)
            else:
                logger.error(f"Unsupported configuration file format: {config_path}")
                return
            
            logger.info(f"Configuration saved to {config_path}")
        
        except Exception as e:
            logger.error(f"Error saving configuration to {config_path}: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.
        
        Args:
            key: Configuration key (dot notation supported)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value by key.
        
        Args:
            key: Configuration key (dot notation supported)
            value: Configuration value
        """
        keys = key.split(".")
        config = self.config
        
        for i, k in enumerate(keys[:-1]):
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        logger.info(f"Configuration value set: {key} = {value}")
    
    def get_framework_config(self) -> Dict[str, Any]:
        """
        Get the framework configuration.
        
        Returns:
            Framework configuration dictionary
        """
        return self.config.get("framework", {})
    
    def get_quantum_config(self) -> Dict[str, Any]:
        """
        Get the quantum configuration.
        
        Returns:
            Quantum configuration dictionary
        """
        return self.config.get("quantum", {})
    
    def get_memory_config(self) -> Dict[str, Any]:
        """
        Get the memory configuration.
        
        Returns:
            Memory configuration dictionary
        """
        return self.config.get("memory", {})
    
    def get_processing_config(self) -> Dict[str, Any]:
        """
        Get the processing configuration.
        
        Returns:
            Processing configuration dictionary
        """
        return self.config.get("processing", {})
    
    def get_algorithms_config(self) -> Dict[str, Any]:
        """
        Get the algorithms configuration.
        
        Returns:
            Algorithms configuration dictionary
        """
        return self.config.get("algorithms", {})
    
    def get_consciousness_config(self) -> Dict[str, Any]:
        """
        Get the consciousness configuration.
        
        Returns:
            Consciousness configuration dictionary
        """
        return self.config.get("consciousness", {})
    
    def get_time_dilation_config(self) -> Dict[str, Any]:
        """
        Get the time dilation configuration.
        
        Returns:
            Time dilation configuration dictionary
        """
        return self.config.get("time_dilation", {})
    
    def get_unity_config(self) -> Dict[str, Any]:
        """
        Get the unity configuration.
        
        Returns:
            Unity configuration dictionary
        """
        return self.config.get("unity", {})
    
    def get_future_states_config(self) -> Dict[str, Any]:
        """
        Get the future states configuration.
        
        Returns:
            Future states configuration dictionary
        """
        return self.config.get("future_states", {})
    
    def get_multiagent_config(self) -> Dict[str, Any]:
        """
        Get the multi-agent configuration.
        
        Returns:
            Multi-agent configuration dictionary
        """
        return self.config.get("multiagent", {})

def create_default_config(config_path: str) -> None:
    """
    Create a default configuration file.
    
    Args:
        config_path: Path to the configuration file
    """
    config_manager = ConfigManager()
    config_manager.save_config(config_path)
    logger.info(f"Default configuration created at {config_path}")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("config_manager.log"),
            logging.StreamHandler()
        ]
    )
    
    # Create a default configuration file
    create_default_config("config.json")
    
    # Load and print the configuration
    config_manager = ConfigManager("config.json")
    print(json.dumps(config_manager.config, indent=2)) 
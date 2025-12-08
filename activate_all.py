#!/usr/bin/env python3
"""
EntangledMultimodalSystem - Full System Activation Script
Date: April 15, 2025

This script provides a simple way to activate the entire EntangledMultimodalSystem
with a single command. It calls the main orchestrator with the --activate-all flag.
"""

import sys
import os
import json
import time
from main import EntangledOrchestrator

def activate_system():
    """Activate the complete EntangledMultimodalSystem"""
    print("-" * 80)
    print("ENTANGLED MULTIMODAL SYSTEM - FULL ACTIVATION")
    print(f"Date: {time.ctime()}")
    print("-" * 80)
    
    # Load configuration from system_config.json
    config_path = os.path.join(os.path.dirname(__file__), 'system_config.json')
    
    if not os.path.exists(config_path):
        print(f"Configuration file not found: {config_path}")
        print("Creating a default configuration file...")
        
        # Create default configuration
        default_config = {
            "quantum_layers": 5,
            "num_qubits": 16,
            "classical_weight": 0.4,
            "quantum_weight": 0.4,
            "fractal_weight": 0.2,
            "fractal_dimension": 1.67,
            "threat_threshold": 0.85,
            "chaos_factor": 0.15,
            "interface": {
                "enable_vscode_integration": True,
                "flask_port": 5000
            }
        }
        
        # Save default configuration
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        print(f"Default configuration saved to {config_path}")
    
    print("Initializing orchestrator...")
    orchestrator = EntangledOrchestrator(config_path, verbose=True)
    
    print("Activating all system components...")
    result = orchestrator.activate_all()
    
    print("\n" + "=" * 80)
    print("ACTIVATION COMPLETE")
    print("=" * 80)
    print(f"Status: {result['status']}")
    print(f"System coherence: {result['coherence'].get('coherence', 'unknown')}")
    print(f"System status: {result['coherence'].get('status', 'unknown')}")
    print(f"VS Code integration: {'Enabled' if result.get('vscode_integrated') else 'Disabled'}")
    print("\nThe EntangledMultimodalSystem is now fully operational.")
    print("Press Ctrl+C to shutdown the system.")
    
    try:
        # Keep the script running to maintain all background threads
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down EntangledMultimodalSystem...")
        print("Thank you for using the EntangledMultimodalSystem!")

if __name__ == "__main__":
    activate_system()
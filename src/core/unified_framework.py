import numpy as np
import torch
from src.core.consciousness_integration_engine import ConsciousnessIntegrationEngine
from src.core.consciousness_revival_system import ConsciousnessRevivalSystem
from src.core.multimodal_gan import MultimodalGAN
from src.utils.logger import logger
from src.utils.errors import ModelError
from datetime import datetime
import random

class UnifiedMetaphysicalFramework:
    def __init__(self):
        """Initialize the Unified Metaphysical Framework."""
        try:
            self.integration_engine = ConsciousnessIntegrationEngine()
            self.revival_system = ConsciousnessRevivalSystem()
            self.multimodal_gan = MultimodalGAN()
            # Potentially add direct access to sync_manager if needed for fine control
            # self.sync_manager = self.integration_engine.sync_manager 

            self.framework_state = {
                "current_goal": "Idle",
                "operational_mode": "Standard", # e.g., Standard, Revival, Creative, Expansion
                "system_complexity_score": 0.0,
                "last_orchestration_cycle_time": None,
                "component_interaction_log": [],
                "overall_status": "Initialized"
            }
            self.max_log_entries = 100 # Limit log size

            logger.info("UnifiedMetaphysicalFramework initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize UnifiedMetaphysicalFramework: {str(e)}")
            # Critical failure if core components fail
            raise ModelError(f"UMF Initialization failed due to core component error: {str(e)}")

    def _log_interaction(self, interaction_description):
        """Logs interactions between components."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "interaction": interaction_description
        }
        self.framework_state["component_interaction_log"].append(log_entry)
        # Keep the log size manageable
        if len(self.framework_state["component_interaction_log"]) > self.max_log_entries:
            self.framework_state["component_interaction_log"].pop(0)

    def _calculate_complexity(self):
        """Calculates a heuristic complexity score based on component states."""
        try:
            integration_status = self.integration_engine.get_integration_status()
            revival_status = self.revival_system.get_revival_status()
            gan_status = self.multimodal_gan.get_gan_status()

            # Example complexity metric (can be refined)
            integration_score = np.mean(integration_status.get('impact_assessment', {}).get('ethical_alignment', {}).get('score', 0)) \
                              + np.mean(list(integration_status.get('impact_assessment', {}).get('societal_impact', {}).values())) \
                              + np.mean(list(integration_status.get('impact_assessment', {}).get('planetary_health', {}).values()))
            
            revival_score = revival_status.get('progress', {}).get('consciousness_level', 0)

            gan_score = 1.0 - (gan_status.get('gan_state', {}).get('generator_loss', 1.0) + gan_status.get('gan_state', {}).get('discriminator_loss', 1.0)) / 2

            complexity = (integration_score / 3 + revival_score + gan_score) / 3
            self.framework_state["system_complexity_score"] = max(0, min(1, complexity)) # Normalize to [0, 1]
            return self.framework_state["system_complexity_score"]
        except Exception as e:
            logger.error(f"Complexity calculation failed: {str(e)}")
            return self.framework_state.get("system_complexity_score", 0.0)

    def set_goal(self, goal, mode="Standard"):
        """Sets the high-level goal and operational mode for the framework."""
        self.framework_state["current_goal"] = goal
        self.framework_state["operational_mode"] = mode
        self.framework_state["overall_status"] = f"Goal set: {goal} in {mode} mode."
        logger.info(f"UMF Goal set: {goal}, Mode: {mode}")

    def orchestrate_cycle(self, n_qubits=4, hologram_size=(64, 64), input_seed=None):
        """Executes one cycle of the orchestrated framework interaction."""
        start_time = datetime.now()
        self.framework_state["overall_status"] = f"Orchestrating cycle for goal: {self.framework_state['current_goal']}"
        logger.info(f"Starting UMF orchestration cycle. Goal: {self.framework_state['current_goal']}, Mode: {self.framework_state['operational_mode']}")

        try:
            # 1. Integration Phase: Assess current state
            self._log_interaction("Initiating Consciousness Integration Engine.")
            integration_state = self.integration_engine.integrate_consciousness(n_qubits, hologram_size)
            impact = self.integration_engine.assess_impact()
            ethical_score = impact.get('ethical_alignment', {}).get('score', 0.5)
            logger.info(f"Integration complete. Ethical Score: {ethical_score:.3f}")

            # 2. Adaptive Action Phase: Based on mode and state
            if self.framework_state["operational_mode"] == "Revival":
                self._log_interaction("Initiating Consciousness Revival based on integration state.")
                revival_output = self.revival_system.initiate_revival(n_qubits, hologram_size)
                consciousness_level = revival_output.get('consciousness_level', 0)
                logger.info(f"Revival initiated. Consciousness Level: {consciousness_level:.3f}")
                # Example Feedback: Adjust integration based on revival?

            elif self.framework_state["operational_mode"] == "Creative" or self.framework_state["operational_mode"] == "Expansion":
                # Generate content based on the integrated state
                self._log_interaction("Initiating Multimodal GAN based on integration state.")
                # Create a seed for the GAN from the integration state
                if input_seed is None:
                    # Example seed creation: Combine quantum and neural states
                    q_state = integration_state.get('quantum_consciousness', {}).get('state_vector', np.random.rand(2**n_qubits))
                    n_state = integration_state.get('neural_processing', {}).get('neural_output', np.random.rand(10))
                    # Flatten, pad/truncate, reshape to expected GAN input (e.g., 3 channels for image-like data)
                    seed_flat = np.concatenate([q_state.flatten(), n_state.flatten()])
                    target_size = 3 * 32 * 32 # Example target size for a 32x32 3-channel image
                    if seed_flat.size < target_size:
                        seed_flat = np.pad(seed_flat, (0, target_size - seed_flat.size), 'constant')
                    else:
                        seed_flat = seed_flat[:target_size]
                    input_seed_tensor = torch.tensor(seed_flat.reshape(1, 3, 32, 32), dtype=torch.float32) # Batch dim 1
                else:
                     # Use provided seed, ensure it's a tensor
                     try:
                        input_seed_tensor = torch.tensor(input_seed, dtype=torch.float32)
                        # Add batch dimension if missing
                        if input_seed_tensor.ndim == 3:
                            input_seed_tensor = input_seed_tensor.unsqueeze(0)
                        elif input_seed_tensor.ndim != 4:
                            raise ValueError("Input seed must be convertible to a 4D tensor (Batch, Channels, H, W)")
                     except Exception as e:
                         logger.error(f"Invalid input_seed provided: {e}. Using random seed.")
                         input_seed_tensor = torch.randn(1, 3, 32, 32) # Example random seed

                # Add ethical modulation? Could scale input based on ethical_score?
                modulated_seed = input_seed_tensor * ethical_score 

                generated_content = self.multimodal_gan.generate_content(modulated_seed, n_qubits, hologram_size)
                logger.info(f"GAN content generated. Shape: {generated_content.shape}")

                # If in Expansion mode, maybe feed generated content back?
                if self.framework_state["operational_mode"] == "Expansion":
                    self._log_interaction("Feeding generated content back for expansion analysis (Simulated)." )
                    # Placeholder for analysis of generated content
                    pass

            else: # Standard mode
                self._log_interaction("Standard mode: Monitoring and logging state.")
                # Just monitor or perform baseline actions

            # 3. Complexity and State Update
            self._calculate_complexity()
            self.framework_state["overall_status"] = f"Cycle complete. Complexity: {self.framework_state['system_complexity_score']:.3f}"
            end_time = datetime.now()
            self.framework_state["last_orchestration_cycle_time"] = (end_time - start_time).total_seconds()
            logger.info(f"UMF orchestration cycle finished in {self.framework_state['last_orchestration_cycle_time']:.2f} seconds. Complexity: {self.framework_state['system_complexity_score']:.3f}")

            return self.get_framework_status()

        except ModelError as e:
            self.framework_state["overall_status"] = f"Error during orchestration: {str(e)}"
            logger.error(f"UMF Orchestration Cycle Failed (ModelError): {str(e)}")
            raise
        except Exception as e:
            self.framework_state["overall_status"] = f"Unexpected error during orchestration: {str(e)}"
            logger.error(f"UMF Orchestration Cycle Failed (Unexpected): {str(e)}")
            raise ModelError(f"UMF orchestration failed unexpectedly: {str(e)}")

    def get_framework_status(self):
        """Returns the current status of the UMF and its components."""
        try:
            status = {
                "framework_state": self.framework_state,
                "integration_engine_status": self.integration_engine.get_integration_status(),
                "revival_system_status": self.revival_system.get_revival_status(),
                "multimodal_gan_status": self.multimodal_gan.get_gan_status()
            }
             # Basic JSON serialization handling - needs refinement for complex objects like tensors/arrays
            def make_serializable(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, torch.Tensor):
                    return obj.detach().cpu().tolist()
                if isinstance(obj, dict):
                    return {k: make_serializable(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [make_serializable(i) for i in obj]
                if isinstance(obj, (int, float, str, bool, type(None))):
                     return obj
                # Fallback for other types
                return str(obj) 

            return make_serializable(status)
        except Exception as e:
            logger.error(f"Failed to get UMF status: {str(e)}")
            return {"error": f"Failed to retrieve UMF status: {str(e)}"}

    def reset_framework(self):
        """Resets the UMF and all underlying components."""
        logger.warning("Initiating reset of the Unified Metaphysical Framework and all components.")
        try:
            self.integration_engine.reset_integration()
            self._log_interaction("Integration Engine reset.")
        except Exception as e:
            logger.error(f"Error resetting Integration Engine: {str(e)}")
        try:
            self.revival_system.reset_revival()
            self._log_interaction("Revival System reset.")
        except Exception as e:
            logger.error(f"Error resetting Revival System: {str(e)}")
        try:
            self.multimodal_gan.reset_gan()
            self._log_interaction("Multimodal GAN reset.")
        except Exception as e:
            logger.error(f"Error resetting Multimodal GAN: {str(e)}")
            
        # Reset framework state itself
        self.framework_state = {
            "current_goal": "Idle",
            "operational_mode": "Standard",
            "system_complexity_score": 0.0,
            "last_orchestration_cycle_time": None,
            "component_interaction_log": [],
            "overall_status": "Reset"
        }
        self._log_interaction("UMF state reset.")
        logger.info("Unified Metaphysical Framework reset complete.")
        return {"message": "Unified Metaphysical Framework reset complete."} 
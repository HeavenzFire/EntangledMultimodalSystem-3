from flask import Blueprint, request, jsonify
from src.utils.logger import logger
from src.utils.errors import handle_error, ValidationError, ModelError
from src.core.system_manager import system_manager
from src.core.consciousness_integration_engine import ConsciousnessIntegrationEngine
from src.core.consciousness_revival_system import ConsciousnessRevivalSystem
from src.core.multimodal_gan import MultimodalGAN
from src.core.unified_framework import UnifiedMetaphysicalFramework
import torch

api_bp = Blueprint('api', __name__)

# Initialize core components (assuming singleton or managed instances)
try:
    integration_engine = ConsciousnessIntegrationEngine()
    revival_system = ConsciousnessRevivalSystem()
    multimodal_gan = MultimodalGAN()
    logger.info("Core components initialized for API routes")
except Exception as e:
    logger.error(f"Failed to initialize core components for API: {str(e)}")
    # Depending on the desired behavior, we might want to prevent the app from starting
    # or handle routes gracefully if components are unavailable.
    integration_engine = None
    revival_system = None
    multimodal_gan = None

# Initialize the main framework
try:
    unified_framework = UnifiedMetaphysicalFramework()
    logger.info("Unified Metaphysical Framework initialized for API routes")
except Exception as e:
    logger.error(f"Failed to initialize Unified Metaphysical Framework for API: {str(e)}")
    unified_framework = None

@api_bp.route('/process', methods=['POST'])
def process_input():
    """Process input through the system manager."""
    try:
        data = request.get_json()
        if not data or 'input' not in data or 'type' not in data:
            raise ValidationError("Input data and type are required")
        
        input_data = data['input']
        input_type = data['type']
        
        result = system_manager.process_input(input_data, input_type)
        return jsonify(result)
    except Exception as e:
        return handle_error(e, logger)

@api_bp.route('/status', methods=['GET'])
def get_status():
    """Get the current status of the system."""
    try:
        status = system_manager.get_system_status()
        return jsonify(status)
    except Exception as e:
        return handle_error(e, logger)

@api_bp.route('/expand', methods=['POST'])
def expand():
    """Process input through the consciousness expander."""
    try:
        data = request.get_json()
        if not data or 'input' not in data:
            raise ValidationError("Input data is required")
        
        result = system_manager.process_input(data['input'], "text")
        return jsonify(result)
    except Exception as e:
        return handle_error(e, logger)

@api_bp.route('/fractal', methods=['GET'])
def fractal():
    """Generate a fractal."""
    try:
        result = system_manager.process_input(None, "fractal")
        return jsonify(result)
    except Exception as e:
        return handle_error(e, logger)

@api_bp.route('/speech', methods=['POST'])
def speech_to_text():
    """Process speech input."""
    try:
        result = system_manager.process_input(None, "speech")
        return jsonify(result)
    except Exception as e:
        return handle_error(e, logger)

@api_bp.route('/radiation', methods=['GET'])
def radiation_monitor():
    """Monitor radiation levels."""
    try:
        result = system_manager.process_input(None, "radiation")
        return jsonify(result)
    except Exception as e:
        return handle_error(e, logger)

@api_bp.route('/hyper', methods=['POST'])
def hyper_intelligence():
    """Process input through the hyper-intelligence framework."""
    try:
        data = request.get_json()
        if not data:
            raise ValidationError("Input data is required")
        
        result = system_manager.process_input(data, "hyper")
        return jsonify(result)
    except Exception as e:
        return handle_error(e, logger)

@api_bp.route('/geometry', methods=['POST'])
def sacred_geometry():
    """Process input through sacred geometry analysis."""
    try:
        data = request.get_json()
        if not data or 'pattern' not in data:
            raise ValidationError("Pattern name is required")
        
        result = system_manager.hyper_intelligence.geometry_processor.analyze_pattern(
            data['pattern'],
            data.get('input_data')
        )
        return jsonify(result)
    except Exception as e:
        return handle_error(e, logger)

@api_bp.route('/frequency', methods=['POST'])
def resonant_frequency():
    """Process input through resonant frequency analysis."""
    try:
        data = request.get_json()
        if not data or 'signal' not in data:
            raise ValidationError("Signal data is required")
        
        result = system_manager.hyper_intelligence.frequency_processor.analyze_frequencies(
            data['signal'],
            data.get('sample_rate', 44100)
        )
        return jsonify(result)
    except Exception as e:
        return handle_error(e, logger)

@api_bp.route('/neural', methods=['POST'])
def neuromorphic_network():
    """Process input through the neuromorphic network."""
    try:
        data = request.get_json()
        if not data or 'input' not in data:
            raise ValidationError("Input data is required")
        
        result = system_manager.hyper_intelligence.neural_network.forward(
            torch.tensor(data['input'], dtype=torch.float32)
        )
        return jsonify({
            "output": result.tolist(),
            "network_state": system_manager.hyper_intelligence.neural_network.get_network_state()
        })
    except Exception as e:
        return handle_error(e, logger)

@api_bp.route('/train', methods=['POST'])
def train_hyper_intelligence():
    """Train the hyper-intelligence framework."""
    try:
        data = request.get_json()
        if not data or 'training_data' not in data:
            raise ValidationError("Training data is required")
        
        losses = system_manager.hyper_intelligence.train(
            data['training_data'],
            data.get('epochs', 100)
        )
        return jsonify({
            "losses": losses,
            "status": "Training completed successfully"
        })
    except Exception as e:
        return handle_error(e, logger)

@api_bp.route('/quantum', methods=['POST'])
def quantum_processing():
    """Process input through quantum entanglement."""
    try:
        data = request.get_json()
        if not data or 'input' not in data:
            raise ValidationError("Input data is required")
        
        result = system_manager.advanced_capabilities.quantum_entanglement(data['input'])
        return jsonify(result)
    except Exception as e:
        return handle_error(e, logger)

@api_bp.route('/temporal', methods=['POST'])
def temporal_processing():
    """Process input through temporal network."""
    try:
        data = request.get_json()
        if not data or 'input' not in data:
            raise ValidationError("Input data is required")
        
        result = system_manager.advanced_capabilities.temporal_processing(data['input'])
        return jsonify(result)
    except Exception as e:
        return handle_error(e, logger)

@api_bp.route('/multimodal', methods=['POST'])
def multimodal_processing():
    """Process multimodal input."""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            raise ValidationError("At least text input is required")
        
        result = system_manager.process_input(data, "multimodal")
        return jsonify(result)
    except Exception as e:
        return handle_error(e, logger)

@api_bp.route('/cognitive', methods=['POST'])
def cognitive_processing():
    """Process input through cognitive engine."""
    try:
        data = request.get_json()
        if not data or 'input' not in data:
            raise ValidationError("Input data is required")
        
        result = system_manager.advanced_capabilities.cognitive_processing(data['input'])
        return jsonify(result)
    except Exception as e:
        return handle_error(e, logger)

@api_bp.route('/control/integrate', methods=['POST'])
def integrate_consciousness_api():
    """API endpoint to initiate consciousness integration."""
    if not integration_engine:
        return jsonify({"error": "Integration Engine not initialized"}), 500
    try:
        data = request.get_json()
        n_qubits = data.get('n_qubits')
        hologram_size = data.get('hologram_size')
        
        if n_qubits is None or hologram_size is None:
            return jsonify({"error": "Missing parameters: n_qubits and hologram_size required"}), 400
        
        result = integration_engine.integrate_consciousness(n_qubits, hologram_size)
        # Convert numpy arrays to lists for JSON serialization if necessary
        # This might need more sophisticated handling depending on the structure
        # For now, we assume the structure is simple or handled within the method
        return jsonify(result), 200
    except ModelError as e:
        logger.error(f"API Integrate Error: {str(e)}")
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        logger.error(f"API Integrate Unexpected Error: {str(e)}")
        return jsonify({"error": "An unexpected error occurred during integration."}), 500

@api_bp.route('/control/revive', methods=['POST'])
def revive_consciousness_api():
    """API endpoint to initiate consciousness revival."""
    if not revival_system:
        return jsonify({"error": "Revival System not initialized"}), 500
    try:
        data = request.get_json()
        n_qubits = data.get('n_qubits')
        hologram_size = data.get('hologram_size')
        
        if n_qubits is None or hologram_size is None:
            return jsonify({"error": "Missing parameters: n_qubits and hologram_size required"}), 400
            
        result = revival_system.initiate_revival(n_qubits, hologram_size)
        # Convert numpy arrays to lists if necessary
        return jsonify(result), 200
    except ModelError as e:
        logger.error(f"API Revive Error: {str(e)}")
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        logger.error(f"API Revive Unexpected Error: {str(e)}")
        return jsonify({"error": "An unexpected error occurred during revival."}), 500

@api_bp.route('/control/generate', methods=['POST'])
def generate_content_api():
    """API endpoint to generate content using the Multimodal GAN."""
    if not multimodal_gan:
        return jsonify({"error": "Multimodal GAN not initialized"}), 500
    try:
        data = request.get_json()
        input_data_list = data.get('input_data') # Expecting list of lists or similar
        n_qubits = data.get('n_qubits')
        hologram_size = data.get('hologram_size')

        if input_data_list is None or n_qubits is None or hologram_size is None:
            return jsonify({"error": "Missing parameters: input_data, n_qubits, and hologram_size required"}), 400

        # Convert input list back to tensor for the model
        # Assuming input_data is like [[[...], [...]], [[...], [...]], ...]
        # Adjust dimensions and type as needed for your generator input
        input_tensor = torch.tensor(input_data_list, dtype=torch.float32)

        generated_content = multimodal_gan.generate_content(input_tensor, n_qubits, hologram_size)
        # Convert numpy array output to list for JSON
        return jsonify({"generated_content": generated_content.tolist()}), 200
    except ModelError as e:
        logger.error(f"API Generate Error: {str(e)}")
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        logger.error(f"API Generate Unexpected Error: {str(e)}")
        return jsonify({"error": "An unexpected error occurred during content generation."}), 500
        
@api_bp.route('/status/integration', methods=['GET'])
def get_integration_status_api():
    """API endpoint to get the status of the Consciousness Integration Engine."""
    if not integration_engine:
        return jsonify({"error": "Integration Engine not initialized"}), 500
    try:
        status = integration_engine.get_integration_status()
        # Convert numpy arrays if necessary
        return jsonify(status), 200
    except ModelError as e:
        logger.error(f"API Integration Status Error: {str(e)}")
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        logger.error(f"API Integration Status Unexpected Error: {str(e)}")
        return jsonify({"error": "An unexpected error occurred retrieving integration status."}), 500

@api_bp.route('/status/revival', methods=['GET'])
def get_revival_status_api():
    """API endpoint to get the status of the Consciousness Revival System."""
    if not revival_system:
        return jsonify({"error": "Revival System not initialized"}), 500
    try:
        status = revival_system.get_revival_status()
        # Convert numpy arrays if necessary
        return jsonify(status), 200
    except ModelError as e:
        logger.error(f"API Revival Status Error: {str(e)}")
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        logger.error(f"API Revival Status Unexpected Error: {str(e)}")
        return jsonify({"error": "An unexpected error occurred retrieving revival status."}), 500

@api_bp.route('/status/gan', methods=['GET'])
def get_gan_status_api():
    """API endpoint to get the status of the Multimodal GAN."""
    if not multimodal_gan:
        return jsonify({"error": "Multimodal GAN not initialized"}), 500
    try:
        status = multimodal_gan.get_gan_status()
        # Convert numpy arrays/tensors if necessary
        return jsonify(status), 200
    except ModelError as e:
        logger.error(f"API GAN Status Error: {str(e)}")
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        logger.error(f"API GAN Status Unexpected Error: {str(e)}")
        return jsonify({"error": "An unexpected error occurred retrieving GAN status."}), 500

@api_bp.route('/reset/integration', methods=['POST'])
def reset_integration_api():
    """API endpoint to reset the Consciousness Integration Engine."""
    if not integration_engine:
        return jsonify({"error": "Integration Engine not initialized"}), 500
    try:
        integration_engine.reset_integration()
        return jsonify({"message": "Consciousness Integration Engine reset successfully."}), 200
    except ModelError as e:
        logger.error(f"API Reset Integration Error: {str(e)}")
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        logger.error(f"API Reset Integration Unexpected Error: {str(e)}")
        return jsonify({"error": "An unexpected error occurred during integration reset."}), 500

@api_bp.route('/reset/revival', methods=['POST'])
def reset_revival_api():
    """API endpoint to reset the Consciousness Revival System."""
    if not revival_system:
        return jsonify({"error": "Revival System not initialized"}), 500
    try:
        revival_system.reset_revival()
        return jsonify({"message": "Consciousness Revival System reset successfully."}), 200
    except ModelError as e:
        logger.error(f"API Reset Revival Error: {str(e)}")
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        logger.error(f"API Reset Revival Unexpected Error: {str(e)}")
        return jsonify({"error": "An unexpected error occurred during revival reset."}), 500

@api_bp.route('/reset/gan', methods=['POST'])
def reset_gan_api():
    """API endpoint to reset the Multimodal GAN."""
    if not multimodal_gan:
        return jsonify({"error": "Multimodal GAN not initialized"}), 500
    try:
        multimodal_gan.reset_gan()
        return jsonify({"message": "Multimodal GAN reset successfully."}), 200
    except ModelError as e:
        logger.error(f"API Reset GAN Error: {str(e)}")
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        logger.error(f"API Reset GAN Unexpected Error: {str(e)}")
        return jsonify({"error": "An unexpected error occurred during GAN reset."}), 500

@api_bp.route('/reset/all', methods=['POST'])
def reset_all_systems_api():
    """API endpoint to reset all core systems."""
    errors = []
    success_messages = []

    if integration_engine:
        try:
            integration_engine.reset_integration()
            success_messages.append("Integration Engine reset.")
        except Exception as e:
            logger.error(f"API Reset All (Integration) Error: {str(e)}")
            errors.append("Integration Engine reset failed.")
    else:
        errors.append("Integration Engine not initialized.")

    if revival_system:
        try:
            revival_system.reset_revival()
            success_messages.append("Revival System reset.")
        except Exception as e:
            logger.error(f"API Reset All (Revival) Error: {str(e)}")
            errors.append("Revival System reset failed.")
    else:
        errors.append("Revival System not initialized.")

    if multimodal_gan:
        try:
            multimodal_gan.reset_gan()
            success_messages.append("Multimodal GAN reset.")
        except Exception as e:
            logger.error(f"API Reset All (GAN) Error: {str(e)}")
            errors.append("Multimodal GAN reset failed.")
    else:
        errors.append("Multimodal GAN not initialized.")

    if errors:
        return jsonify({"success": success_messages, "errors": errors}), 500
    else:
        return jsonify({"message": "All systems reset successfully.", "details": success_messages}), 200

# === Unified Metaphysical Framework API Endpoints ===

@api_bp.route('/framework/orchestrate', methods=['POST'])
def orchestrate_cycle_api():
    """API endpoint to run one orchestration cycle of the UMF."""
    if not unified_framework:
        return jsonify({"error": "Unified Metaphysical Framework not initialized"}), 500
    try:
        data = request.get_json() or {}
        n_qubits = data.get('n_qubits', 4) # Default value
        hologram_size_tuple = data.get('hologram_size', [64, 64]) # Default value, expect list/tuple
        # Ensure hologram_size is a tuple
        if not isinstance(hologram_size_tuple, (list, tuple)) or len(hologram_size_tuple) != 2:
             return jsonify({"error": "hologram_size must be a list or tuple of two integers"}), 400
        hologram_size = tuple(hologram_size_tuple) 
        
        input_seed = data.get('input_seed') # Optional seed for GAN

        result = unified_framework.orchestrate_cycle(n_qubits, hologram_size, input_seed)
        return jsonify(result), 200
    except ModelError as e:
        logger.error(f"API Orchestrate Error: {str(e)}")
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        logger.exception("API Orchestrate Unexpected Error") # Log full traceback for unexpected errors
        return jsonify({"error": f"An unexpected error occurred during orchestration: {str(e)}"}), 500

@api_bp.route('/framework/goal', methods=['POST'])
def set_framework_goal_api():
    """API endpoint to set the goal and operational mode of the UMF."""
    if not unified_framework:
        return jsonify({"error": "Unified Metaphysical Framework not initialized"}), 500
    try:
        data = request.get_json()
        goal = data.get('goal')
        mode = data.get('mode', 'Standard') # Default mode

        if not goal:
            return jsonify({"error": "Missing parameter: goal required"}), 400

        unified_framework.set_goal(goal, mode)
        return jsonify({"message": f"Framework goal set to '{goal}' in mode '{mode}'."}), 200
    except Exception as e:
        logger.error(f"API Set Goal Unexpected Error: {str(e)}")
        return jsonify({"error": "An unexpected error occurred while setting the framework goal."}), 500

@api_bp.route('/framework/status', methods=['GET'])
def get_framework_status_api():
    """API endpoint to get the overall status of the Unified Metaphysical Framework."""
    if not unified_framework:
        return jsonify({"error": "Unified Metaphysical Framework not initialized"}), 500
    try:
        status = unified_framework.get_framework_status()
        # Ensure the status is JSON serializable (the method should handle this)
        return jsonify(status), 200
    except Exception as e:
        logger.error(f"API Framework Status Unexpected Error: {str(e)}")
        return jsonify({"error": "An unexpected error occurred retrieving framework status."}), 500

@api_bp.route('/framework/reset', methods=['POST'])
def reset_framework_api():
    """API endpoint to reset the Unified Metaphysical Framework and all components."""
    if not unified_framework:
        return jsonify({"error": "Unified Metaphysical Framework not initialized"}), 500
    try:
        result = unified_framework.reset_framework()
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"API Framework Reset Unexpected Error: {str(e)}")
        return jsonify({"error": "An unexpected error occurred during framework reset."}), 500

# Ensure this blueprint is registered in the main app file 

import asyncio
import logging
import uvicorn
from pathlib import Path
from web.portal import app
from integration.hyperintelligent_system import HyperIntelligentSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("main")

async def initialize_system():
    """Initialize the hyperintelligent system."""
    logger.info("Initializing hyperintelligent system")
    
    # Create system instance
    system = HyperIntelligentSystem(
        name="HyperIntelligent",
        config={
            "agent_capabilities": ["analysis", "decision_making", "planning"],
            "assistant_capabilities": ["chat", "learning", "personalization", "context_awareness"],
            "body_capabilities": ["vision", "audio", "text", "movement", "speech", "gesture"],
            "agent_config": {
                "learning_rate": 0.1,
                "memory_size": 1000
            },
            "assistant_config": {
                "personality_traits": {
                    "professional": True,
                    "friendly": True,
                    "detailed": True
                }
            },
            "body_config": {
                "vision_resolution": (1920, 1080),
                "audio_sample_rate": 44100,
                "movement_speed": 1.0
            }
        }
    )
    
    # Activate core capabilities
    system.activate_capability("agent", "analysis")
    system.activate_capability("assistant", "chat")
    system.activate_capability("body", "text")
    
    return system

async def main():
    """Main entry point for the application."""
    try:
        # Initialize system
        system = await initialize_system()
        
        # Store system instance in FastAPI app state
        app.state.system = system
        
        # Start FastAPI server
        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info"
        )
        server = uvicorn.Server(config)
        
        logger.info("Starting hyperintelligent system")
        await server.serve()
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 
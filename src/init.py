import os
import sys
from pathlib import Path
from src.utils.logger import logger
from src.config import Config
from src.core.system_manager import system_manager
from src.utils.errors import handle_error
from src.api.routes import api_bp
from flask import Flask

def setup_environment():
    """Set up the environment for the system."""
    try:
        # Create necessary directories
        directories = [
            Config.MODEL_PATH,
            'static',
            'logs'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
        
        # Check for required environment variables
        required_vars = [
            'SECRET_KEY',
            'AUTH_TOKEN',
            'GOOGLE_APPLICATION_CREDENTIALS'
        ]
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        # Check for Google Cloud credentials
        credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        if not os.path.exists(credentials_path):
            raise FileNotFoundError(f"Google Cloud credentials not found at: {credentials_path}")
        
        logger.info("Environment setup completed successfully")
        return True
    except Exception as e:
        logger.error(f"Environment setup failed: {str(e)}")
        return False

def initialize_system():
    """Initialize the system and all its components."""
    try:
        # Test each component
        logger.info("Testing system components...")
        
        # Test consciousness expander
        test_input = [1.0, 2.0, 3.0]
        result = system_manager.expander.evolve(test_input)
        logger.info("Consciousness expander test successful")
        
        # Test NLP processor
        test_text = "Hello, world!"
        result = system_manager.nlp_processor.generate_text(test_text)
        logger.info("NLP processor test successful")
        
        # Test fractal generator
        result = system_manager.fractal_generator.generate_mandelbrot()
        logger.info("Fractal generator test successful")
        
        # Test radiation monitor
        result = system_manager.radiation_monitor.monitor_radiation()
        logger.info("Radiation monitor test successful")
        
        logger.info("System initialization completed successfully")
        return True
    except Exception as e:
        logger.error(f"System initialization failed: {str(e)}")
        return False

def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    
    # Register blueprints
    app.register_blueprint(api_bp, url_prefix='/api')
    
    # Configure logging
    logger.info("Application initialized")
    
    return app

if __name__ == "__main__":
    if setup_environment() and initialize_system():
        logger.info("System is ready to start")
        sys.exit(0)
    else:
        logger.error("System initialization failed")
        sys.exit(1)

    app = create_app()
    app.run(host='0.0.0.0', port=5000) 
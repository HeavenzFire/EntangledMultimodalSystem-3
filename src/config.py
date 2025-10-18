import os
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any, Union # Add common typing imports for potential future use
import pathlib # Often useful for path configurations

# Define the base directory of the project if needed for path construction
# Example: BASE_DIR = pathlib.Path(__file__).resolve().parent.parent

# Load environment variables
load_dotenv()

class Config:
    # Flask settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'default_secret_key')
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    
    # API settings
    AUTH_TOKEN = os.getenv('AUTH_TOKEN', 'default_auth_token')
    
    # Speech recognition settings
    SPEECH_SAMPLE_RATE = int(os.getenv('SPEECH_SAMPLE_RATE', '16000'))
    SPEECH_LANGUAGE = os.getenv('SPEECH_LANGUAGE', 'en-US')
    
    # Model settings
    MODEL_PATH = os.getenv('MODEL_PATH', 'models/')
    MAX_SEQUENCE_LENGTH = int(os.getenv('MAX_SEQUENCE_LENGTH', '150'))
    
    # Logging settings
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'app.log')
    
    # External API settings
    RADIATION_API_URL = os.getenv('RADIATION_API_URL', 'https://api.example.com/radiation')
    
    @staticmethod
    def init_app(app):
        # Initialize any app-specific configurations
        pass
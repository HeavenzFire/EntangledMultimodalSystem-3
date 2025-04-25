import pytest
import os
from src.config import Config

class TestConfig(Config):
    TESTING = True
    DEBUG = True
    LOG_LEVEL = 'DEBUG'
    LOG_FILE = 'test.log'
    MODEL_PATH = 'tests/models/'
    RADIATION_API_URL = 'http://test-api.example.com/radiation'

@pytest.fixture(scope='session', autouse=True)
def setup_test_environment():
    """Setup test environment before running tests."""
    # Create test directories
    os.makedirs(TestConfig.MODEL_PATH, exist_ok=True)
    
    # Setup test environment variables
    os.environ['TESTING'] = 'True'
    os.environ['DEBUG'] = 'True'
    
    yield
    
    # Cleanup after tests
    if os.path.exists(TestConfig.LOG_FILE):
        os.remove(TestConfig.LOG_FILE) 
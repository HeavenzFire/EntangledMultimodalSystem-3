from flask import Flask
from src.config import Config
from src.utils.logger import logger
from src.utils.errors import handle_error

def create_app(config_class=Config):
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Initialize app with config
    config_class.init_app(app)
    
    # Register blueprints
    from src.api.routes import api_bp
    app.register_blueprint(api_bp, url_prefix='/api')
    
    # Error handlers
    @app.errorhandler(Exception)
    def handle_exception(error):
        return handle_error(error, logger)
    
    # Health check endpoint
    @app.route('/health')
    def health_check():
        return {'status': 'healthy'}
    
    return app 
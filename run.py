from src import create_app
from src.config import Config
from src.utils.logger import logger

app = create_app(Config)

if __name__ == '__main__':
    try:
        logger.info("Starting Entangled Multimodal System...")
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=Config.DEBUG,
            use_reloader=False,
            threaded=True
        )
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}", exc_info=True)
        raise 

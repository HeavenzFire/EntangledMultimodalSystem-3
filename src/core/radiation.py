import requests
import numpy as np
from datetime import datetime
from src.utils.logger import logger
from src.utils.errors import APIError
from src.config import Config

class RadiationMonitor:
    def __init__(self):
        """Initialize the radiation monitor."""
        try:
            self.api_url = Config.RADIATION_API_URL
            logger.info("RadiationMonitor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RadiationMonitor: {str(e)}")
            raise APIError(f"Radiation monitor initialization failed: {str(e)}")

    def fetch_radiation_data(self):
        """Fetch radiation data from the external API."""
        try:
            response = requests.get(self.api_url, timeout=5)
            if response.status_code != 200:
                raise APIError(f"API request failed with status code: {response.status_code}")
            
            data = response.json()
            logger.info("Successfully fetched radiation data")
            return data
        except Exception as e:
            logger.error(f"Error in fetch_radiation_data method: {str(e)}")
            raise APIError(f"Failed to fetch radiation data: {str(e)}")

    def analyze_radiation_data(self, data):
        """Analyze the radiation data and provide insights."""
        try:
            if not data or not isinstance(data, dict):
                raise ValueError("Invalid radiation data format")
            
            # Extract relevant metrics
            radiation_levels = data.get('radiation_levels', [])
            timestamps = data.get('timestamps', [])
            
            if not radiation_levels or not timestamps:
                raise ValueError("Missing required data fields")
            
            # Calculate statistics
            mean_level = np.mean(radiation_levels)
            std_level = np.std(radiation_levels)
            max_level = np.max(radiation_levels)
            min_level = np.min(radiation_levels)
            
            # Determine safety status
            safety_threshold = 100  # Example threshold in microsieverts
            safety_status = "safe" if mean_level < safety_threshold else "warning" if mean_level < safety_threshold * 2 else "danger"
            
            analysis = {
                "mean_level": mean_level,
                "std_level": std_level,
                "max_level": max_level,
                "min_level": min_level,
                "safety_status": safety_status,
                "timestamp": datetime.now().isoformat(),
                "data_points": len(radiation_levels)
            }
            
            logger.info(f"Successfully analyzed radiation data. Safety status: {safety_status}")
            return analysis
        except Exception as e:
            logger.error(f"Error in analyze_radiation_data method: {str(e)}")
            raise APIError(f"Radiation data analysis failed: {str(e)}")

    def monitor_radiation(self):
        """Monitor radiation levels and provide real-time analysis."""
        try:
            data = self.fetch_radiation_data()
            analysis = self.analyze_radiation_data(data)
            return {
                "raw_data": data,
                "analysis": analysis
            }
        except Exception as e:
            logger.error(f"Error in monitor_radiation method: {str(e)}")
            raise APIError(f"Radiation monitoring failed: {str(e)}")

# Create a global instance
radiation_monitor = RadiationMonitor()

def external_radiation_monitor():
    """Global function to monitor radiation using the radiation monitor."""
    return radiation_monitor.monitor_radiation() 
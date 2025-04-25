import os
import base64
import requests
from typing import Dict, Any, Optional, Union
from src.utils.errors import ModelError
from src.utils.logger import logger
from dotenv import load_dotenv

class GeminiIntegration:
    """Gemini API Integration for advanced multimodal capabilities."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Gemini Integration.
        
        Args:
            api_key: Optional Gemini API key. If not provided, will try to get from environment.
        """
        try:
            # Load environment variables
            load_dotenv()
            
            # Get API key from parameter or environment
            self.api_key = api_key or os.getenv("GEMINI_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "Gemini API key not found. Please provide it or set GEMINI_API_KEY environment variable. "
                    "See .env.example for guidance."
                )
            
            # Validate API key format
            if not self._validate_api_key(self.api_key):
                raise ValueError("Invalid Gemini API key format")
            
            # Initialize endpoints
            self.text_endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.0-pro-latest:generateContent?key={self.api_key}"
            self.vision_endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.0-pro-vision-latest:generateContent?key={self.api_key}"
            
            # Initialize state
            self.state = {
                "status": "active",
                "last_request": None,
                "request_count": 0,
                "error_count": 0
            }
            
            logger.info("GeminiIntegration initialized")
            
        except Exception as e:
            logger.error(f"Error initializing GeminiIntegration: {str(e)}")
            raise ModelError(f"Failed to initialize GeminiIntegration: {str(e)}")

    def _validate_api_key(self, api_key: str) -> bool:
        """Validate the format of the API key.
        
        Args:
            api_key: The API key to validate
            
        Returns:
            bool: True if the key format is valid
        """
        # Basic validation - check if it starts with 'AIza' and has correct length
        return api_key.startswith('AIza') and len(api_key) > 30

    def generate_text(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate text using Gemini API.
        
        Args:
            prompt: The text prompt to generate from
            temperature: Controls randomness in generation (0.0 to 1.0)
            
        Returns:
            Generated text response
        """
        try:
            payload = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {"temperature": temperature}
            }
            
            response = requests.post(self.text_endpoint, json=payload)
            response.raise_for_status()
            data = response.json()
            
            # Update state
            self.state["last_request"] = "text"
            self.state["request_count"] += 1
            
            return data["candidates"][0]["content"]["parts"][0]["text"]
            
        except Exception as e:
            self.state["error_count"] += 1
            logger.error(f"Error generating text: {str(e)}")
            raise ModelError(f"Text generation failed: {str(e)}")

    def generate_multimodal(self, prompt: str, image_path: str) -> str:
        """Generate multimodal response using Gemini Vision API.
        
        Args:
            prompt: The text prompt to generate from
            image_path: Path to the image file
            
        Returns:
            Generated multimodal response
        """
        try:
            with open(image_path, "rb") as img_file:
                image_b64 = base64.b64encode(img_file.read()).decode("utf-8")
                
            payload = {
                "contents": [{
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {
                            "mime_type": "image/jpeg",
                            "data": image_b64
                        }}
                    ]
                }]
            }
            
            response = requests.post(self.vision_endpoint, json=payload)
            response.raise_for_status()
            data = response.json()
            
            # Update state
            self.state["last_request"] = "multimodal"
            self.state["request_count"] += 1
            
            return data["candidates"][0]["content"]["parts"][0]["text"]
            
        except Exception as e:
            self.state["error_count"] += 1
            logger.error(f"Error generating multimodal response: {str(e)}")
            raise ModelError(f"Multimodal generation failed: {str(e)}")

    def get_state(self) -> Dict[str, Any]:
        """Get current integration state."""
        return self.state

    def reset(self) -> None:
        """Reset integration state."""
        self.state.update({
            "status": "active",
            "last_request": None,
            "request_count": 0,
            "error_count": 0
        }) 
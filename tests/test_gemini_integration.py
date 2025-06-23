import os
import pytest
from unittest.mock import patch, MagicMock
from src.core.gemini_integration import GeminiIntegration
from src.utils.errors import ModelError

class TestGeminiIntegration:
    def test_initialization(self):
        """Test successful initialization with valid API key."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaSyDyVazOUDgSt8n45gi9NedFQAnEQpSxQRc"}):
            integration = GeminiIntegration()
            assert integration.api_key == "AIzaSyDyVazOUDgSt8n45gi9NedFQAnEQpSxQRc"
            assert integration.state["status"] == "active"
            assert integration.state["request_count"] == 0
            assert integration.state["error_count"] == 0

    def test_initialization_no_api_key(self):
        """Test initialization without API key."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                GeminiIntegration()
            assert "Gemini API key not found" in str(exc_info.value)

    def test_initialization_invalid_api_key(self):
        """Test initialization with invalid API key format."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "invalid_key"}):
            with pytest.raises(ValueError) as exc_info:
                GeminiIntegration()
            assert "Invalid Gemini API key format" in str(exc_info.value)

    def test_generate_text(self):
        """Test text generation."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "candidates": [{
                "content": {
                    "parts": [{"text": "Generated text response"}]
                }
            }]
        }
        mock_response.raise_for_status.return_value = None

        with patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaSyDyVazOUDgSt8n45gi9NedFQAnEQpSxQRc"}):
            with patch("requests.post", return_value=mock_response):
                integration = GeminiIntegration()
                response = integration.generate_text("Test prompt")
                assert response == "Generated text response"
                assert integration.state["last_request"] == "text"
                assert integration.state["request_count"] == 1

    def test_generate_multimodal(self):
        """Test multimodal generation."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "candidates": [{
                "content": {
                    "parts": [{"text": "Generated multimodal response"}]
                }
            }]
        }
        mock_response.raise_for_status.return_value = None

        with patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaSyDyVazOUDgSt8n45gi9NedFQAnEQpSxQRc"}):
            with patch("requests.post", return_value=mock_response):
                with patch("builtins.open", MagicMock()):
                    integration = GeminiIntegration()
                    response = integration.generate_multimodal("Test prompt", "test_image.jpg")
                    assert response == "Generated multimodal response"
                    assert integration.state["last_request"] == "multimodal"
                    assert integration.state["request_count"] == 1

    def test_reset(self):
        """Test reset functionality."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaSyDyVazOUDgSt8n45gi9NedFQAnEQpSxQRc"}):
            integration = GeminiIntegration()
            integration.state.update({
                "last_request": "text",
                "request_count": 5,
                "error_count": 2
            })
            integration.reset()
            assert integration.state["last_request"] is None
            assert integration.state["request_count"] == 0
            assert integration.state["error_count"] == 0 
# src/integration/n8n_client.py
import requests
import logging
import os
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# --- Configuration ---
# It's highly recommended to load these from environment variables or a secure config
# Example using environment variables:
N8N_BASE_URL = os.getenv("N8N_BASE_URL", "http://localhost:5678") # Default n8n URL
# If using API Key authentication (optional, webhooks are often easier)
N8N_API_KEY = os.getenv("N8N_API_KEY")
# ---------------------

def trigger_n8n_webhook(webhook_url: str, data: Optional[Dict[str, Any]] = None, method: str = 'POST') -> bool:
    """
    Triggers a specific n8n workflow via its webhook URL.

    Args:
        webhook_url: The full webhook URL obtained from the n8n workflow trigger node.
                     Should include the base URL, e.g., http://localhost:5678/webhook/your-path
        data: A dictionary containing the JSON payload to send to the workflow (optional).
        method: The HTTP method expected by the webhook ('POST' or 'GET').

    Returns:
        True if the request was successful (e.g., status code 2xx), False otherwise.
    """
    if not webhook_url.startswith(('http://', 'https://')):
        logger.error(f"Invalid webhook URL provided: {webhook_url}. It must be a full URL.")
        return False

    try:
        headers = {'Content-Type': 'application/json'}
        # Add API key header if provided (for API endpoint calls, not usually needed for webhooks)
        # if N8N_API_KEY:
        #     headers['X-N8N-API-KEY'] = N8N_API_KEY

        if method.upper() == 'POST':
            response = requests.post(webhook_url, json=data, headers=headers, timeout=15) # 15 second timeout
        elif method.upper() == 'GET':
            # For GET requests, data is typically sent as query parameters
            response = requests.get(webhook_url, params=data, headers=headers, timeout=15)
        else:
            logger.error(f"Unsupported HTTP method for n8n webhook: {method}")
            return False

        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        logger.info(f"Successfully triggered n8n webhook {webhook_url} (status: {response.status_code})")
        # Optionally return response content if needed
        # return response.json() # Or response.text
        return True

    except requests.exceptions.Timeout:
        logger.error(f"Timeout error triggering n8n webhook {webhook_url}")
        return False
    except requests.exceptions.ConnectionError:
        logger.error(f"Connection error triggering n8n webhook {webhook_url}. Is n8n running and accessible at the specified URL?")
        return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Error triggering n8n webhook {webhook_url}: {e}", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred while triggering n8n webhook {webhook_url}: {e}", exc_info=True)
        return False

# --- Example Usage (call this from your main framework) ---
# if __name__ == '__main__':
#     logging.basicConfig(level=logging.INFO)
#     # IMPORTANT: Replace with your actual n8n workflow webhook URL
#     # You get this from the 'Webhook' trigger node in your n8n workflow
#     test_webhook_url = f"{N8N_BASE_URL}/webhook/YOUR_UNIQUE_WORKFLOW_PATH_HERE"
#
#     if not test_webhook_url.endswith("YOUR_UNIQUE_WORKFLOW_PATH_HERE"): # Basic check
#         payload = {"message": "Hello from Python!", "value": 123, "source": "EntangledMultimodalSystem"}
#
#         print(f"Attempting to trigger n8n webhook at: {test_webhook_url}")
#         if trigger_n8n_webhook(test_webhook_url, data=payload):
#             print("n8n workflow triggered successfully via POST.")
#         else:
#             print("Failed to trigger n8n workflow via POST.")
#     else:
#         print("Please replace 'YOUR_UNIQUE_WORKFLOW_PATH_HERE' in the example usage with your actual n8n webhook path.")


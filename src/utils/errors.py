class EntangledSystemError(Exception):
    """Base exception class for the Entangled System."""
    pass

class ModelError(EntangledSystemError):
    """Raised when there's an error with the model operations."""
    pass

class SpeechRecognitionError(EntangledSystemError):
    """Raised when there's an error with speech recognition."""
    pass

class APIError(EntangledSystemError):
    """Raised when there's an error with API operations."""
    pass

class ConfigurationError(EntangledSystemError):
    """Raised when there's an error with system configuration."""
    pass

class ValidationError(EntangledSystemError):
    """Raised when input validation fails."""
    pass

class QuantumError(EntangledSystemError):
    """Raised when there's an error with quantum operations."""
    pass

def handle_error(error, logger):
    """Handle errors and log them appropriately."""
    if isinstance(error, EntangledSystemError):
        logger.error(f"System error: {str(error)}")
    else:
        logger.error(f"Unexpected error: {str(error)}", exc_info=True)
    
    # Return appropriate error response
    return {
        "error": str(error),
        "type": error.__class__.__name__
    } 
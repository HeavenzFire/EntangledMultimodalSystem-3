from typing import Dict, List, Any
import os
from pathlib import Path

class SecurityConfig:
    """Security configuration and policy management."""
    
    # Security levels
    SECURITY_LEVELS = {
        "low": {
            "min_password_length": 8,
            "require_special_chars": False,
            "require_numbers": False,
            "require_upper_lower": False,
            "max_login_attempts": 5,
            "session_timeout": 3600,  # 1 hour
        },
        "medium": {
            "min_password_length": 12,
            "require_special_chars": True,
            "require_numbers": True,
            "require_upper_lower": True,
            "max_login_attempts": 3,
            "session_timeout": 1800,  # 30 minutes
        },
        "high": {
            "min_password_length": 16,
            "require_special_chars": True,
            "require_numbers": True,
            "require_upper_lower": True,
            "max_login_attempts": 2,
            "session_timeout": 900,  # 15 minutes
        }
    }
    
    # Default security settings
    DEFAULT_SECURITY_LEVEL = "medium"
    
    # File paths
    SECURITY_DIR = Path("security")
    LOG_DIR = SECURITY_DIR / "logs"
    KEY_DIR = SECURITY_DIR / "keys"
    
    # Encryption settings
    ENCRYPTION_ALGORITHM = "AES-256-CBC"
    KEY_LENGTH = 32  # bytes
    IV_LENGTH = 16  # bytes
    HASH_ALGORITHM = "SHA3-256"
    
    # Session settings
    SESSION_COOKIE_NAME = "quantum_session"
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Strict"
    
    # Rate limiting
    RATE_LIMIT_REQUESTS = 100
    RATE_LIMIT_PERIOD = 60  # seconds
    
    # Audit logging
    AUDIT_LOG_ENABLED = True
    AUDIT_LOG_LEVEL = "INFO"
    AUDIT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Security headers
    SECURITY_HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Content-Security-Policy": "default-src 'self'",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
    }
    
    @classmethod
    def initialize(cls) -> None:
        """Initialize security directories and files."""
        try:
            # Create security directories
            cls.SECURITY_DIR.mkdir(exist_ok=True)
            cls.LOG_DIR.mkdir(exist_ok=True)
            cls.KEY_DIR.mkdir(exist_ok=True)
            
            # Set directory permissions
            cls.SECURITY_DIR.chmod(0o700)
            cls.LOG_DIR.chmod(0o700)
            cls.KEY_DIR.chmod(0o700)
            
            # Create .gitignore in security directory
            gitignore_path = cls.SECURITY_DIR / ".gitignore"
            if not gitignore_path.exists():
                with open(gitignore_path, "w") as f:
                    f.write("*\n!.gitignore\n")
                    
        except Exception as e:
            raise SecurityConfigError(f"Failed to initialize security configuration: {str(e)}")
            
    @classmethod
    def get_security_level(cls, level: str = None) -> Dict[str, Any]:
        """Get security settings for specified level."""
        level = level or cls.DEFAULT_SECURITY_LEVEL
        if level not in cls.SECURITY_LEVELS:
            raise SecurityConfigError(f"Invalid security level: {level}")
        return cls.SECURITY_LEVELS[level]
        
    @classmethod
    def validate_security_settings(cls, settings: Dict[str, Any]) -> bool:
        """Validate security settings."""
        required_keys = [
            "min_password_length",
            "require_special_chars",
            "require_numbers",
            "require_upper_lower",
            "max_login_attempts",
            "session_timeout"
        ]
        
        return all(key in settings for key in required_keys)

class SecurityConfigError(Exception):
    """Exception raised for security configuration errors."""
    pass 
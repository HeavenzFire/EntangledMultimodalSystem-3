from typing import Callable, Dict, Any
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import time
import logging
from .security_config import SecurityConfig

class SecurityMiddleware(BaseHTTPMiddleware):
    """Middleware for enforcing security policies."""
    
    def __init__(
        self,
        app: ASGIApp,
        security_config: SecurityConfig = None
    ):
        super().__init__(app)
        self.security_config = security_config or SecurityConfig()
        self.logger = logging.getLogger("SecurityMiddleware")
        self.rate_limit = {}
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and enforce security policies."""
        try:
            # Apply security headers
            response = await call_next(request)
            self._add_security_headers(response)
            
            # Rate limiting
            client_ip = request.client.host
            if not self._check_rate_limit(client_ip):
                return Response(
                    content="Too many requests",
                    status_code=429
                )
                
            # Log security events
            self._log_security_event(request, response)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Security middleware error: {str(e)}")
            return Response(
                content="Internal server error",
                status_code=500
            )
            
    def _add_security_headers(self, response: Response) -> None:
        """Add security headers to response."""
        for header, value in self.security_config.SECURITY_HEADERS.items():
            response.headers[header] = value
            
    def _check_rate_limit(self, client_ip: str) -> bool:
        """Check if client has exceeded rate limit."""
        current_time = time.time()
        
        # Clean up old entries
        self.rate_limit = {
            ip: times for ip, times in self.rate_limit.items()
            if current_time - times[-1] < self.security_config.RATE_LIMIT_PERIOD
        }
        
        # Add new request time
        if client_ip not in self.rate_limit:
            self.rate_limit[client_ip] = []
        self.rate_limit[client_ip].append(current_time)
        
        # Check rate limit
        request_count = len([
            t for t in self.rate_limit[client_ip]
            if current_time - t < self.security_config.RATE_LIMIT_PERIOD
        ])
        
        return request_count <= self.security_config.RATE_LIMIT_REQUESTS
        
    def _log_security_event(self, request: Request, response: Response) -> None:
        """Log security-related events."""
        if not self.security_config.AUDIT_LOG_ENABLED:
            return
            
        log_data = {
            "timestamp": time.time(),
            "client_ip": request.client.host,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "user_agent": request.headers.get("user-agent"),
            "referer": request.headers.get("referer")
        }
        
        self.logger.info(f"Security event: {log_data}")
        
class SessionMiddleware(BaseHTTPMiddleware):
    """Middleware for managing secure sessions."""
    
    def __init__(
        self,
        app: ASGIApp,
        security_config: SecurityConfig = None
    ):
        super().__init__(app)
        self.security_config = security_config or SecurityConfig()
        self.logger = logging.getLogger("SessionMiddleware")
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and manage session security."""
        try:
            # Check session cookie
            session_cookie = request.cookies.get(
                self.security_config.SESSION_COOKIE_NAME
            )
            
            if session_cookie:
                # Validate session
                if not self._validate_session(session_cookie):
                    return Response(
                        content="Invalid session",
                        status_code=401
                    )
                    
            # Process request
            response = await call_next(request)
            
            # Set secure session cookie
            self._set_session_cookie(response)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Session middleware error: {str(e)}")
            return Response(
                content="Internal server error",
                status_code=500
            )
            
    def _validate_session(self, session_cookie: str) -> bool:
        """Validate session cookie."""
        # Implement session validation logic
        # This is a placeholder for actual session validation
        return True
        
    def _set_session_cookie(self, response: Response) -> None:
        """Set secure session cookie."""
        response.set_cookie(
            key=self.security_config.SESSION_COOKIE_NAME,
            value="session_token",  # Replace with actual session token
            secure=self.security_config.SESSION_COOKIE_SECURE,
            httponly=self.security_config.SESSION_COOKIE_HTTPONLY,
            samesite=self.security_config.SESSION_COOKIE_SAMESITE,
            max_age=self.security_config.get_security_level()["session_timeout"]
        ) 
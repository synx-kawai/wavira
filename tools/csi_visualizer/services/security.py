#!/usr/bin/env python3
"""
Security utilities for Wavira API services.

Issue #40: セキュリティ強化
- API key authentication
- Rate limiting
- Input validation
- Security headers
"""

import hashlib
import hmac
import os
import re
import secrets
import time
from collections import defaultdict
from dataclasses import dataclass
from functools import wraps
from typing import Callable, Dict, List, Optional, Set

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class SecurityConfig:
    """Security configuration."""
    # API Key settings
    api_keys: List[str] = None  # List of valid API keys
    require_api_key: bool = False  # Whether to require API key for protected endpoints

    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100  # Max requests per window
    rate_limit_window: int = 60  # Window in seconds

    # CORS
    cors_origins: List[str] = None  # Allowed origins, None = allow all
    cors_allow_credentials: bool = True

    # Security headers
    add_security_headers: bool = True

    def __post_init__(self):
        if self.api_keys is None:
            self.api_keys = []
        if self.cors_origins is None:
            self.cors_origins = ["*"]


def load_security_config() -> SecurityConfig:
    """Load security configuration from environment variables."""
    api_keys_str = os.environ.get("API_KEYS", "")
    api_keys = [k.strip() for k in api_keys_str.split(",") if k.strip()]

    cors_origins_str = os.environ.get("CORS_ORIGINS", "*")
    if cors_origins_str == "*":
        cors_origins = ["*"]
    else:
        cors_origins = [o.strip() for o in cors_origins_str.split(",") if o.strip()]

    return SecurityConfig(
        api_keys=api_keys,
        require_api_key=os.environ.get("REQUIRE_API_KEY", "false").lower() == "true",
        rate_limit_enabled=os.environ.get("RATE_LIMIT_ENABLED", "true").lower() == "true",
        rate_limit_requests=int(os.environ.get("RATE_LIMIT_REQUESTS", "100")),
        rate_limit_window=int(os.environ.get("RATE_LIMIT_WINDOW", "60")),
        cors_origins=cors_origins,
        cors_allow_credentials=os.environ.get("CORS_ALLOW_CREDENTIALS", "true").lower() == "true",
        add_security_headers=os.environ.get("ADD_SECURITY_HEADERS", "true").lower() == "true",
    )


# =============================================================================
# API Key Authentication
# =============================================================================


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def generate_api_key(prefix: str = "wvr") -> str:
    """Generate a secure API key."""
    random_bytes = secrets.token_bytes(24)
    key = secrets.token_urlsafe(24)
    return f"{prefix}_{key}"


def hash_api_key(api_key: str) -> str:
    """Hash an API key for storage."""
    return hashlib.sha256(api_key.encode()).hexdigest()


def verify_api_key(api_key: str, hashed_keys: List[str]) -> bool:
    """Verify an API key against a list of hashed keys."""
    key_hash = hash_api_key(api_key)
    return any(hmac.compare_digest(key_hash, h) for h in hashed_keys)


class APIKeyAuth:
    """API Key authentication dependency."""

    def __init__(self, config: SecurityConfig):
        self.config = config
        # Pre-hash the API keys for constant-time comparison
        self.hashed_keys = [hash_api_key(k) for k in config.api_keys]

    async def __call__(
        self,
        request: Request,
        api_key: Optional[str] = Depends(api_key_header),
    ) -> Optional[str]:
        """Validate API key if required."""
        if not self.config.require_api_key:
            return api_key

        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing API key",
                headers={"WWW-Authenticate": "ApiKey"},
            )

        if not verify_api_key(api_key, self.hashed_keys):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "ApiKey"},
            )

        return api_key


# =============================================================================
# Rate Limiting
# =============================================================================


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, requests: int = 100, window: int = 60):
        self.requests = requests
        self.window = window
        self.clients: Dict[str, List[float]] = defaultdict(list)
        self._last_cleanup = time.time()

    def _get_client_id(self, request: Request) -> str:
        """Get client identifier from request."""
        # Use X-Forwarded-For if behind proxy, otherwise use client host
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _cleanup(self):
        """Remove expired entries."""
        now = time.time()
        if now - self._last_cleanup < 60:  # Cleanup every minute
            return

        cutoff = now - self.window
        for client_id in list(self.clients.keys()):
            self.clients[client_id] = [
                ts for ts in self.clients[client_id] if ts > cutoff
            ]
            if not self.clients[client_id]:
                del self.clients[client_id]

        self._last_cleanup = now

    def is_allowed(self, request: Request) -> tuple[bool, Dict[str, str]]:
        """Check if request is allowed and return rate limit headers."""
        self._cleanup()

        client_id = self._get_client_id(request)
        now = time.time()
        cutoff = now - self.window

        # Remove old requests
        self.clients[client_id] = [
            ts for ts in self.clients[client_id] if ts > cutoff
        ]

        # Check limit
        current = len(self.clients[client_id])
        remaining = max(0, self.requests - current)

        headers = {
            "X-RateLimit-Limit": str(self.requests),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": str(int(now + self.window)),
        }

        if current >= self.requests:
            return False, headers

        # Record request
        self.clients[client_id].append(now)
        headers["X-RateLimit-Remaining"] = str(remaining - 1)

        return True, headers


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware."""

    def __init__(self, app, config: SecurityConfig):
        super().__init__(app)
        self.config = config
        self.limiter = RateLimiter(
            requests=config.rate_limit_requests,
            window=config.rate_limit_window,
        )
        # Paths that are exempt from rate limiting
        self.exempt_paths: Set[str] = {"/api/v1/health", "/health", "/"}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not self.config.rate_limit_enabled:
            return await call_next(request)

        # Skip rate limiting for exempt paths
        if request.url.path in self.exempt_paths:
            return await call_next(request)

        allowed, headers = self.limiter.is_allowed(request)

        if not allowed:
            response = Response(
                content='{"detail": "Rate limit exceeded"}',
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                media_type="application/json",
            )
            for key, value in headers.items():
                response.headers[key] = value
            return response

        response = await call_next(request)
        for key, value in headers.items():
            response.headers[key] = value

        return response


# =============================================================================
# Security Headers
# =============================================================================


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to responses."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)

        # Prevent MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"

        # XSS protection
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # Prevent clickjacking
        response.headers["X-Frame-Options"] = "DENY"

        # Referrer policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Content Security Policy (basic)
        response.headers["Content-Security-Policy"] = "default-src 'self'"

        # Cache control for API responses
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store, max-age=0"

        return response


# =============================================================================
# Input Validation
# =============================================================================


# Device ID pattern: alphanumeric, hyphens, underscores, max 64 chars
DEVICE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

# MAC address pattern
MAC_PATTERN = re.compile(r"^([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}$")


def validate_device_id(device_id: str) -> bool:
    """Validate device ID format."""
    return bool(DEVICE_ID_PATTERN.match(device_id))


def validate_mac_address(mac: str) -> bool:
    """Validate MAC address format."""
    return bool(MAC_PATTERN.match(mac))


def sanitize_string(value: str, max_length: int = 256) -> str:
    """Sanitize a string value."""
    if not isinstance(value, str):
        return ""
    # Truncate
    value = value[:max_length]
    # Remove null bytes
    value = value.replace("\x00", "")
    return value


class InputValidator:
    """Input validation dependency."""

    @staticmethod
    def device_id(device_id: str) -> str:
        """Validate and return device ID."""
        device_id = sanitize_string(device_id, max_length=64)
        if not validate_device_id(device_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid device ID format. Use alphanumeric, hyphens, underscores only.",
            )
        return device_id


# =============================================================================
# MQTT Security
# =============================================================================


def get_mqtt_credentials() -> Dict[str, str]:
    """Get MQTT credentials from environment variables."""
    return {
        "username": os.environ.get("MQTT_USERNAME", ""),
        "password": os.environ.get("MQTT_PASSWORD", ""),
    }


def mask_password(password: str) -> str:
    """Mask password for logging."""
    if not password:
        return "(not set)"
    if len(password) <= 4:
        return "*" * len(password)
    return password[:2] + "*" * (len(password) - 4) + password[-2:]

#!/usr/bin/env python3
"""
Unit tests for security module.
Issue #40: セキュリティ強化
"""

import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add the tools/csi_visualizer/services directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "csi_visualizer" / "services"))

from security import (
    SecurityConfig,
    load_security_config,
    generate_api_key,
    hash_api_key,
    verify_api_key,
    RateLimiter,
    validate_device_id,
    validate_mac_address,
    sanitize_string,
    InputValidator,
    mask_password,
)


# =============================================================================
# SecurityConfig Tests
# =============================================================================


class TestSecurityConfig:
    """Tests for SecurityConfig."""

    def test_default_config(self):
        """Default configuration values."""
        config = SecurityConfig()
        assert config.api_keys == []
        assert config.require_api_key is False
        assert config.rate_limit_enabled is True
        assert config.rate_limit_requests == 100
        assert config.rate_limit_window == 60
        assert config.cors_origins == ["*"]
        assert config.add_security_headers is True

    def test_custom_config(self):
        """Custom configuration values."""
        config = SecurityConfig(
            api_keys=["key1", "key2"],
            require_api_key=True,
            rate_limit_requests=50,
            cors_origins=["https://example.com"],
        )
        assert config.api_keys == ["key1", "key2"]
        assert config.require_api_key is True
        assert config.rate_limit_requests == 50
        assert config.cors_origins == ["https://example.com"]

    def test_load_from_env(self):
        """Load configuration from environment variables."""
        with patch.dict(os.environ, {
            "API_KEYS": "key1,key2,key3",
            "REQUIRE_API_KEY": "true",
            "RATE_LIMIT_ENABLED": "false",
            "RATE_LIMIT_REQUESTS": "200",
            "CORS_ORIGINS": "https://a.com,https://b.com",
        }):
            config = load_security_config()
            assert config.api_keys == ["key1", "key2", "key3"]
            assert config.require_api_key is True
            assert config.rate_limit_enabled is False
            assert config.rate_limit_requests == 200
            assert config.cors_origins == ["https://a.com", "https://b.com"]

    def test_load_from_env_defaults(self):
        """Load configuration with default values."""
        with patch.dict(os.environ, {}, clear=True):
            config = load_security_config()
            assert config.api_keys == []
            assert config.require_api_key is False
            assert config.rate_limit_enabled is True


# =============================================================================
# API Key Tests
# =============================================================================


class TestAPIKey:
    """Tests for API key functionality."""

    def test_generate_api_key(self):
        """Generate API key with prefix."""
        key = generate_api_key("wvr")
        assert key.startswith("wvr_")
        assert len(key) > 10

    def test_generate_api_key_custom_prefix(self):
        """Generate API key with custom prefix."""
        key = generate_api_key("test")
        assert key.startswith("test_")

    def test_generate_unique_keys(self):
        """Generated keys are unique."""
        keys = [generate_api_key() for _ in range(100)]
        assert len(set(keys)) == 100

    def test_hash_api_key(self):
        """Hash API key produces consistent results."""
        key = "test_key_123"
        hash1 = hash_api_key(key)
        hash2 = hash_api_key(key)
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 produces 64 hex chars

    def test_hash_different_keys(self):
        """Different keys produce different hashes."""
        hash1 = hash_api_key("key1")
        hash2 = hash_api_key("key2")
        assert hash1 != hash2

    def test_verify_api_key_valid(self):
        """Verify valid API key."""
        key = "my_secret_key"
        hashed = hash_api_key(key)
        assert verify_api_key(key, [hashed]) is True

    def test_verify_api_key_invalid(self):
        """Verify invalid API key."""
        key = "wrong_key"
        hashed = hash_api_key("correct_key")
        assert verify_api_key(key, [hashed]) is False

    def test_verify_api_key_multiple_hashes(self):
        """Verify API key against multiple hashes."""
        key = "key2"
        hashes = [hash_api_key("key1"), hash_api_key("key2"), hash_api_key("key3")]
        assert verify_api_key(key, hashes) is True


# =============================================================================
# Rate Limiter Tests
# =============================================================================


class TestRateLimiter:
    """Tests for rate limiter."""

    def test_allows_under_limit(self):
        """Requests under limit are allowed."""
        limiter = RateLimiter(requests=5, window=60)
        mock_request = MagicMock()
        mock_request.headers = {}
        mock_request.client.host = "127.0.0.1"

        for _ in range(5):
            allowed, headers = limiter.is_allowed(mock_request)
            assert allowed is True
            assert "X-RateLimit-Limit" in headers
            assert "X-RateLimit-Remaining" in headers

    def test_blocks_over_limit(self):
        """Requests over limit are blocked."""
        limiter = RateLimiter(requests=3, window=60)
        mock_request = MagicMock()
        mock_request.headers = {}
        mock_request.client.host = "127.0.0.1"

        for _ in range(3):
            limiter.is_allowed(mock_request)

        allowed, headers = limiter.is_allowed(mock_request)
        assert allowed is False
        assert headers["X-RateLimit-Remaining"] == "0"

    def test_per_client_limiting(self):
        """Each client has independent limit."""
        limiter = RateLimiter(requests=2, window=60)

        request1 = MagicMock()
        request1.headers = {}
        request1.client.host = "192.168.1.1"

        request2 = MagicMock()
        request2.headers = {}
        request2.client.host = "192.168.1.2"

        # Exhaust limit for client 1
        limiter.is_allowed(request1)
        limiter.is_allowed(request1)
        allowed1, _ = limiter.is_allowed(request1)

        # Client 2 should still be allowed
        allowed2, _ = limiter.is_allowed(request2)

        assert allowed1 is False
        assert allowed2 is True

    def test_x_forwarded_for(self):
        """Uses X-Forwarded-For header when present."""
        limiter = RateLimiter(requests=1, window=60)

        request = MagicMock()
        request.headers = {"X-Forwarded-For": "10.0.0.1, 10.0.0.2"}
        request.client.host = "127.0.0.1"

        limiter.is_allowed(request)
        # Second request from same forwarded IP should be blocked
        allowed, _ = limiter.is_allowed(request)
        assert allowed is False

    def test_rate_limit_headers(self):
        """Rate limit headers are correct."""
        limiter = RateLimiter(requests=10, window=60)
        mock_request = MagicMock()
        mock_request.headers = {}
        mock_request.client.host = "127.0.0.1"

        allowed, headers = limiter.is_allowed(mock_request)

        assert headers["X-RateLimit-Limit"] == "10"
        assert int(headers["X-RateLimit-Remaining"]) == 9
        assert int(headers["X-RateLimit-Reset"]) > time.time()


# =============================================================================
# Input Validation Tests
# =============================================================================


class TestInputValidation:
    """Tests for input validation."""

    def test_valid_device_ids(self):
        """Valid device IDs pass validation."""
        valid_ids = [
            "esp32-001",
            "device_123",
            "MyDevice",
            "a",
            "a-b_c-d",
            "A1B2C3",
        ]
        for device_id in valid_ids:
            assert validate_device_id(device_id) is True, f"Should be valid: {device_id}"

    def test_invalid_device_ids(self):
        """Invalid device IDs fail validation."""
        invalid_ids = [
            "",
            "device with spaces",
            "device@special",
            "../path/traversal",
            "a" * 65,  # Too long
            "device\x00null",
        ]
        for device_id in invalid_ids:
            assert validate_device_id(device_id) is False, f"Should be invalid: {device_id}"

    def test_valid_mac_addresses(self):
        """Valid MAC addresses pass validation."""
        valid_macs = [
            "AA:BB:CC:DD:EE:FF",
            "aa:bb:cc:dd:ee:ff",
            "00:11:22:33:44:55",
        ]
        for mac in valid_macs:
            assert validate_mac_address(mac) is True, f"Should be valid: {mac}"

    def test_invalid_mac_addresses(self):
        """Invalid MAC addresses fail validation."""
        invalid_macs = [
            "invalid",
            "AA-BB-CC-DD-EE-FF",  # Wrong separator
            "AA:BB:CC:DD:EE",  # Too short
            "AA:BB:CC:DD:EE:FF:GG",  # Too long
            "GG:HH:II:JJ:KK:LL",  # Invalid hex
        ]
        for mac in invalid_macs:
            assert validate_mac_address(mac) is False, f"Should be invalid: {mac}"

    def test_sanitize_string(self):
        """String sanitization works correctly."""
        # Truncates long strings
        assert len(sanitize_string("a" * 300, max_length=100)) == 100

        # Removes null bytes
        assert sanitize_string("hello\x00world") == "helloworld"

        # Handles empty strings
        assert sanitize_string("") == ""

        # Handles non-strings
        assert sanitize_string(123) == ""

    def test_input_validator_device_id(self):
        """InputValidator.device_id works correctly."""
        # Valid ID
        result = InputValidator.device_id("esp32-001")
        assert result == "esp32-001"

        # Invalid ID raises HTTPException
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            InputValidator.device_id("../invalid")
        assert exc_info.value.status_code == 400


# =============================================================================
# Utility Tests
# =============================================================================


class TestUtilities:
    """Tests for utility functions."""

    def test_mask_password_empty(self):
        """Empty password masking."""
        assert mask_password("") == "(not set)"

    def test_mask_password_short(self):
        """Short password masking."""
        assert mask_password("abc") == "***"
        assert mask_password("abcd") == "****"

    def test_mask_password_normal(self):
        """Normal password masking."""
        result = mask_password("mypassword")
        assert result.startswith("my")
        assert result.endswith("rd")
        assert "*" in result

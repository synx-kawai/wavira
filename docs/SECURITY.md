# Wavira Security Guide

This document describes security features and best practices for deploying Wavira in production environments.

## Overview

Wavira implements multiple layers of security:
- API authentication (API keys)
- Rate limiting
- Input validation
- MQTT authentication and ACL
- Security headers
- TLS/SSL support (optional)

## API Security

### API Key Authentication

The History Collector API supports API key authentication.

#### Generating API Keys

```python
from security import generate_api_key

# Generate a new API key
key = generate_api_key("wvr")  # e.g., "wvr_abc123..."
```

#### Enabling API Key Authentication

Set environment variables:

```bash
# Comma-separated list of valid API keys
export API_KEYS="wvr_key1,wvr_key2"

# Require API key for protected endpoints
export REQUIRE_API_KEY=true
```

#### Using API Keys

Include the API key in request headers:

```bash
curl -H "X-API-Key: wvr_your_key_here" http://localhost:8080/api/v1/devices
```

### Rate Limiting

Rate limiting protects against abuse and DoS attacks.

#### Configuration

```bash
# Enable/disable rate limiting (default: true)
export RATE_LIMIT_ENABLED=true

# Maximum requests per window (default: 100)
export RATE_LIMIT_REQUESTS=100

# Window size in seconds (default: 60)
export RATE_LIMIT_WINDOW=60
```

#### Rate Limit Headers

Responses include rate limit headers:
- `X-RateLimit-Limit`: Maximum requests allowed
- `X-RateLimit-Remaining`: Remaining requests in window
- `X-RateLimit-Reset`: Unix timestamp when window resets

### Input Validation

All inputs are validated:
- Device IDs: Alphanumeric, hyphens, underscores (max 64 chars)
- MAC addresses: Standard format (AA:BB:CC:DD:EE:FF)
- Query parameters: Type and range validation

### Security Headers

The following security headers are added to all responses:
- `X-Content-Type-Options: nosniff`
- `X-XSS-Protection: 1; mode=block`
- `X-Frame-Options: DENY`
- `Referrer-Policy: strict-origin-when-cross-origin`
- `Content-Security-Policy: default-src 'self'`
- `Cache-Control: no-store, max-age=0` (for API responses)

### CORS Configuration

```bash
# Allowed origins (comma-separated, or * for all)
export CORS_ORIGINS="https://dashboard.example.com"

# Allow credentials
export CORS_ALLOW_CREDENTIALS=true
```

## MQTT Security

### Authentication

For production, use password authentication:

1. Generate password file:
```bash
mosquitto_passwd -c mosquitto/passwords.txt admin
mosquitto_passwd mosquitto/passwords.txt csi_processor
mosquitto_passwd mosquitto/passwords.txt esp32_device1
```

2. Use production config:
```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up
```

### Access Control Lists (ACL)

The `acl.conf` file controls topic access:
- Admin users have full access
- ESP32 devices can only publish to their own topics
- Backend services have appropriate read/write permissions
- Dashboard has read-only access

### TLS/SSL

For encrypted MQTT connections:

1. Generate certificates:
```bash
# Generate CA
openssl genrsa -out ca.key 2048
openssl req -new -x509 -days 365 -key ca.key -out ca.crt

# Generate server certificate
openssl genrsa -out server.key 2048
openssl req -new -key server.key -out server.csr
openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out server.crt -days 365
```

2. Mount certificates in Docker:
```yaml
volumes:
  - ./certs:/mosquitto/certs:ro
```

3. Enable TLS in mosquitto.prod.conf (see comments in file)

## Secrets Management

### Environment Variables

Never commit secrets to version control. Use environment variables:

```bash
# .env file (add to .gitignore)
API_KEYS=wvr_secret_key_1,wvr_secret_key_2
MQTT_USERNAME=admin
MQTT_PASSWORD=secure_password
```

### Docker Secrets (Recommended for Production)

```yaml
secrets:
  api_keys:
    file: ./secrets/api_keys.txt
  mqtt_password:
    file: ./secrets/mqtt_password.txt
```

## Deployment Checklist

- [ ] Enable API key authentication (`REQUIRE_API_KEY=true`)
- [ ] Generate strong API keys
- [ ] Configure rate limiting appropriately
- [ ] Enable MQTT password authentication
- [ ] Configure MQTT ACL
- [ ] Enable TLS for MQTT and HTTP
- [ ] Set appropriate CORS origins
- [ ] Review and restrict container permissions
- [ ] Enable logging and monitoring
- [ ] Set up log rotation
- [ ] Configure firewall rules
- [ ] Regular security updates

## Reporting Security Issues

If you discover a security vulnerability, please report it responsibly:
1. Do not open a public issue
2. Email the maintainers directly
3. Provide detailed reproduction steps
4. Allow time for a fix before public disclosure

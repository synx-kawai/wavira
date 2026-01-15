# API Reference

This document describes the REST API provided by the History Collector service.

## Base URL

```
http://localhost:8080/api/v1
```

## Authentication

When API key authentication is enabled, include the key in the `X-API-Key` header:

```bash
curl -H "X-API-Key: wvr_your_api_key" http://localhost:8080/api/v1/devices
```

## Rate Limiting

Default limits: 100 requests per 60 seconds per IP.

Response headers:
- `X-RateLimit-Limit`: Maximum requests allowed
- `X-RateLimit-Remaining`: Remaining requests
- `X-RateLimit-Reset`: Unix timestamp when limit resets

## Endpoints

### Health Check

```
GET /health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": 1705320000.0
}
```

---

### List Devices

```
GET /api/v1/devices
```

Returns list of known devices with last seen timestamps.

Response:
```json
{
  "devices": [
    {
      "device_id": "esp32-001",
      "last_seen": 1705320000.0,
      "sample_count": 15000
    },
    {
      "device_id": "esp32-002",
      "last_seen": 1705319900.0,
      "sample_count": 12500
    }
  ]
}
```

---

### Get Device History

```
GET /api/v1/history/{device_id}
```

Parameters:
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `device_id` | path | Yes | Device identifier |
| `limit` | query | No | Max records (default: 1000, max: 10000) |
| `since` | query | No | Unix timestamp to filter from |

Response:
```json
{
  "device_id": "esp32-001",
  "records": [
    {
      "timestamp": 1705320000.0,
      "rssi": -45,
      "amplitude": [1.2, 1.5, 1.3, ...],
      "phase": [0.1, -0.2, 0.15, ...]
    }
  ],
  "count": 100
}
```

Error Responses:
- `400 Bad Request`: Invalid device_id format
- `404 Not Found`: Device not found
- `429 Too Many Requests`: Rate limit exceeded

---

### Get Hourly Summary

```
GET /api/v1/summary/{device_id}
```

Parameters:
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `device_id` | path | Yes | Device identifier |
| `hours` | query | No | Hours to look back (default: 24, max: 168) |

Response:
```json
{
  "device_id": "esp32-001",
  "summary": [
    {
      "hour": "2024-01-15T14:00:00Z",
      "sample_count": 3600,
      "avg_rssi": -47.5,
      "min_rssi": -65,
      "max_rssi": -35
    }
  ]
}
```

---

### Get Latest Data

```
GET /api/v1/latest/{device_id}
```

Returns the most recent CSI sample for a device.

Response:
```json
{
  "device_id": "esp32-001",
  "timestamp": 1705320000.0,
  "rssi": -45,
  "data": {
    "amplitude": [...],
    "phase": [...]
  }
}
```

---

### Get Analysis Results

```
GET /api/v1/analysis/{device_id}
```

Parameters:
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `device_id` | path | Yes | Device identifier |
| `type` | query | No | Analysis type: `crowd`, `gesture`, `presence` |
| `limit` | query | No | Max results (default: 100) |

Response:
```json
{
  "device_id": "esp32-001",
  "type": "crowd",
  "results": [
    {
      "timestamp": 1705320000.0,
      "prediction": 2,
      "confidence": 0.85,
      "label": "Medium (3-5 people)"
    }
  ]
}
```

---

## Error Responses

All errors follow this format:

```json
{
  "detail": "Error message description"
}
```

HTTP Status Codes:
- `400 Bad Request`: Invalid input
- `401 Unauthorized`: Missing or invalid API key
- `404 Not Found`: Resource not found
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error

---

## Input Validation

### Device ID

- Pattern: `^[a-zA-Z0-9_-]+$`
- Length: 1-64 characters

### MAC Address

- Pattern: `^([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}$`
- Example: `AA:BB:CC:DD:EE:FF`

---

## Examples

### Python (requests)

```python
import requests

BASE_URL = "http://localhost:8080/api/v1"
headers = {"X-API-Key": "wvr_your_key"}

# List devices
response = requests.get(f"{BASE_URL}/devices", headers=headers)
devices = response.json()["devices"]

# Get history
response = requests.get(
    f"{BASE_URL}/history/esp32-001",
    headers=headers,
    params={"limit": 100}
)
history = response.json()
```

### JavaScript (fetch)

```javascript
const BASE_URL = "http://localhost:8080/api/v1";
const headers = { "X-API-Key": "wvr_your_key" };

// List devices
const response = await fetch(`${BASE_URL}/devices`, { headers });
const { devices } = await response.json();

// Get latest
const latest = await fetch(`${BASE_URL}/latest/esp32-001`, { headers });
const data = await latest.json();
```

### cURL

```bash
# List devices
curl -H "X-API-Key: wvr_your_key" http://localhost:8080/api/v1/devices

# Get history with limit
curl -H "X-API-Key: wvr_your_key" \
  "http://localhost:8080/api/v1/history/esp32-001?limit=50"

# Get hourly summary
curl -H "X-API-Key: wvr_your_key" \
  "http://localhost:8080/api/v1/summary/esp32-001?hours=12"
```

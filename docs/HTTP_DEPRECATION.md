# HTTP CSI Endpoint Deprecation Notice

## Overview

As of version 1.0.0, the HTTP-based CSI data endpoints have been **deprecated and removed** in favor of MQTT-based communication. This document explains the migration path and rationale.

## Deprecated Endpoints

The following HTTP endpoints are no longer available:

| Endpoint | Method | Description | Replacement |
|----------|--------|-------------|-------------|
| `/api/v1/csi` | POST | Submit single CSI packet | MQTT topic `wavira/csi/{device_id}` |
| `/api/v1/csi/batch` | POST | Submit batch of CSI packets | MQTT topic `wavira/csi/{device_id}` |

## Why MQTT?

The migration to MQTT provides several advantages:

### 1. Real-time Communication
- Push-based messaging eliminates polling overhead
- Sub-second latency for CSI data delivery
- Bidirectional communication for device control

### 2. Reliability
- QoS levels (0, 1, 2) for guaranteed delivery
- Automatic reconnection with exponential backoff
- Message persistence during network outages

### 3. Scalability
- Reduced server load (no HTTP connection overhead)
- Efficient pub/sub model for multiple consumers
- Better suited for IoT device patterns

### 4. Device Management
- Last Will and Testament for offline detection
- Real-time device status monitoring
- Centralized credential management

### 5. Security
- Per-device authentication credentials
- Fine-grained ACL (Access Control List)
- Credential rotation support

## Migration Timeline

| Phase | Date | Status |
|-------|------|--------|
| MQTT infrastructure deployed | 2025-12 | Completed |
| ESP32 MQTT firmware released | 2026-01 | Completed |
| HTTP endpoints deprecated | 2026-01 | Completed |
| HTTP endpoints removed | 2026-01 | **Current** |

## How to Migrate

### For ESP32 Devices

1. Update firmware to MQTT-enabled version:
   ```bash
   cd esp-csi/examples/get-started/csi_recv
   idf.py menuconfig  # Select CSI_OUTPUT_MQTT
   idf.py build flash
   ```

2. Configure MQTT settings in menuconfig:
   - Wi-Fi SSID and password
   - MQTT broker URL
   - Device ID and credentials

See [MQTT Migration Guide](./MQTT_MIGRATION.md) for detailed instructions.

### For Custom Clients

Replace HTTP POST calls with MQTT publish:

**Before (HTTP):**
```python
import requests

data = {
    "device_id": "esp32-001",
    "timestamp": 1234567890,
    "rssi": -50,
    "amplitudes": [...]
}
response = requests.post("http://server:8080/api/v1/csi", json=data)
```

**After (MQTT):**
```python
import paho.mqtt.client as mqtt
import json

client = mqtt.Client()
client.connect("mqtt-broker", 1883)

data = {
    "device_id": "esp32-001",
    "timestamp": 1234567890,
    "rssi": -50,
    "amplitudes": [...]
}
client.publish("wavira/csi/esp32-001", json.dumps(data))
```

## API Compatibility

### Current Available Endpoints

The following REST API endpoints remain available for data retrieval:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | GET | Health check |
| `/api/v1/devices` | GET | List known devices |
| `/api/v1/history/{device_id}` | GET | Get device history |
| `/api/v1/history/{device_id}/hourly` | GET | Hourly aggregation |
| `/api/v1/status` | GET | Device online/offline status |
| `/api/v1/status/{device_id}` | GET | Single device status |
| `/api/v1/admin/devices` | POST/GET | Device management |

### MQTT Topics

CSI data should now be published to these topics:

| Topic | Direction | Description |
|-------|-----------|-------------|
| `wavira/csi/{device_id}` | Device → Server | Raw CSI data |
| `wavira/analysis/{device_id}` | Server → Clients | Processed analysis |
| `wavira/device/{device_id}/status` | Device → Server | Online announcement |
| `wavira/device/{device_id}/will` | Broker → Server | Offline detection (Last Will) |
| `wavira/device/{device_id}/ota` | Server → Device | OTA update commands |

## Troubleshooting

### "Connection Refused" on HTTP endpoint

The HTTP CSI endpoints have been removed. Please migrate to MQTT.

### MQTT Connection Issues

1. Verify broker is running: `mosquitto -v`
2. Check credentials: Ensure device has valid MQTT credentials
3. Verify network: Device must reach MQTT broker on port 1883

### Device Not Appearing in Dashboard

1. Check device is publishing to correct topic
2. Verify Last Will message is configured
3. Check server logs for connection events

## Getting Help

- See [MQTT Protocol Documentation](./MQTT_PROTOCOL.md)
- See [MQTT Migration Guide](./MQTT_MIGRATION.md)
- See [Troubleshooting Guide](./TROUBLESHOOTING.md)
- Open an issue on GitHub for further assistance

## Related Issues

- Issue #21: MQTT Broker Setup
- Issue #22: Server-side MQTT Client
- Issue #23: Device Authentication
- Issue #24: Online/Offline Detection
- Issue #25: ESP32 MQTT Client
- Issue #27: MQTT Integration Tests
- Issue #28: Device Migration
- Issue #29: HTTP Deprecation (this document)

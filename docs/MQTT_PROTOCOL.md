# MQTT Protocol Specification

This document describes the MQTT message formats used in the Wavira system.

## Connection

### Broker Settings

| Setting | Development | Production |
|---------|-------------|------------|
| Host | `localhost` | Your broker host |
| MQTT Port | `1883` | `8883` (TLS) |
| WebSocket Port | `9001` | `9002` (WSS) |
| Authentication | Anonymous | Username/Password |

### Client ID Format

```
wavira-<component>-<instance>
```

Examples:
- `wavira-esp32-001`
- `wavira-processor-01`
- `wavira-dashboard-abc123`

## Topic Structure

```
wavira/<category>/<identifier>[/<subcategory>]
```

### Topic Hierarchy

```
wavira/
├── csi/<device_id>           # Raw CSI data from devices
├── analysis/
│   ├── crowd/<device_id>     # Crowd estimation results
│   ├── gesture/<device_id>   # Gesture recognition results
│   └── presence/<device_id>  # Presence detection
├── device/<device_id>/
│   ├── status                # Device status
│   ├── config                # Configuration
│   └── metrics               # Performance metrics
├── command/<device_id>/
│   ├── reset                 # Reset command
│   ├── config                # Configuration update
│   └── led                   # LED control
└── system/
    ├── health                # System health
    └── events                # System events
```

## Message Formats

### CSI Data

**Topic**: `wavira/csi/<device_id>`
**QoS**: 0 (At most once)
**Retain**: false

```json
{
  "timestamp": 1705320000.123,
  "mac": "AA:BB:CC:DD:EE:FF",
  "rssi": -45,
  "rate": 11,
  "noise_floor": -95,
  "channel": 6,
  "bandwidth": 20,
  "csi": {
    "amplitude": [1.2, 1.5, 1.3, ...],
    "phase": [0.1, -0.2, 0.15, ...]
  }
}
```

Fields:
| Field | Type | Description |
|-------|------|-------------|
| timestamp | float | Unix timestamp with milliseconds |
| mac | string | Source MAC address |
| rssi | int | Received Signal Strength Indicator (dBm) |
| rate | int | Data rate (Mbps) |
| noise_floor | int | Noise floor (dBm) |
| channel | int | Wi-Fi channel |
| bandwidth | int | Channel bandwidth (MHz) |
| csi.amplitude | float[] | CSI amplitude per subcarrier |
| csi.phase | float[] | CSI phase per subcarrier (radians) |

### Crowd Estimation Result

**Topic**: `wavira/analysis/crowd/<device_id>`
**QoS**: 1 (At least once)
**Retain**: true

```json
{
  "timestamp": 1705320000.0,
  "device_id": "esp32-001",
  "prediction": {
    "level": 2,
    "label": "Medium (3-5 people)",
    "confidence": 0.85,
    "probabilities": [0.05, 0.08, 0.85, 0.02]
  },
  "model": {
    "name": "crowd_estimator_v1",
    "version": "1.0.0"
  }
}
```

### Gesture Recognition Result

**Topic**: `wavira/analysis/gesture/<device_id>`
**QoS**: 1
**Retain**: false

```json
{
  "timestamp": 1705320000.0,
  "device_id": "esp32-001",
  "prediction": {
    "gesture": "wave_right",
    "confidence": 0.92,
    "duration_ms": 1200
  },
  "model": {
    "name": "gesture_3dcnn",
    "version": "1.0.0"
  }
}
```

### Device Status

**Topic**: `wavira/device/<device_id>/status`
**QoS**: 1
**Retain**: true

```json
{
  "timestamp": 1705320000.0,
  "device_id": "esp32-001",
  "status": "online",
  "uptime_seconds": 3600,
  "wifi": {
    "ssid": "CSI_Network",
    "rssi": -55,
    "channel": 6
  },
  "firmware": {
    "version": "1.2.0",
    "build_date": "2024-01-15"
  },
  "stats": {
    "packets_sent": 15000,
    "errors": 0,
    "buffer_usage": 0.45
  }
}
```

### Device Configuration

**Topic**: `wavira/device/<device_id>/config`
**QoS**: 1
**Retain**: true

```json
{
  "sample_rate_hz": 100,
  "channel": 6,
  "bandwidth": 20,
  "led_enabled": true,
  "led_brightness": 128,
  "publish_interval_ms": 100,
  "buffer_size": 1000
}
```

### Command: LED Control

**Topic**: `wavira/command/<device_id>/led`
**QoS**: 1

```json
{
  "action": "set",
  "led": "status",
  "state": "blink",
  "color": [0, 255, 0],
  "interval_ms": 500
}
```

LED States: `off`, `on`, `blink`, `pulse`

### Command: Reset

**Topic**: `wavira/command/<device_id>/reset`
**QoS**: 1

```json
{
  "type": "soft",
  "reason": "maintenance"
}
```

Reset Types: `soft` (restart), `hard` (full reset), `factory` (factory defaults)

### System Health

**Topic**: `wavira/system/health`
**QoS**: 1
**Retain**: true

```json
{
  "timestamp": 1705320000.0,
  "services": {
    "csi_processor": {
      "status": "healthy",
      "uptime_seconds": 86400,
      "messages_processed": 1500000
    },
    "history_collector": {
      "status": "healthy",
      "database_size_mb": 256
    }
  },
  "devices_online": 3,
  "total_throughput_msg_sec": 150
}
```

## Quality of Service (QoS)

| Topic Pattern | QoS | Rationale |
|---------------|-----|-----------|
| wavira/csi/# | 0 | High frequency, loss acceptable |
| wavira/analysis/# | 1 | Results should be delivered |
| wavira/device/+/status | 1 | Status updates important |
| wavira/command/# | 1 | Commands must be delivered |
| wavira/system/# | 1 | System messages important |

## Retained Messages

| Topic Pattern | Retained | Rationale |
|---------------|----------|-----------|
| wavira/csi/# | No | Real-time stream |
| wavira/analysis/crowd/# | Yes | Last known state |
| wavira/analysis/gesture/# | No | Event-based |
| wavira/device/+/status | Yes | Current device status |
| wavira/device/+/config | Yes | Current configuration |

## Client Examples

### Python (paho-mqtt)

```python
import json
import paho.mqtt.client as mqtt

def on_message(client, userdata, msg):
    data = json.loads(msg.payload)
    print(f"{msg.topic}: {data}")

client = mqtt.Client(client_id="wavira-example")
client.on_message = on_message

client.connect("localhost", 1883)
client.subscribe("wavira/csi/#")
client.loop_forever()
```

### JavaScript (MQTT.js)

```javascript
const mqtt = require('mqtt');

const client = mqtt.connect('ws://localhost:9001');

client.on('connect', () => {
  client.subscribe('wavira/analysis/#');
});

client.on('message', (topic, message) => {
  const data = JSON.parse(message.toString());
  console.log(topic, data);
});
```

## Access Control

See `tools/csi_visualizer/mosquitto/acl.conf` for topic-level access control rules.

Default permissions:
- ESP32 devices: Publish own CSI, subscribe to own commands
- Backend services: Full access to wavira/# topics
- Dashboard: Read-only access to wavira/# topics

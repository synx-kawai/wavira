# MQTT Topic Design

## Topic Hierarchy

```
wavira/
├── device/
│   └── {device_id}/
│       ├── csi/
│       │   └── batch          # CSI batch data
│       ├── status             # Online status notification
│       └── will               # Last Will and Testament
└── system/
    └── announcement           # System-wide announcements
```

## Topic Details

### Device CSI Data

**Topic**: `wavira/device/{device_id}/csi/batch`

**Direction**: Device → Server

**QoS**: 1 (At least once delivery)

**Retained**: No

**Payload Format**:
```json
{
  "device_id": "esp32-001",
  "timestamp": 1703424000000,
  "batch": [
    {
      "mac": "aa:bb:cc:dd:ee:ff",
      "rssi": -45,
      "rate": 11,
      "noise_floor": -95,
      "channel": 6,
      "timestamp": 1703424000000,
      "csi_data": [10, -5, 8, -3, ...]
    }
  ]
}
```

**Field Descriptions**:
| Field | Type | Description |
|-------|------|-------------|
| device_id | string | Unique device identifier |
| timestamp | int64 | Unix timestamp in milliseconds |
| batch | array | Array of CSI measurements |
| mac | string | Source MAC address |
| rssi | int | Received signal strength (-100 to 0 dBm) |
| rate | int | Data rate |
| noise_floor | int | Noise floor in dBm |
| channel | int | Wi-Fi channel |
| csi_data | array | CSI amplitude/phase data |

### Device Status

**Topic**: `wavira/device/{device_id}/status`

**Direction**: Device → Server

**QoS**: 1

**Retained**: Yes (latest status always available)

**Payload Format**:
```json
{
  "device_id": "esp32-001",
  "status": "online",
  "timestamp": 1703424000000,
  "firmware_version": "1.2.0",
  "ip_address": "192.168.1.100",
  "uptime_seconds": 3600
}
```

### Last Will and Testament

**Topic**: `wavira/device/{device_id}/will`

**Direction**: Broker → Server (on unexpected disconnect)

**QoS**: 1

**Retained**: Yes

**Payload Format**:
```json
{
  "device_id": "esp32-001",
  "status": "offline",
  "timestamp": 1703424000000,
  "reason": "unexpected_disconnect"
}
```

### System Announcements

**Topic**: `wavira/system/announcement`

**Direction**: Server → Devices

**QoS**: 0 (Best effort)

**Retained**: No

**Payload Format**:
```json
{
  "type": "maintenance",
  "message": "System maintenance in 30 minutes",
  "timestamp": 1703424000000
}
```

## QoS Level Guidelines

| Use Case | QoS | Reason |
|----------|-----|--------|
| CSI Data | 1 | Ensure delivery, allow duplicates |
| Status Updates | 1 | Important state changes |
| Last Will | 1 | Critical for offline detection |
| Announcements | 0 | Best effort, non-critical |

## Wildcard Subscriptions

The server subscribes to:
- `wavira/device/+/csi/batch` - All device CSI data
- `wavira/device/+/status` - All device status updates
- `wavira/device/+/will` - All device disconnections

## Topic Naming Conventions

1. Use lowercase letters
2. Use hyphens for multi-word identifiers
3. Device IDs should be URL-safe (alphanumeric + hyphens)
4. Maximum topic length: 128 characters

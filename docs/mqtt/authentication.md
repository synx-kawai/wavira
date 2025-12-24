# MQTT Authentication

## Overview

Wavira uses username/password authentication for MQTT connections. Each device receives unique credentials upon registration.

## Authentication Flow

```
┌─────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Admin     │────▶│   API Server    │────▶│    devices.db   │
│  Register   │     │  /api/v1/admin  │     │  (credentials)  │
└─────────────┘     └─────────────────┘     └─────────────────┘
                            │
                            ▼
                    ┌─────────────────┐
                    │  MQTT Password  │
                    │     File        │
                    └─────────────────┘
                            │
                            ▼
                    ┌─────────────────┐
                    │  MQTT Broker    │
                    │  (Mosquitto)    │
                    └─────────────────┘
```

## Credential Format

### Device Registration Response

```json
{
  "device_id": "esp32-001",
  "api_key": "api_xxx...",
  "mqtt_credentials": {
    "username": "device_esp32-001",
    "password": "generated_secure_password",
    "client_id": "wavira_esp32-001"
  }
}
```

### Username Format

```
device_{device_id}
```

Example: `device_esp32-001`

### Client ID Format

```
wavira_{device_id}
```

Example: `wavira_esp32-001`

## Security Considerations

### Password Storage

- Passwords are hashed using bcrypt before storage
- Plain-text passwords are only shown once during registration
- Password rotation is supported via the admin API

### TLS Encryption

For production deployments, enable TLS:

```
# mosquitto.conf
listener 8883
cafile /mosquitto/certs/ca.crt
certfile /mosquitto/certs/server.crt
keyfile /mosquitto/certs/server.key
require_certificate false
```

### Access Control

Each device can only:
- Publish to `wavira/device/{own_device_id}/#`
- Subscribe to `wavira/system/announcement`

ACL configuration:
```
# mosquitto_acl.conf
user device_esp32-001
topic write wavira/device/esp32-001/#
topic read wavira/system/announcement
```

## API Endpoints

### Register New Device

```bash
POST /api/v1/admin/devices
Authorization: Bearer <admin_token>

{
  "device_id": "esp32-001",
  "zone": "office",
  "location": "entrance"
}
```

### Rotate MQTT Password

```bash
POST /api/v1/admin/devices/{device_id}/rotate-mqtt-password
Authorization: Bearer <admin_token>
```

### Revoke Device Access

```bash
DELETE /api/v1/admin/devices/{device_id}
Authorization: Bearer <admin_token>
```

## Troubleshooting

### Connection Refused (Error 5)

- Check username/password
- Verify device is registered
- Check password hasn't expired

### Not Authorized (Error 135)

- Device trying to access unauthorized topic
- Check ACL configuration

### Client ID Already In Use

- Another client using the same client_id
- Ensure unique client_id per device

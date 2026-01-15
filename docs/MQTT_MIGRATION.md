# Existing Device MQTT Migration Guide

This document provides step-by-step instructions for migrating existing ESP32 devices to the new MQTT authentication system introduced in Issue #23.

## Overview

The Wavira system has been updated to support per-device MQTT authentication. This migration guide covers:
- Generating MQTT credentials for existing devices
- Updating ESP32 firmware with new credentials
- Verifying connectivity
- Rollback procedures if issues occur

### What Changed

| Before | After |
|--------|-------|
| Anonymous MQTT (or shared credentials) | Per-device MQTT username/password |
| No device-level access control | ACL-based topic restrictions |
| Shared client ID | Unique client ID per device |

### Benefits

- **Security**: Each device has unique credentials that can be revoked individually
- **Auditing**: Track which device publishes to which topic
- **Access Control**: Devices can only write to their designated topics
- **Credential Rotation**: Rotate individual device credentials without affecting others

## Prerequisites

Before starting the migration, ensure:

1. **Wavira Services Running**
   ```bash
   cd tools/csi_visualizer
   docker-compose up -d mosquitto
   ```

2. **Python Environment**
   ```bash
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```

3. **Access to Device Manager**
   ```bash
   cd tools/csi_visualizer/services
   python device_manager.py --help
   ```

4. **Physical Access to ESP32 Devices** (for firmware update)
   - USB cable (data-capable, not charge-only)
   - Serial port driver installed

5. **Backup Current Configuration**
   - Document current device settings (device IDs, zones, network config)
   - Export current Mosquitto password/ACL files if customized

## Migration Steps

### Step 1: Generate MQTT Credentials

Use `device_manager.py` to generate MQTT credentials for existing devices.

#### For a Single Device

```bash
cd tools/csi_visualizer/services

# Add MQTT credentials to an existing device
python device_manager.py rotate <device_id>

# Example:
python device_manager.py rotate esp32-001
```

Output:
```
MQTT credentials rotated for: esp32-001
  MQTT Username: dev_esp32_001
  MQTT Password: <random-32-char-password>
  MQTT Client ID: wavira_esp32_001

IMPORTANT: Update device firmware with new credentials.
```

> **CRITICAL**: Save the password immediately. It cannot be retrieved later.

#### For All Devices Without MQTT Credentials

```bash
python device_manager.py migrate
```

Output:
```
Migrated 3 devices:

esp32-001:
  MQTT Username: dev_esp32_001
  MQTT Password: <password>
  MQTT Client ID: wavira_esp32_001

esp32-002:
  MQTT Username: dev_esp32_002
  MQTT Password: <password>
  MQTT Client ID: wavira_esp32_002

...
```

#### Verify Device Status

```bash
python device_manager.py list
```

Output:
```
Device ID            MQTT User                 Zone       Online
----------------------------------------------------------------------
esp32-001            dev_esp32_001             default    No
esp32-002            dev_esp32_002             zone-a     No
```

### Step 2: Update Mosquitto Configuration

After generating credentials, update Mosquitto password and ACL files.

```bash
cd tools/csi_visualizer/services

# Generate Mosquitto configuration files
python device_manager.py mosquitto \
    --passwd-file ../mosquitto/passwords.txt \
    --acl-file ../mosquitto/acl.conf

# Restart Mosquitto to apply changes
docker-compose restart mosquitto
```

Verify Mosquitto is running with authentication:
```bash
# Should fail without credentials
mosquitto_sub -h localhost -t "wavira/#" -v
# Error: Connection refused: not authorized

# Should work with credentials
mosquitto_sub -h localhost -t "wavira/#" -v \
    -u dev_esp32_001 -P "<password>"
```

### Step 3: Update ESP32 Firmware

There are two methods to update the firmware: OTA (Over-The-Air) or manual flashing.

#### Method A: Manual Flashing (Recommended for First Migration)

1. **Connect ESP32 via USB**
   ```bash
   # List available ports (macOS)
   ls /dev/cu.usbserial-*

   # Common port names:
   # macOS: /dev/cu.usbserial-0001
   # Linux: /dev/ttyUSB0
   # Windows: COM3
   ```

2. **Configure Firmware**
   ```bash
   cd esp-csi/examples/get-started/csi_recv

   # Set up ESP-IDF environment
   source $HOME/esp/esp-idf/export.sh

   # Open configuration menu
   idf.py menuconfig
   ```

3. **Update Configuration in menuconfig**

   Navigate to `Wavira CSI MQTT Configuration`:

   | Setting | Path | Value |
   |---------|------|-------|
   | Output Mode | `CSI Data Output Mode` | `MQTT Output (Wi-Fi)` |
   | Wi-Fi SSID | `Wi-Fi Configuration > Wi-Fi SSID` | Your network SSID |
   | Wi-Fi Password | `Wi-Fi Configuration > Wi-Fi Password` | Your network password |
   | MQTT Broker URL | `MQTT Broker Configuration > Broker URL` | `mqtt://192.168.x.x` |
   | MQTT Username | `MQTT Broker Configuration > Username` | `dev_esp32_001` |
   | MQTT Password | `MQTT Broker Configuration > Password` | `<generated-password>` |
   | Device ID | `Device Configuration > Device ID` | `esp32-001` |

4. **Build and Flash**
   ```bash
   # Build firmware
   idf.py build

   # Flash to device (replace port as needed)
   idf.py -p /dev/cu.usbserial-0001 flash

   # Monitor output
   idf.py -p /dev/cu.usbserial-0001 monitor
   ```

5. **Expected Serial Output**
   ```
   I (xxx) wavira_csi: Connecting to Wi-Fi SSID: YourNetwork
   I (xxx) wavira_csi: Wi-Fi connected, IP: 192.168.1.xxx
   I (xxx) wavira_csi: Connecting to MQTT broker: mqtt://192.168.1.100
   I (xxx) wavira_csi: MQTT connected successfully
   I (xxx) wavira_csi: Device ID: esp32-001
   I (xxx) wavira_csi: Publishing to: wavira/csi/esp32-001
   ```

#### Method B: Using sdkconfig File

For multiple devices, create a template `sdkconfig` file:

```bash
cd esp-csi/examples/get-started/csi_recv

# Create device-specific config
cat > sdkconfig.esp32-001 << 'EOF'
CONFIG_CSI_OUTPUT_MQTT=y
CONFIG_WAVIRA_WIFI_SSID="YourNetwork"
CONFIG_WAVIRA_WIFI_PASSWORD="YourPassword"
CONFIG_WAVIRA_MQTT_BROKER_URL="mqtt://192.168.1.100"
CONFIG_WAVIRA_MQTT_USERNAME="dev_esp32_001"
CONFIG_WAVIRA_MQTT_PASSWORD="<generated-password>"
CONFIG_WAVIRA_DEVICE_ID="esp32-001"
CONFIG_WAVIRA_DEVICE_ZONE="default"
EOF

# Build with custom config
cp sdkconfig.esp32-001 sdkconfig
idf.py build
idf.py -p /dev/cu.usbserial-0001 flash
```

#### Method C: OTA Update (Future Support)

OTA firmware updates are planned for Issue #44. Once available:

```bash
# Placeholder for future OTA command
python scripts/ota_update.py --device esp32-001 --firmware build/wavira_csi.bin
```

### Step 4: Verify Device Connectivity

After updating the firmware, verify the device is connecting properly.

#### Check Device Online Status

```bash
cd tools/csi_visualizer/services

# Check all devices
python device_manager.py list --online
```

#### Verify MQTT Messages

```bash
# Subscribe to device CSI topic
mosquitto_sub -h localhost -t "wavira/csi/esp32-001" -v \
    -u wavira_server -P "<server-password>"

# Subscribe to device status
mosquitto_sub -h localhost -t "wavira/device/esp32-001/status" -v \
    -u wavira_server -P "<server-password>"
```

Expected output:
```
wavira/csi/esp32-001 {"timestamp":1736928000.123,"device_id":"esp32-001","rssi":-45,...}
wavira/device/esp32-001/status {"status":"online","uptime_seconds":120,...}
```

#### Check LED Status

| LED State | Meaning |
|-----------|---------|
| Green solid | Connected and operating normally |
| Green pulse | Sending CSI data |
| Blue blink | Connecting to MQTT |
| Yellow blink | Connecting to Wi-Fi |
| Red blink | Error (check serial output) |

#### Test with Dashboard

```bash
# Start full stack
cd tools/csi_visualizer
docker-compose up -d

# Open dashboard in browser
open dashboard_multi.html
```

The device should appear in the device list with real-time CSI data.

## Troubleshooting

### Connection Refused (MQTT Authentication Failed)

**Symptoms:**
```
E (xxx) wavira_csi: MQTT connection refused: not authorized
```

**Solutions:**
1. Verify username/password in firmware match device_manager output
2. Check Mosquitto password file was regenerated:
   ```bash
   python device_manager.py mosquitto
   docker-compose restart mosquitto
   ```
3. Ensure `allow_anonymous false` in Mosquitto config

### Device Shows Offline

**Symptoms:** Device connects but appears offline in device list.

**Solutions:**
1. Verify device ID in firmware matches registered device:
   ```bash
   python device_manager.py list
   ```
2. Check device is publishing status messages:
   ```bash
   mosquitto_sub -t "wavira/device/+/status" -v -u admin -P <password>
   ```
3. Verify MQTT client ID is unique (no conflicts with other devices)

### Wi-Fi Connection Failed

**Symptoms:**
```
E (xxx) wavira_csi: Failed to connect to Wi-Fi
```

**Solutions:**
1. Verify SSID is exactly correct (case-sensitive)
2. Ensure router is 2.4 GHz (ESP32 does not support 5 GHz)
3. Move ESP32 closer to router
4. Check Wi-Fi password contains no special characters that need escaping

### Certificate/TLS Errors (Production)

**Symptoms:**
```
E (xxx) wavira_csi: TLS handshake failed
```

**Solutions:**
1. Verify broker URL uses correct protocol (`mqtts://` for TLS)
2. Check CA certificate is valid and not expired
3. Ensure system time is synchronized on ESP32
4. See [SECURITY.md](SECURITY.md) for TLS configuration

### Credential Rotation Issues

If credentials need to be rotated after migration:

```bash
# Rotate credentials for a specific device
python device_manager.py rotate esp32-001

# Update Mosquitto files
python device_manager.py mosquitto
docker-compose restart mosquitto

# Re-flash device with new credentials (or use OTA when available)
```

## Rollback Procedure

If migration fails and you need to revert to anonymous MQTT:

### Step 1: Revert Mosquitto Configuration

```bash
cd tools/csi_visualizer/mosquitto

# Edit mosquitto.conf
# Change: allow_anonymous false
# To:     allow_anonymous true

# Comment out password and ACL files
# #password_file /mosquitto/config/passwords.txt
# #acl_file /mosquitto/config/acl.conf

# Restart Mosquitto
docker-compose restart mosquitto
```

### Step 2: Revert ESP32 Firmware

```bash
cd esp-csi/examples/get-started/csi_recv
idf.py menuconfig
```

In menuconfig:
1. Navigate to `MQTT Broker Configuration`
2. Clear `Username` field
3. Clear `Password` field

```bash
idf.py build
idf.py -p /dev/cu.usbserial-0001 flash
```

### Step 3: Verify Anonymous Connection

```bash
# Should work without credentials
mosquitto_sub -h localhost -t "wavira/#" -v
```

### Step 4: Document Issues

Before rolling back, document:
- Error messages encountered
- Steps to reproduce the issue
- Device IDs affected
- Create a GitHub issue with this information

## Migration Checklist

Use this checklist to track migration progress for each device:

```
Device: _______________

[ ] 1. Generate MQTT credentials
    Username: _______________
    Password: (saved securely)
    Client ID: _______________

[ ] 2. Update Mosquitto config files
    [ ] Password file regenerated
    [ ] ACL file regenerated
    [ ] Mosquitto restarted

[ ] 3. Update ESP32 firmware
    [ ] Wi-Fi settings configured
    [ ] MQTT credentials configured
    [ ] Device ID verified
    [ ] Firmware flashed

[ ] 4. Verify connectivity
    [ ] Device shows online
    [ ] CSI data flowing
    [ ] LED status green
    [ ] Dashboard shows device

[ ] 5. Post-migration
    [ ] Old credentials documented for rollback
    [ ] Device added to monitoring
```

## Batch Migration Script

For migrating multiple devices efficiently:

```bash
#!/bin/bash
# migrate_devices.sh

DEVICES=("esp32-001" "esp32-002" "esp32-003")
OUTPUT_FILE="migration_credentials.txt"

cd tools/csi_visualizer/services

echo "=== Wavira MQTT Migration ===" > $OUTPUT_FILE
echo "Generated: $(date)" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

for device in "${DEVICES[@]}"; do
    echo "Migrating $device..."
    python device_manager.py rotate "$device" >> $OUTPUT_FILE 2>&1
    echo "" >> $OUTPUT_FILE
done

# Update Mosquitto
python device_manager.py mosquitto

echo "Migration complete. Credentials saved to $OUTPUT_FILE"
echo "IMPORTANT: Store credentials securely and delete this file after use."
```

## References

- [MQTT Protocol Specification](MQTT_PROTOCOL.md)
- [Security Guide](SECURITY.md)
- [Troubleshooting Guide](TROUBLESHOOTING.md)
- [ESP32 Firmware README](../esp-csi/examples/get-started/csi_recv/README.md)
- [Device Manager Source](../tools/csi_visualizer/services/device_manager.py)

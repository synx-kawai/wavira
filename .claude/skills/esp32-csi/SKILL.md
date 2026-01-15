---
name: esp32-csi
description: Use this skill when working with ESP32 CSI hardware, including serial port monitoring, firmware building/flashing, MQTT configuration, CSI data collection, LED status interpretation, or troubleshooting ESP32 connectivity issues.
---

# ESP32 CSI Hardware Operations

This skill provides expertise for ESP32 Channel State Information (CSI) operations in the Wavira project.

## CRITICAL: Serial Port Safety Rules

**NEVER use the following commands - they will cause Claude to freeze:**

```bash
# FORBIDDEN - causes indefinite hang
cat /dev/cu.usbserial-*
timeout X cat /dev/cu.usbserial-* | head -N
stty -f /dev/cu.usbserial-* && cat /dev/cu.usbserial-*
dd if=/dev/cu.usbserial-*
```

**Why these freeze:**
1. Serial devices block indefinitely waiting for data
2. `timeout` doesn't work reliably with piped serial output
3. Baud rate mismatch causes garbage data without proper line endings

**Safe alternatives:**
1. Use `scripts/esp32_monitor.py` (pyserial-based, timeout-safe)
2. Check MQTT dashboard at https://wavira.takezou.com/
3. Use `idf.py monitor` in interactive terminal (not Claude)

## Project Structure

```
esp-csi/examples/get-started/csi_recv/
├── main/
│   ├── app_main.c          # Main firmware source
│   └── Kconfig.projbuild   # Configuration menu definitions
├── sdkconfig.defaults      # Default configuration
├── sdkconfig               # Current build configuration
├── CMakeLists.txt
└── build/                  # Build output
```

## Serial Port Detection

```bash
# macOS
ls /dev/cu.usbserial-*

# Linux
ls /dev/ttyUSB*
```

## ESP-IDF Environment Setup

```bash
# Source ESP-IDF environment (required before any idf.py command)
source ~/esp/esp-idf/export.sh

# Or on some systems
source ~/esp/v5.5/esp-idf/export.sh
```

## Firmware Building

```bash
cd esp-csi/examples/get-started/csi_recv

# Clean build (recommended when changing configuration)
rm -rf build
rm -f sdkconfig
idf.py build

# Incremental build
idf.py build
```

## Firmware Flashing

```bash
# Flash single device
idf.py -p /dev/cu.usbserial-XXXX flash

# Flash with specific baud rate
idf.py -p /dev/cu.usbserial-XXXX -b 460800 flash
```

### Multiple Device Flashing

Each device needs a unique `CONFIG_WAVIRA_DEVICE_ID`. Update `sdkconfig` before each flash:

```bash
# For esp32-001
sed -i '' 's/CONFIG_WAVIRA_DEVICE_ID="esp32-[0-9]*"/CONFIG_WAVIRA_DEVICE_ID="esp32-001"/' sdkconfig
idf.py build && idf.py -p /dev/cu.usbserial-2110 flash

# For esp32-002
sed -i '' 's/CONFIG_WAVIRA_DEVICE_ID="esp32-001"/CONFIG_WAVIRA_DEVICE_ID="esp32-002"/' sdkconfig
idf.py build && idf.py -p /dev/cu.usbserial-2120 flash

# For esp32-003
sed -i '' 's/CONFIG_WAVIRA_DEVICE_ID="esp32-002"/CONFIG_WAVIRA_DEVICE_ID="esp32-003"/' sdkconfig
idf.py build && idf.py -p /dev/cu.usbserial-5AE90127161 flash
```

## Configuration (sdkconfig.defaults)

### Key Settings

```ini
# Wi-Fi Configuration
CONFIG_WAVIRA_WIFI_SSID="your-ssid"
CONFIG_WAVIRA_WIFI_PASSWORD="your-password"

# MQTT Configuration
CONFIG_WAVIRA_MQTT_BROKER_URL="mqtt://IP_ADDRESS"
CONFIG_WAVIRA_MQTT_PORT=1883

# Device Identification (MUST be unique per device)
CONFIG_WAVIRA_DEVICE_ID="esp32-001"
CONFIG_WAVIRA_DEVICE_ZONE="default"

# CSI Trigger Mode
CONFIG_CSI_TRIGGER_ROUTER=y    # Use router ping for CSI
CONFIG_CSI_TRIGGER_FREQUENCY=10  # 10Hz

# Debug
CONFIG_WAVIRA_DEBUG_SERIAL=y
CONFIG_WAVIRA_LED_GPIO=2
```

### Changing MQTT Server IP

1. Edit `sdkconfig.defaults`:
   ```ini
   CONFIG_WAVIRA_MQTT_BROKER_URL="mqtt://NEW_IP_ADDRESS"
   ```

2. Clean rebuild required:
   ```bash
   rm -f sdkconfig
   idf.py build
   ```

3. Flash all devices with unique IDs

## LED Status Indicators

| LED Behavior | Status |
|--------------|--------|
| Slow blink (500ms) | Waiting for Wi-Fi connection |
| Fast blink (200ms) | MQTT disconnected (problem) |
| Solid ON | MQTT connected, waiting for data |
| Quick OFF pulse (30ms) | CSI data being sent |

## CSI Trigger Mechanism

**Important**: CSI data is only generated when Wi-Fi packets are exchanged. The firmware must actively trigger CSI measurements.

### Router Ping Mode (CONFIG_CSI_TRIGGER_ROUTER=y)

The firmware sends UDP packets to the gateway at 10Hz to trigger CSI measurements:

```c
// CSI trigger task - sends UDP packets to gateway
static void csi_trigger_task(void *pvParameter)
{
    int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    struct sockaddr_in dest_addr;
    dest_addr.sin_family = AF_INET;
    dest_addr.sin_port = htons(12345);
    dest_addr.sin_addr.s_addr = s_gateway_ip;

    while (1) {
        if (s_wifi_connected) {
            sendto(sock, "CSI", 3, 0, ...);
        }
        vTaskDelay(pdMS_TO_TICKS(100));  // 10Hz
    }
}
```

**If CSI trigger is not implemented, data will be intermittent!**

## MQTT Topics

| Topic | Purpose |
|-------|---------|
| `wavira/csi/{device_id}` | CSI data stream |
| `wavira/device/{device_id}/status` | Online status |
| `wavira/device/{device_id}/will` | Last Will (offline detection) |

### CSI Data Format (JSON)

```json
{
  "id": "esp32-001",
  "ts": 1234567890,
  "rssi": -70,
  "data": [1, 2, 3, ...]
}
```

## Troubleshooting

### No Serial Port Detected
- Check USB cable (use data cable, not charge-only)
- Try different USB port
- Install CP210x or CH340 drivers if needed

### Garbled Serial Output
- Verify baud rate is 115200
- Serial monitor may not work with `idf.py monitor` (requires TTY)
- Use `cat /dev/cu.usbserial-XXXX` for basic output

### LED Not Blinking / No Data
1. **Check Wi-Fi connection**: LED should stop slow blinking after connecting
2. **Check MQTT connection**: Fast blinking = MQTT problem
3. **Verify CSI trigger task**: Without active packet sending, CSI data is intermittent
4. **Check server IP**: Ensure MQTT broker is reachable

### All Devices Show Same ID on Dashboard
- Each device needs unique `CONFIG_WAVIRA_DEVICE_ID`
- Must rebuild and reflash each device with different ID

### MQTT Connection Unstable
Current firmware settings for high-latency networks:
```c
mqtt_cfg.network.timeout_ms = 30000;           // 30s timeout
mqtt_cfg.network.reconnect_timeout_ms = 5000;  // 5s reconnect delay
mqtt_cfg.session.keepalive = 30;               // 30s keepalive
```

### CSI Data Intermittent
**Root cause**: CSI trigger not implemented or not working

**Solution**: Ensure `csi_trigger_task` is running and sending UDP packets to gateway

## Serial Monitoring

```bash
# Using idf.py (requires TTY - may not work in non-interactive shells)
idf.py -p /dev/cu.usbserial-XXXX monitor

# Using screen
screen /dev/cu.usbserial-XXXX 115200

# Using cat (basic, no formatting)
cat /dev/cu.usbserial-XXXX
```

## Data Collection

```bash
# Collect crowd level data (for training)
python scripts/collect_crowd.py --level 0 --location office --num-files 10
```

## Dashboard

Production dashboard: https://wavira.takezou.com/

Local development:
```bash
cd tools/csi_visualizer
docker-compose up -d
```

## Common Device Serial Ports (Reference)

| Device | Port | Notes |
|--------|------|-------|
| esp32-001 | /dev/cu.usbserial-2110 | |
| esp32-002 | /dev/cu.usbserial-2120 | |
| esp32-003 | /dev/cu.usbserial-5AE90127161 | |

Note: Serial port assignments may change when devices are reconnected.

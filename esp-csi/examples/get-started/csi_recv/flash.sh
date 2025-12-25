#!/bin/bash
# Wavira ESP32 Flash Script
# Usage: ./flash.sh <wifi> <device_id> [port]
#   wifi: home | office
#   device_id: esp32-001 | esp32-002 | esp32-003
#   port: (optional) /dev/cu.usbserial-XXXX

set -e

WIFI_ENV="${1:-office}"
DEVICE_ID="${2:-esp32-001}"
PORT="${3:-}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Validate WiFi environment
if [[ "$WIFI_ENV" != "home" && "$WIFI_ENV" != "office" ]]; then
    echo "Error: WiFi must be 'home' or 'office'"
    echo "Usage: ./flash.sh <wifi> <device_id> [port]"
    exit 1
fi

# Load WiFi config
WIFI_CONFIG="sdkconfig.wifi.$WIFI_ENV"
if [[ ! -f "$WIFI_CONFIG" ]]; then
    echo "Error: WiFi config not found: $WIFI_CONFIG"
    exit 1
fi

echo "=========================================="
echo " Wavira ESP32 Flash"
echo "=========================================="
echo " WiFi:     $WIFI_ENV"
echo " Device:   $DEVICE_ID"
echo " Port:     ${PORT:-auto}"
echo "=========================================="

# Read WiFi settings
WIFI_SSID=$(grep CONFIG_WAVIRA_WIFI_SSID "$WIFI_CONFIG" | cut -d'"' -f2)
WIFI_PASS=$(grep CONFIG_WAVIRA_WIFI_PASSWORD "$WIFI_CONFIG" | cut -d'"' -f2)

echo "WiFi SSID: $WIFI_SSID"

# Update sdkconfig.defaults with WiFi and device ID
sed -i '' "s/CONFIG_WAVIRA_WIFI_SSID=.*/CONFIG_WAVIRA_WIFI_SSID=\"$WIFI_SSID\"/" sdkconfig.defaults
sed -i '' "s/CONFIG_WAVIRA_WIFI_PASSWORD=.*/CONFIG_WAVIRA_WIFI_PASSWORD=\"$WIFI_PASS\"/" sdkconfig.defaults
sed -i '' "s/CONFIG_WAVIRA_DEVICE_ID=.*/CONFIG_WAVIRA_DEVICE_ID=\"$DEVICE_ID\"/" sdkconfig.defaults

echo "Updated sdkconfig.defaults"

# Remove old sdkconfig to force rebuild with new settings
rm -f sdkconfig

# Source ESP-IDF
if [[ -f ~/esp/esp-idf/export.sh ]]; then
    source ~/esp/esp-idf/export.sh >/dev/null 2>&1
else
    echo "Error: ESP-IDF not found at ~/esp/esp-idf"
    exit 1
fi

# Set target and build
echo ""
echo "Building firmware..."
idf.py set-target esp32 >/dev/null 2>&1
idf.py build 2>&1 | tail -5

# Flash
echo ""
echo "Flashing to device..."
if [[ -n "$PORT" ]]; then
    idf.py -p "$PORT" flash 2>&1 | tail -10
else
    idf.py flash 2>&1 | tail -10
fi

echo ""
echo "=========================================="
echo " Done! Device $DEVICE_ID flashed"
echo " WiFi: $WIFI_SSID ($WIFI_ENV)"
echo "=========================================="

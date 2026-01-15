#!/bin/bash
#
# ESP32 CSI Firmware Setup Script
# Minimal input required - auto-detects most settings
#

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WAVIRA_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}  ESP32 CSI Firmware Setup${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# Check ESP-IDF
if [ -z "$IDF_PATH" ]; then
    echo -e "${RED}Error: ESP-IDF not sourced${NC}"
    echo "Run: source ~/esp/esp-idf/export.sh"
    exit 1
fi

# Auto-detect ESP32 port
PORT=$(ls /dev/cu.usbserial* 2>/dev/null | head -1)
if [ -z "$PORT" ]; then
    echo -e "${RED}Error: ESP32 not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓ ESP32: $PORT${NC}"

# Auto-detect local IP
LOCAL_IP=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | head -1 | awk '{print $2}')
if [ -z "$LOCAL_IP" ]; then
    LOCAL_IP="192.168.1.100"
fi
echo -e "${GREEN}✓ Server IP: $LOCAL_IP${NC}"

# Only ask for Wi-Fi credentials (required)
echo ""
echo -e "${YELLOW}Wi-Fi Settings (required):${NC}"
read -p "  SSID: " WIFI_SSID
read -sp "  Password: " WIFI_PASSWORD
echo ""

if [ -z "$WIFI_SSID" ] || [ -z "$WIFI_PASSWORD" ]; then
    echo -e "${RED}Error: Wi-Fi credentials required${NC}"
    exit 1
fi

# Generate config
echo ""
echo -e "${YELLOW}Building...${NC}"

cd "$SCRIPT_DIR"

# Apply settings to sdkconfig (MQTT-based architecture)
cat > sdkconfig.local << EOF
CONFIG_WAVIRA_WIFI_SSID="$WIFI_SSID"
CONFIG_WAVIRA_WIFI_PASSWORD="$WIFI_PASSWORD"
CONFIG_WAVIRA_MQTT_BROKER_URL="mqtt://$LOCAL_IP:1883"
CONFIG_WAVIRA_DEVICE_ID="esp32-001"
EOF

cat sdkconfig.defaults sdkconfig.local > sdkconfig 2>/dev/null || cp sdkconfig.defaults sdkconfig
cat sdkconfig.local >> sdkconfig
rm -f sdkconfig.local

# Build and flash
idf.py build > /dev/null 2>&1 &
PID=$!
while kill -0 $PID 2>/dev/null; do
    echo -n "."
    sleep 1
done
wait $PID
BUILD_STATUS=$?

if [ $BUILD_STATUS -ne 0 ]; then
    echo ""
    echo -e "${RED}Build failed. Running with output:${NC}"
    idf.py build
    exit 1
fi
echo ""
echo -e "${GREEN}✓ Build complete${NC}"

echo -e "${YELLOW}Flashing...${NC}"
idf.py -p "$PORT" flash > /dev/null 2>&1
echo -e "${GREEN}✓ Flash complete${NC}"

echo ""
echo -e "${BLUE}================================${NC}"
echo -e "${GREEN}  Setup Complete!${NC}"
echo -e "${BLUE}================================${NC}"
echo ""
echo "Start services: cd $WAVIRA_ROOT/tools/csi_visualizer && docker-compose up -d"
echo "Monitor ESP32:  idf.py -p $PORT monitor"
echo ""

read -p "Start monitor? [Y/n]: " START
if [[ "${START:-Y}" =~ ^[Yy]$ ]]; then
    idf.py -p "$PORT" monitor
fi

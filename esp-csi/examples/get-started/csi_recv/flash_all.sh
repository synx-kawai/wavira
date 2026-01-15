#!/bin/bash
# Flash all 3 ESP32 devices with stability fixes

set -e

# Device mapping
declare -A DEVICES=(
    ["/dev/cu.usbserial-10"]="esp32-002"
    ["/dev/cu.usbserial-2110"]="esp32-001"
    ["/dev/cu.usbserial-5AE90127161"]="esp32-003"
)

echo "=== Flashing all ESP32 devices with stability fixes ==="
echo ""

for PORT in "${!DEVICES[@]}"; do
    DEVICE_ID="${DEVICES[$PORT]}"
    echo ">>> Flashing $DEVICE_ID on $PORT"
    
    # Update device ID in sdkconfig.defaults
    sed -i '' "s/CONFIG_WAVIRA_DEVICE_ID=.*/CONFIG_WAVIRA_DEVICE_ID=\"$DEVICE_ID\"/" sdkconfig.defaults
    
    # Clean and build
    rm -rf build sdkconfig
    idf.py set-target esp32
    idf.py build
    
    # Flash
    idf.py -p "$PORT" flash
    
    echo ">>> $DEVICE_ID flashed successfully"
    echo ""
done

echo "=== All devices flashed ==="

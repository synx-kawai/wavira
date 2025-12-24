#!/bin/bash
# Initialize MQTT password file for Mosquitto
# Usage: ./init_mqtt_passwords.sh [server_password] [admin_password]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PASSWD_FILE="${SCRIPT_DIR}/../docker/mosquitto/passwd"

# Default passwords (change in production!)
SERVER_PASSWORD="${1:-wavira_server_password}"
ADMIN_PASSWORD="${2:-wavira_admin_password}"

echo "Initializing MQTT password file..."

# Check if mosquitto_passwd is available
if ! command -v mosquitto_passwd &> /dev/null; then
    echo "mosquitto_passwd not found. Installing mosquitto-clients..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y mosquitto-clients
    elif command -v brew &> /dev/null; then
        brew install mosquitto
    else
        echo "Please install mosquitto-clients manually"
        exit 1
    fi
fi

# Create password file
echo "Creating password file at ${PASSWD_FILE}"

# Create new password file with server user
mosquitto_passwd -c -b "${PASSWD_FILE}" wavira_server "${SERVER_PASSWORD}"

# Add admin user
mosquitto_passwd -b "${PASSWD_FILE}" wavira_admin "${ADMIN_PASSWORD}"

echo "Password file created successfully!"
echo ""
echo "Users created:"
echo "  - wavira_server (for API server)"
echo "  - wavira_admin (for management)"
echo ""
echo "IMPORTANT: Change these passwords in production!"
echo ""
echo "To add a device user:"
echo "  mosquitto_passwd -b ${PASSWD_FILE} device_<device_id> <password>"

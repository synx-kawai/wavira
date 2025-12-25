#!/bin/bash
set -e

# Log all output
exec > >(tee /var/log/user-data.log) 2>&1
echo "Starting Wavira server setup at $(date)"

# Variables from Terraform
MQTT_BROKER_HOST="${mqtt_broker_host}"
ENVIRONMENT="${environment}"

# Update system
yum update -y

# Install Docker
yum install -y docker
systemctl start docker
systemctl enable docker
usermod -aG docker ec2-user

# Install Python and dependencies
yum install -y python3 python3-pip nginx git

# Create wavira directory
mkdir -p /opt/wavira
cd /opt/wavira

# Create Python virtual environment
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install paho-mqtt websockets

# Determine MQTT host
if [ -z "$MQTT_BROKER_HOST" ]; then
    MQTT_HOST="localhost"

    # Run local Mosquitto MQTT broker
    docker run -d \
        --name mosquitto \
        --restart unless-stopped \
        -p 1883:1883 \
        -v /opt/wavira/mosquitto/config:/mosquitto/config \
        -v /opt/wavira/mosquitto/data:/mosquitto/data \
        -v /opt/wavira/mosquitto/log:/mosquitto/log \
        eclipse-mosquitto

    # Create Mosquitto config
    mkdir -p /opt/wavira/mosquitto/config
    cat > /opt/wavira/mosquitto/config/mosquitto.conf << 'MQTTCONF'
listener 1883 0.0.0.0
allow_anonymous true
persistence true
persistence_location /mosquitto/data/
log_dest stdout
MQTTCONF

    # Restart Mosquitto with config
    docker restart mosquitto
else
    MQTT_HOST="$MQTT_BROKER_HOST"
fi

# Download mqtt_ws_bridge.py from GitHub
cat > /opt/wavira/mqtt_ws_bridge.py << 'BRIDGESCRIPT'
#!/usr/bin/env python3
"""
MQTT-WebSocket Bridge for Wavira CSI Dashboard
Bridges MQTT CSI data to WebSocket clients with history persistence.
"""

import asyncio
import json
import argparse
import logging
import sqlite3
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Set
import threading

import paho.mqtt.client as mqtt
import websockets

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class ZoneHistoryDB:
    """SQLite database for zone history persistence."""

    def __init__(self, db_path: str = "history.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.lock = threading.Lock()
        self._init_db()
        logger.info(f"Zone history database initialized: {db_path}")

    def _init_db(self):
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS csi_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    device_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    rssi INTEGER,
                    amplitude REAL,
                    breath_ratio REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_device_ts ON csi_history(device_id, timestamp)')
            self.conn.commit()

    def add_entry(self, device_id: str, timestamp: float, rssi: int, amplitude: float, breath_ratio: float):
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute(
                'INSERT INTO csi_history (device_id, timestamp, rssi, amplitude, breath_ratio) VALUES (?, ?, ?, ?, ?)',
                (device_id, timestamp, rssi, amplitude, breath_ratio)
            )
            self.conn.commit()

    def get_history(self, device_id: str, limit: int = 1200) -> List[Dict]:
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute(
                'SELECT timestamp, rssi, amplitude, breath_ratio FROM csi_history WHERE device_id = ? ORDER BY timestamp DESC LIMIT ?',
                (device_id, limit)
            )
            rows = cursor.fetchall()
            return [
                {"ts": row[0], "rssi": row[1], "amp": row[2], "breath": row[3]}
                for row in reversed(rows)
            ]

    def cleanup_old(self, hours: int = 24):
        with self.lock:
            cutoff = datetime.now() - timedelta(hours=hours)
            cursor = self.conn.cursor()
            cursor.execute('DELETE FROM csi_history WHERE created_at < ?', (cutoff,))
            self.conn.commit()


class MQTTWebSocketBridge:
    def __init__(self, mqtt_host: str, mqtt_port: int, ws_host: str, ws_port: int):
        self.mqtt_host = mqtt_host
        self.mqtt_port = mqtt_port
        self.ws_host = ws_host
        self.ws_port = ws_port

        self.websocket_clients: Set[websockets.WebSocketServerProtocol] = set()
        self.devices: Dict[str, Dict] = {}
        self.history_db = ZoneHistoryDB()

        # MQTT client setup
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_connect = self._on_mqtt_connect
        self.mqtt_client.on_message = self._on_mqtt_message
        self.mqtt_client.on_disconnect = self._on_mqtt_disconnect

        self.loop = None

    def _on_mqtt_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logger.info(f"Connected to MQTT broker at {self.mqtt_host}:{self.mqtt_port}")
            client.subscribe("wavira/#")
            logger.info("Subscribed to wavira/#")
        else:
            logger.error(f"MQTT connection failed with code {rc}")

    def _on_mqtt_disconnect(self, client, userdata, rc):
        logger.warning(f"MQTT disconnected with code {rc}")

    def _on_mqtt_message(self, client, userdata, msg):
        try:
            topic = msg.topic
            payload = json.loads(msg.payload.decode())

            if topic.startswith("wavira/csi/"):
                device_id = topic.split("/")[-1]
                self._process_csi_data(device_id, payload)
            elif topic.startswith("wavira/device/") and topic.endswith("/status"):
                device_id = topic.split("/")[2]
                logger.info(f"Device {device_id} status: {payload}")
                if device_id not in self.devices:
                    self.devices[device_id] = {"status": "online", "last_seen": datetime.now()}
                    logger.info(f"New device registered: {device_id}")
        except Exception as e:
            logger.error(f"Error processing MQTT message: {e}")

    def _process_csi_data(self, device_id: str, data: Dict):
        try:
            timestamp = data.get("ts", 0)
            rssi = data.get("rssi", 0)
            csi_data = data.get("data", [])

            # Calculate amplitude
            if csi_data:
                amplitudes = []
                for i in range(0, len(csi_data) - 1, 2):
                    real = csi_data[i]
                    imag = csi_data[i + 1] if i + 1 < len(csi_data) else 0
                    amp = (real ** 2 + imag ** 2) ** 0.5
                    amplitudes.append(amp)
                avg_amp = sum(amplitudes) / len(amplitudes) if amplitudes else 0
            else:
                amplitudes = []
                avg_amp = 0

            # Breath ratio (placeholder calculation)
            breath_ratio = min(avg_amp / 50.0, 1.0) if avg_amp > 0 else 0

            # Store in history
            self.history_db.add_entry(device_id, timestamp, rssi, avg_amp, breath_ratio)

            # Update device info
            self.devices[device_id] = {
                "status": "online",
                "last_seen": datetime.now(),
                "rssi": rssi,
                "amplitude": avg_amp
            }

            # Broadcast to WebSocket clients
            ws_data = {
                "type": "csi",
                "device_id": device_id,
                "ts": timestamp,
                "rssi": rssi,
                "amps": amplitudes,
                "avg_amp": avg_amp,
                "breath": {
                    "breathing": breath_ratio > 0.1,
                    "breath_ratio": breath_ratio,
                    "hold_duration": 0
                }
            }

            if self.loop and self.websocket_clients:
                asyncio.run_coroutine_threadsafe(
                    self._broadcast(json.dumps(ws_data)),
                    self.loop
                )
        except Exception as e:
            logger.error(f"Error processing CSI data: {e}")

    async def _broadcast(self, message: str):
        if self.websocket_clients:
            await asyncio.gather(
                *[client.send(message) for client in self.websocket_clients],
                return_exceptions=True
            )

    async def _handle_websocket(self, websocket: websockets.WebSocketServerProtocol, path: str):
        logger.info(f"connection open")
        logger.info(f"WebSocket client connected: {websocket.remote_address}")
        self.websocket_clients.add(websocket)

        try:
            # Send history for all known devices
            for device_id in self.devices:
                history = self.history_db.get_history(device_id)
                if history:
                    await websocket.send(json.dumps({
                        "type": "history",
                        "device_id": device_id,
                        "data": history
                    }))
                    logger.info(f"Sent {len(history)} history entries for {device_id}")

            # Keep connection alive
            async for message in websocket:
                pass
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.websocket_clients.discard(websocket)
            logger.info(f"WebSocket client disconnected: {websocket.remote_address}")

    async def run(self):
        self.loop = asyncio.get_event_loop()

        # Start MQTT client in background thread
        self.mqtt_client.connect_async(self.mqtt_host, self.mqtt_port)
        self.mqtt_client.loop_start()
        logger.info("MQTT client started")

        # Start WebSocket server
        server = await websockets.serve(
            self._handle_websocket,
            self.ws_host,
            self.ws_port
        )
        logger.info(f"WebSocket server started on ws://{self.ws_host}:{self.ws_port}")

        # Periodic cleanup
        async def cleanup_task():
            while True:
                await asyncio.sleep(3600)
                self.history_db.cleanup_old(24)

        asyncio.create_task(cleanup_task())

        await server.wait_closed()


def main():
    parser = argparse.ArgumentParser(description="MQTT-WebSocket Bridge for Wavira")
    parser.add_argument("--mqtt-host", default="localhost", help="MQTT broker host")
    parser.add_argument("--mqtt-port", type=int, default=1883, help="MQTT broker port")
    parser.add_argument("--ws-host", default="0.0.0.0", help="WebSocket host")
    parser.add_argument("--ws-port", type=int, default=8765, help="WebSocket port")
    args = parser.parse_args()

    logger.info("Starting MQTT-WebSocket Bridge")
    logger.info(f"  MQTT: {args.mqtt_host}:{args.mqtt_port}")
    logger.info(f"  WebSocket: ws://{args.ws_host}:{args.ws_port}")

    bridge = MQTTWebSocketBridge(
        args.mqtt_host, args.mqtt_port,
        args.ws_host, args.ws_port
    )

    asyncio.run(bridge.run())


if __name__ == "__main__":
    main()
BRIDGESCRIPT

chmod +x /opt/wavira/mqtt_ws_bridge.py

# Create systemd service for WebSocket bridge
cat > /etc/systemd/system/wavira-bridge.service << SERVICEEOF
[Unit]
Description=Wavira MQTT-WebSocket Bridge
After=network.target docker.service

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/opt/wavira
ExecStart=/opt/wavira/venv/bin/python /opt/wavira/mqtt_ws_bridge.py --mqtt-host $MQTT_HOST --ws-port 8765
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
SERVICEEOF

# Download dashboard
mkdir -p /opt/wavira/dashboard
cat > /opt/wavira/dashboard/index.html << 'DASHBOARDHTML'
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Wavira CSI Monitor</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: #1a1a2e; color: #eee; font-family: system-ui, sans-serif; }
.container { max-width: 1400px; margin: 0 auto; padding: 20px; }
header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
h1 { color: #00ff88; font-size: 24px; }
.status { display: flex; gap: 20px; align-items: center; }
.status-item { display: flex; align-items: center; gap: 8px; }
.dot { width: 12px; height: 12px; border-radius: 50%; background: #f44; }
.dot.connected { background: #0f8; }
.stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 20px; }
.stat-card { background: #16213e; padding: 20px; border-radius: 10px; text-align: center; }
.stat-value { font-size: 36px; font-weight: bold; color: #00ff88; }
.stat-label { color: #888; margin-top: 5px; }
.devices { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }
.device-card { background: #16213e; border-radius: 10px; padding: 20px; }
.device-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; }
.device-name { font-size: 18px; font-weight: bold; }
.device-status { display: flex; align-items: center; gap: 8px; }
.device-stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 15px; }
.device-stat { text-align: center; padding: 10px; background: #0d1b2a; border-radius: 5px; }
.device-stat-value { font-size: 20px; color: #00ff88; }
.device-stat-label { font-size: 12px; color: #888; }
canvas { width: 100%; height: 150px; background: #0d1b2a; border-radius: 5px; }
</style>
</head>
<body>
<div class="container">
    <header>
        <h1>Wavira CSI Monitor</h1>
        <div class="status">
            <div class="status-item">
                <div class="dot" id="statusDot"></div>
                <span id="statusText">Disconnected</span>
            </div>
            <div class="status-item">Devices: <span id="deviceCount">0</span></div>
        </div>
    </header>

    <div class="stats">
        <div class="stat-card">
            <div class="stat-value" id="totalDevices">0</div>
            <div class="stat-label">Total Devices</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" id="onlineDevices">0</div>
            <div class="stat-label">Online</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" id="avgRssi">--</div>
            <div class="stat-label">Avg RSSI</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" id="packetsPerSec">0</div>
            <div class="stat-label">Packets/sec</div>
        </div>
    </div>

    <div class="devices" id="devicesContainer"></div>
</div>

<script>
const devices = {};
let packetCount = 0;
let lastPacketTime = Date.now();

function connect() {
    const params = new URLSearchParams(window.location.search);
    const wsParam = params.get('ws');
    const wsUrl = wsParam ? 'ws://' + wsParam : 'ws://' + window.location.hostname + ':8765';

    console.log('Connecting to:', wsUrl);
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        document.getElementById('statusDot').classList.add('connected');
        document.getElementById('statusText').textContent = 'Connected';
    };

    ws.onclose = () => {
        document.getElementById('statusDot').classList.remove('connected');
        document.getElementById('statusText').textContent = 'Reconnecting...';
        setTimeout(connect, 2000);
    };

    ws.onmessage = (e) => {
        try {
            const data = JSON.parse(e.data);
            if (data.type === 'csi') {
                updateDevice(data);
                packetCount++;
            } else if (data.type === 'history') {
                console.log('Restoring ' + data.data.length + ' history entries for ' + data.device_id);
            }
        } catch (err) {}
    };
}

function updateDevice(data) {
    const id = data.device_id;
    if (!devices[id]) {
        devices[id] = { history: [] };
        createDeviceCard(id);
    }

    devices[id].rssi = data.rssi;
    devices[id].amplitude = data.avg_amp;
    devices[id].breathing = data.breath?.breathing;
    devices[id].lastSeen = Date.now();
    devices[id].history.push(data.avg_amp);
    if (devices[id].history.length > 100) devices[id].history.shift();

    updateDeviceCard(id);
    updateStats();
}

function createDeviceCard(id) {
    const container = document.getElementById('devicesContainer');
    const card = document.createElement('div');
    card.className = 'device-card';
    card.id = 'device-' + id;
    card.innerHTML = '<div class="device-header">' +
        '<span class="device-name">' + id + '</span>' +
        '<div class="device-status"><div class="dot connected"></div><span>Online</span></div></div>' +
        '<div class="device-stats">' +
        '<div class="device-stat"><div class="device-stat-value" id="' + id + '-rssi">--</div><div class="device-stat-label">RSSI</div></div>' +
        '<div class="device-stat"><div class="device-stat-value" id="' + id + '-amp">--</div><div class="device-stat-label">Amplitude</div></div>' +
        '<div class="device-stat"><div class="device-stat-value" id="' + id + '-breath">--</div><div class="device-stat-label">Breathing</div></div></div>' +
        '<canvas id="' + id + '-canvas"></canvas>';
    container.appendChild(card);
}

function updateDeviceCard(id) {
    const d = devices[id];
    document.getElementById(id + '-rssi').textContent = d.rssi + ' dBm';
    document.getElementById(id + '-amp').textContent = d.amplitude.toFixed(1);
    document.getElementById(id + '-breath').textContent = d.breathing ? 'Yes' : 'No';

    const canvas = document.getElementById(id + '-canvas');
    const ctx = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;

    ctx.fillStyle = '#0d1b2a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    if (d.history.length > 1) {
        const max = Math.max(...d.history, 50);
        ctx.strokeStyle = '#00ff88';
        ctx.lineWidth = 2;
        ctx.beginPath();
        d.history.forEach((v, i) => {
            const x = (i / (d.history.length - 1)) * canvas.width;
            const y = canvas.height - (v / max) * canvas.height * 0.9;
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        });
        ctx.stroke();
    }
}

function updateStats() {
    const ids = Object.keys(devices);
    const online = ids.filter(id => Date.now() - devices[id].lastSeen < 5000);
    const rssiValues = online.map(id => devices[id].rssi).filter(v => v);
    const avgRssi = rssiValues.length ? Math.round(rssiValues.reduce((a, b) => a + b) / rssiValues.length) : '--';

    document.getElementById('totalDevices').textContent = ids.length;
    document.getElementById('onlineDevices').textContent = online.length;
    document.getElementById('deviceCount').textContent = ids.length;
    document.getElementById('avgRssi').textContent = avgRssi;

    const now = Date.now();
    const pps = Math.round(packetCount / ((now - lastPacketTime) / 1000));
    document.getElementById('packetsPerSec').textContent = pps;
    packetCount = 0;
    lastPacketTime = now;
}

setInterval(updateStats, 1000);
connect();
</script>
</body>
</html>
DASHBOARDHTML

# Configure nginx
cat > /etc/nginx/conf.d/wavira.conf << 'NGINXCONF'
server {
    listen 80;
    server_name _;

    root /opt/wavira/dashboard;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    location /ws {
        proxy_pass http://127.0.0.1:8765;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 86400;
    }
}
NGINXCONF

# Set permissions
chown -R ec2-user:ec2-user /opt/wavira

# Enable and start services
systemctl daemon-reload
systemctl enable nginx
systemctl start nginx
systemctl enable wavira-bridge
sleep 5  # Wait for Docker/Mosquitto to be ready
systemctl start wavira-bridge

echo "Wavira server setup completed at $(date)"

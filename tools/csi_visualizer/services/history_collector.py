#!/usr/bin/env python3
"""
History Collector Service for Wavira

Subscribes to all wavira/# MQTT topics, stores data in SQLite,
and provides a REST API for historical queries.

MQTT Topics:
    Subscribe: wavira/#

REST API Endpoints:
    GET /api/v1/health          - Health check
    GET /api/v1/devices         - List known devices
    GET /api/v1/history/{device_id}    - Get device history
    GET /api/v1/history/{device_id}/hourly - Hourly aggregation
    GET /api/v1/zones           - Zone statistics
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import sqlite3
import sys
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import paho.mqtt.client as mqtt
from fastapi import Depends, FastAPI, HTTPException, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from pydantic import BaseModel, Field

from security import (
    SecurityConfig,
    load_security_config,
    APIKeyAuth,
    RateLimitMiddleware,
    SecurityHeadersMiddleware,
    InputValidator,
    get_mqtt_credentials,
    mask_password,
)
from device_manager import (
    DeviceManager,
    DeviceManagerConfig,
    MQTTCredentials,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class HistoryDatabase:
    """SQLite database for CSI history."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get a new database connection."""
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _init_db(self):
        """Initialize database schema."""
        with self.lock:
            conn = self._get_conn()
            cursor = conn.cursor()

            # CSI data history
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS csi_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    device_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    rssi INTEGER,
                    avg_amplitude REAL,
                    variance REAL,
                    present INTEGER,
                    breath_ratio REAL,
                    breathing INTEGER,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cursor.execute(
                'CREATE INDEX IF NOT EXISTS idx_csi_device_ts '
                'ON csi_history(device_id, timestamp)'
            )

            # Device status
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS devices (
                    device_id TEXT PRIMARY KEY,
                    last_seen REAL,
                    rssi INTEGER,
                    online INTEGER DEFAULT 1,
                    zone TEXT DEFAULT 'default',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Zone history (hourly aggregation)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS zone_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    datetime TEXT,
                    zone TEXT,
                    present_count INTEGER,
                    device_count INTEGER,
                    avg_amplitude REAL
                )
            ''')
            cursor.execute(
                'CREATE INDEX IF NOT EXISTS idx_zone_ts '
                'ON zone_history(timestamp)'
            )

            conn.commit()
            conn.close()
            logger.info(f"Database initialized: {self.db_path}")

    def add_csi_entry(
        self,
        device_id: str,
        timestamp: float,
        rssi: int,
        avg_amplitude: float,
        variance: float,
        present: bool,
        breath_ratio: float,
        breathing: bool
    ):
        """Add CSI history entry."""
        with self.lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO csi_history
                (device_id, timestamp, rssi, avg_amplitude, variance, present, breath_ratio, breathing)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (device_id, timestamp, rssi, avg_amplitude, variance,
                  1 if present else 0, breath_ratio, 1 if breathing else 0))

            # Update device status
            cursor.execute('''
                INSERT INTO devices (device_id, last_seen, rssi, online)
                VALUES (?, ?, ?, 1)
                ON CONFLICT(device_id) DO UPDATE SET
                    last_seen = excluded.last_seen,
                    rssi = excluded.rssi,
                    online = 1,
                    updated_at = CURRENT_TIMESTAMP
            ''', (device_id, timestamp, rssi))

            conn.commit()
            conn.close()

    def get_device_history(
        self,
        device_id: str,
        limit: int = 1000,
        since: Optional[float] = None
    ) -> List[Dict]:
        """Get device history."""
        with self.lock:
            conn = self._get_conn()
            cursor = conn.cursor()

            if since:
                cursor.execute('''
                    SELECT timestamp, rssi, avg_amplitude, variance, present, breath_ratio, breathing
                    FROM csi_history
                    WHERE device_id = ? AND timestamp > ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (device_id, since, limit))
            else:
                cursor.execute('''
                    SELECT timestamp, rssi, avg_amplitude, variance, present, breath_ratio, breathing
                    FROM csi_history
                    WHERE device_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (device_id, limit))

            rows = cursor.fetchall()
            conn.close()

            return [
                {
                    "timestamp": row[0],
                    "rssi": row[1],
                    "avg_amplitude": row[2],
                    "variance": row[3],
                    "present": bool(row[4]),
                    "breath_ratio": row[5],
                    "breathing": bool(row[6]),
                }
                for row in reversed(rows)
            ]

    def get_hourly_summary(
        self,
        device_id: Optional[str] = None,
        hours: int = 24
    ) -> List[Dict]:
        """Get hourly aggregated data."""
        with self.lock:
            conn = self._get_conn()
            cursor = conn.cursor()

            since = time.time() - (hours * 3600)

            if device_id:
                cursor.execute('''
                    SELECT
                        strftime('%Y-%m-%d %H:00', datetime(timestamp, 'unixepoch', 'localtime')) as hour,
                        device_id,
                        AVG(avg_amplitude) as avg_amp,
                        AVG(variance) as avg_var,
                        SUM(present) * 100.0 / COUNT(*) as presence_pct,
                        COUNT(*) as sample_count
                    FROM csi_history
                    WHERE device_id = ? AND timestamp > ?
                    GROUP BY hour, device_id
                    ORDER BY hour DESC
                ''', (device_id, since))
            else:
                cursor.execute('''
                    SELECT
                        strftime('%Y-%m-%d %H:00', datetime(timestamp, 'unixepoch', 'localtime')) as hour,
                        device_id,
                        AVG(avg_amplitude) as avg_amp,
                        AVG(variance) as avg_var,
                        SUM(present) * 100.0 / COUNT(*) as presence_pct,
                        COUNT(*) as sample_count
                    FROM csi_history
                    WHERE timestamp > ?
                    GROUP BY hour, device_id
                    ORDER BY hour DESC
                ''', (since,))

            rows = cursor.fetchall()
            conn.close()

            return [
                {
                    "hour": row[0],
                    "device_id": row[1],
                    "avg_amplitude": round(row[2], 2) if row[2] else 0,
                    "avg_variance": round(row[3], 2) if row[3] else 0,
                    "presence_pct": round(row[4], 1) if row[4] else 0,
                    "sample_count": row[5],
                }
                for row in rows
            ]

    def get_devices(self) -> List[Dict]:
        """Get list of known devices."""
        with self.lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute('''
                SELECT device_id, last_seen, rssi, online, zone
                FROM devices
                ORDER BY last_seen DESC
            ''')
            rows = cursor.fetchall()
            conn.close()

            now = time.time()
            return [
                {
                    "device_id": row[0],
                    "last_seen": row[1],
                    "rssi": row[2],
                    "online": now - row[1] < 60 if row[1] else False,
                    "zone": row[3],
                }
                for row in rows
            ]

    def cleanup_old_data(self, retention_days: int = 7):
        """Delete old data."""
        with self.lock:
            cutoff = time.time() - (retention_days * 24 * 3600)
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute('DELETE FROM csi_history WHERE timestamp < ?', (cutoff,))
            deleted = cursor.rowcount
            conn.commit()
            conn.close()
            if deleted > 0:
                logger.info(f"Cleaned up {deleted} old history records")


class MQTTCollector:
    """
    MQTT subscriber for collecting data and managing device status.

    Issue #24: オンライン/オフライン検知のMQTT対応
    - Handles Last Will messages for offline detection
    - Tracks device online/offline status in real-time
    - Publishes status updates for dashboard notifications
    """

    def __init__(
        self,
        mqtt_host: str,
        mqtt_port: int,
        db: HistoryDatabase,
        dm: Optional[DeviceManager] = None,
    ):
        self.mqtt_host = mqtt_host
        self.mqtt_port = mqtt_port
        self.db = db
        self.device_manager = dm
        self.running = False

        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        """MQTT connection callback."""
        if reason_code == 0:
            logger.info(f"MQTT connected to {self.mqtt_host}:{self.mqtt_port}")
            # Subscribe to all wavira topics
            client.subscribe("wavira/analysis/#", qos=0)
            client.subscribe("wavira/csi/#", qos=0)
            # Device status topics (QoS 1 for reliability)
            client.subscribe("wavira/device/+/status", qos=1)
            client.subscribe("wavira/device/+/will", qos=1)
            logger.info("Subscribed to wavira/# topics including device status")
        else:
            logger.error(f"MQTT connection failed: {reason_code}")

    def _on_disconnect(self, client, userdata, flags, reason_code, properties):
        """MQTT disconnection callback."""
        logger.warning(f"MQTT disconnected: {reason_code}")

    def _on_message(self, client, userdata, msg):
        """MQTT message callback."""
        try:
            topic = msg.topic
            payload = json.loads(msg.payload.decode())

            # Handle device status messages (Issue #24)
            if "/status" in topic or "/will" in topic:
                self._handle_device_status(topic, payload)
                return

            # Store analysis results
            if topic.startswith("wavira/analysis/") and not topic.endswith("/presence") and not topic.endswith("/breathing"):
                device_id = payload.get("device_id")
                if device_id:
                    self.db.add_csi_entry(
                        device_id=device_id,
                        timestamp=payload.get("timestamp", time.time()),
                        rssi=payload.get("rssi", -100),
                        avg_amplitude=payload.get("avg_amplitude", 0),
                        variance=payload.get("variance", 0),
                        present=payload.get("present", False),
                        breath_ratio=payload.get("breath_ratio", 0),
                        breathing=payload.get("breathing", False),
                    )
                    # Update device last_seen on any data received
                    if self.device_manager:
                        self.device_manager.update_device_status(
                            device_id,
                            online=True,
                            last_seen=time.time(),
                        )

        except json.JSONDecodeError:
            pass
        except Exception as e:
            logger.error(f"Error processing message: {e}")

    def _handle_device_status(self, topic: str, payload: dict):
        """
        Handle device online/offline status messages.

        Topics:
            wavira/device/{device_id}/status - Device announces online
            wavira/device/{device_id}/will - Last Will (device offline)
        """
        # Extract device_id from topic
        # Format: wavira/device/{device_id}/status or wavira/device/{device_id}/will
        parts = topic.split("/")
        if len(parts) < 4:
            return

        device_id = parts[2]
        message_type = parts[3]  # 'status' or 'will'
        status = payload.get("status", "unknown")
        timestamp = payload.get("timestamp", time.time())

        if message_type == "status":
            # Device announced its status
            is_online = status == "online"
            logger.info(f"Device {device_id} status: {status}")

            if self.device_manager:
                self.device_manager.update_device_status(
                    device_id,
                    online=is_online,
                    last_seen=timestamp if is_online else None,
                )

            # Publish status update for dashboard
            self._publish_status_update(device_id, is_online, timestamp)

        elif message_type == "will":
            # Last Will message - device went offline unexpectedly
            logger.warning(f"Device {device_id} offline (Last Will received)")

            if self.device_manager:
                self.device_manager.update_device_status(
                    device_id,
                    online=False,
                    last_seen=timestamp,
                )

            # Publish offline notification
            self._publish_status_update(device_id, False, timestamp)

    def _publish_status_update(self, device_id: str, online: bool, timestamp: float):
        """Publish device status update for real-time dashboard notifications."""
        status_payload = json.dumps({
            "device_id": device_id,
            "online": online,
            "status": "online" if online else "offline",
            "timestamp": timestamp,
            "source": "mqtt_will" if not online else "mqtt_status",
        })

        # Publish to a topic that dashboards can subscribe to
        self.client.publish(
            f"wavira/status/{device_id}",
            status_payload,
            qos=1,
            retain=True,  # Retain so new subscribers get current status
        )

    def start(self):
        """Start MQTT collector in background thread."""
        self.running = True

        def run_mqtt():
            try:
                self.client.connect(self.mqtt_host, self.mqtt_port, keepalive=60)
                self.client.loop_forever()
            except Exception as e:
                logger.error(f"MQTT error: {e}")

        thread = threading.Thread(target=run_mqtt, daemon=True)
        thread.start()
        logger.info("MQTT collector started")

    def stop(self):
        """Stop MQTT collector."""
        self.running = False
        self.client.disconnect()


# Global instances
db: Optional[HistoryDatabase] = None
mqtt_collector: Optional[MQTTCollector] = None
security_config: Optional[SecurityConfig] = None
api_key_auth: Optional[APIKeyAuth] = None
device_manager: Optional[DeviceManager] = None


# =============================================================================
# Pydantic Models for API
# =============================================================================


class DeviceRegistrationRequest(BaseModel):
    """Request model for device registration."""
    device_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        pattern=r"^[a-zA-Z0-9_-]+$",
        description="Unique device identifier"
    )
    zone: str = Field(
        default="default",
        max_length=64,
        description="Device zone/location"
    )
    generate_mqtt: bool = Field(
        default=True,
        description="Whether to generate MQTT credentials"
    )


class DeviceRegistrationResponse(BaseModel):
    """Response model for device registration."""
    device_id: str
    api_key: str
    mqtt_credentials: Optional[dict] = None
    message: str


class MQTTCredentialsResponse(BaseModel):
    """Response model for MQTT credentials."""
    device_id: str
    username: str
    password: str
    client_id: str
    message: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global db, mqtt_collector, security_config, api_key_auth, device_manager

    # Startup
    db_path = os.environ.get("DB_PATH", "history.db")
    devices_db_path = os.environ.get("DEVICES_DB_PATH", "devices.db")
    mqtt_host = os.environ.get("MQTT_HOST", "localhost")
    mqtt_port = int(os.environ.get("MQTT_PORT", 1883))

    # Initialize security
    security_config = load_security_config()
    api_key_auth = APIKeyAuth(security_config)
    logger.info(f"Security: API key required={security_config.require_api_key}, "
                f"rate limit={security_config.rate_limit_requests}/{security_config.rate_limit_window}s")

    # Initialize device manager
    device_manager_config = DeviceManagerConfig(db_path=devices_db_path)
    device_manager = DeviceManager(device_manager_config)
    logger.info(f"Device manager initialized: {devices_db_path}")

    db = HistoryDatabase(db_path)
    mqtt_collector = MQTTCollector(mqtt_host, mqtt_port, db, device_manager)
    mqtt_collector.start()

    # Start cleanup task
    async def cleanup_task():
        while True:
            await asyncio.sleep(3600)  # Every hour
            if db:
                db.cleanup_old_data()

    cleanup_handle = asyncio.create_task(cleanup_task())

    yield

    # Shutdown
    cleanup_handle.cancel()
    if mqtt_collector:
        mqtt_collector.stop()


# FastAPI app
app = FastAPI(
    title="Wavira History API",
    description="REST API for Wavira CSI historical data",
    version="1.0.0",
    lifespan=lifespan
)


def setup_middleware(app: FastAPI, config: SecurityConfig):
    """Setup security middleware."""
    # Security headers (added first, runs last)
    if config.add_security_headers:
        app.add_middleware(SecurityHeadersMiddleware)

    # Rate limiting
    if config.rate_limit_enabled:
        app.add_middleware(RateLimitMiddleware, config=config)

    # CORS (added last, runs first)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=config.cors_allow_credentials,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["X-API-Key", "Content-Type", "Authorization"],
    )


# Setup middleware with default config (will be reconfigured in lifespan if needed)
_default_config = SecurityConfig()
setup_middleware(app, _default_config)


def get_api_key_auth():
    """Get API key auth dependency."""
    if api_key_auth:
        return api_key_auth
    # Return a no-op auth if not configured
    return APIKeyAuth(SecurityConfig(require_api_key=False))


@app.get("/api/v1/health")
async def health():
    """Health check endpoint (no auth required)."""
    return {
        "status": "ok",
        "timestamp": time.time(),
        "security": {
            "api_key_required": security_config.require_api_key if security_config else False,
            "rate_limit_enabled": security_config.rate_limit_enabled if security_config else True,
        }
    }


@app.get("/api/v1/devices")
async def get_devices(
    _api_key: str = Depends(get_api_key_auth),
):
    """Get list of known devices."""
    if not db:
        raise HTTPException(status_code=503, detail="Database not initialized")
    return db.get_devices()


@app.get("/api/v1/history/{device_id}")
async def get_history(
    device_id: str = Path(..., min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$"),
    limit: int = Query(default=1000, ge=1, le=10000),
    since: Optional[float] = Query(default=None, ge=0),
    _api_key: str = Depends(get_api_key_auth),
):
    """Get device history."""
    if not db:
        raise HTTPException(status_code=503, detail="Database not initialized")
    # Additional validation
    device_id = InputValidator.device_id(device_id)
    return db.get_device_history(device_id, limit, since)


@app.get("/api/v1/history/{device_id}/hourly")
async def get_hourly_summary(
    device_id: str = Path(..., min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$"),
    hours: int = Query(default=24, ge=1, le=168),
    _api_key: str = Depends(get_api_key_auth),
):
    """Get hourly aggregated data for a device."""
    if not db:
        raise HTTPException(status_code=503, detail="Database not initialized")
    device_id = InputValidator.device_id(device_id)
    return db.get_hourly_summary(device_id, hours)


@app.get("/api/v1/hourly")
async def get_all_hourly_summary(
    hours: int = Query(default=24, ge=1, le=168),
    _api_key: str = Depends(get_api_key_auth),
):
    """Get hourly aggregated data for all devices."""
    if not db:
        raise HTTPException(status_code=503, detail="Database not initialized")
    return db.get_hourly_summary(None, hours)


# =============================================================================
# Device Management API (Issue #23)
# =============================================================================


@app.post("/api/v1/admin/devices", response_model=DeviceRegistrationResponse)
async def register_device(
    request: DeviceRegistrationRequest,
    _api_key: str = Depends(get_api_key_auth),
):
    """
    Register a new device and generate credentials.

    This endpoint creates a new device with:
    - API key for REST API authentication
    - MQTT credentials (username, password, client_id) for MQTT authentication

    IMPORTANT: Save the returned credentials securely.
    The API key and MQTT password cannot be retrieved later.
    """
    if not device_manager:
        raise HTTPException(status_code=503, detail="Device manager not initialized")

    try:
        device, api_key, mqtt_creds = device_manager.register_device(
            device_id=request.device_id,
            zone=request.zone,
            generate_mqtt=request.generate_mqtt,
        )

        response = DeviceRegistrationResponse(
            device_id=device.device_id,
            api_key=api_key,
            mqtt_credentials=mqtt_creds.to_dict(include_password=True) if mqtt_creds else None,
            message="Device registered successfully. Save credentials securely.",
        )
        return response

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/v1/admin/devices")
async def list_managed_devices(
    zone: Optional[str] = Query(default=None, max_length=64),
    online_only: bool = Query(default=False),
    _api_key: str = Depends(get_api_key_auth),
):
    """List all registered devices with their status."""
    if not device_manager:
        raise HTTPException(status_code=503, detail="Device manager not initialized")

    devices = device_manager.list_devices(zone=zone, online_only=online_only)
    return [device.to_dict() for device in devices]


@app.get("/api/v1/admin/devices/{device_id}")
async def get_managed_device(
    device_id: str = Path(..., min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$"),
    _api_key: str = Depends(get_api_key_auth),
):
    """Get a specific device's information."""
    if not device_manager:
        raise HTTPException(status_code=503, detail="Device manager not initialized")

    device_id = InputValidator.device_id(device_id)
    device = device_manager.get_device(device_id)

    if not device:
        raise HTTPException(status_code=404, detail=f"Device not found: {device_id}")

    return device.to_dict()


@app.post("/api/v1/admin/devices/{device_id}/rotate-mqtt", response_model=MQTTCredentialsResponse)
async def rotate_mqtt_credentials(
    device_id: str = Path(..., min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$"),
    _api_key: str = Depends(get_api_key_auth),
):
    """
    Rotate MQTT credentials for a device.

    This generates new MQTT username, password, and client_id.
    The old credentials will be invalidated.

    IMPORTANT: Update the device firmware with new credentials.
    """
    if not device_manager:
        raise HTTPException(status_code=503, detail="Device manager not initialized")

    device_id = InputValidator.device_id(device_id)
    creds = device_manager.rotate_mqtt_credentials(device_id)

    if not creds:
        raise HTTPException(status_code=404, detail=f"Device not found: {device_id}")

    return MQTTCredentialsResponse(
        device_id=device_id,
        username=creds.username,
        password=creds.password,
        client_id=creds.client_id,
        message="MQTT credentials rotated. Update device firmware with new credentials.",
    )


@app.delete("/api/v1/admin/devices/{device_id}")
async def delete_device(
    device_id: str = Path(..., min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$"),
    _api_key: str = Depends(get_api_key_auth),
):
    """Delete a device and its credentials."""
    if not device_manager:
        raise HTTPException(status_code=503, detail="Device manager not initialized")

    device_id = InputValidator.device_id(device_id)
    deleted = device_manager.delete_device(device_id)

    if not deleted:
        raise HTTPException(status_code=404, detail=f"Device not found: {device_id}")

    return {"message": f"Device deleted: {device_id}"}


@app.get("/api/v1/admin/devices-without-mqtt")
async def get_devices_without_mqtt(
    _api_key: str = Depends(get_api_key_auth),
):
    """Get devices that don't have MQTT credentials (for migration)."""
    if not device_manager:
        raise HTTPException(status_code=503, detail="Device manager not initialized")

    devices = device_manager.get_devices_without_mqtt()
    return [device.to_dict() for device in devices]


@app.post("/api/v1/admin/migrate-mqtt")
async def migrate_devices_to_mqtt(
    _api_key: str = Depends(get_api_key_auth),
):
    """
    Add MQTT credentials to all devices that don't have them.

    This is useful for migrating existing devices to MQTT authentication.

    IMPORTANT: Save the returned credentials securely.
    """
    if not device_manager:
        raise HTTPException(status_code=503, detail="Device manager not initialized")

    results = device_manager.migrate_all_devices_to_mqtt()

    if not results:
        return {"message": "No devices need migration", "devices": []}

    return {
        "message": f"Migrated {len(results)} devices",
        "devices": [
            {
                "device_id": device_id,
                "mqtt_credentials": creds.to_dict(include_password=True),
            }
            for device_id, creds in results
        ],
    }


# =============================================================================
# Device Status API (Issue #24)
# =============================================================================


@app.get("/api/v1/status")
async def get_all_device_status(
    _api_key: str = Depends(get_api_key_auth),
):
    """
    Get online/offline status of all devices.

    Returns real-time device status based on MQTT Last Will mechanism.
    """
    if not device_manager:
        raise HTTPException(status_code=503, detail="Device manager not initialized")

    devices = device_manager.list_devices()
    return {
        "timestamp": time.time(),
        "devices": [
            {
                "device_id": d.device_id,
                "online": d.online,
                "last_seen": d.last_seen,
                "zone": d.zone,
            }
            for d in devices
        ],
        "summary": {
            "total": len(devices),
            "online": sum(1 for d in devices if d.online),
            "offline": sum(1 for d in devices if not d.online),
        },
    }


@app.get("/api/v1/status/{device_id}")
async def get_device_status(
    device_id: str = Path(..., min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$"),
    _api_key: str = Depends(get_api_key_auth),
):
    """
    Get online/offline status of a specific device.
    """
    if not device_manager:
        raise HTTPException(status_code=503, detail="Device manager not initialized")

    device_id = InputValidator.device_id(device_id)
    device = device_manager.get_device(device_id)

    if not device:
        raise HTTPException(status_code=404, detail=f"Device not found: {device_id}")

    return {
        "device_id": device.device_id,
        "online": device.online,
        "last_seen": device.last_seen,
        "zone": device.zone,
        "has_mqtt_credentials": device.has_mqtt_credentials,
    }


@app.get("/api/v1/status/online")
async def get_online_devices(
    _api_key: str = Depends(get_api_key_auth),
):
    """Get list of currently online devices."""
    if not device_manager:
        raise HTTPException(status_code=503, detail="Device manager not initialized")

    devices = device_manager.list_devices(online_only=True)
    return {
        "timestamp": time.time(),
        "count": len(devices),
        "devices": [
            {
                "device_id": d.device_id,
                "last_seen": d.last_seen,
                "zone": d.zone,
            }
            for d in devices
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="History Collector Service")
    parser.add_argument(
        "--mqtt-host",
        default=os.environ.get("MQTT_HOST", "localhost"),
        help="MQTT broker host"
    )
    parser.add_argument(
        "--mqtt-port",
        type=int,
        default=int(os.environ.get("MQTT_PORT", 1883)),
        help="MQTT broker port"
    )
    parser.add_argument(
        "--api-port",
        type=int,
        default=int(os.environ.get("API_PORT", 8080)),
        help="REST API port"
    )
    parser.add_argument(
        "--db-path",
        default=os.environ.get("DB_PATH", "history.db"),
        help="SQLite database path"
    )
    args = parser.parse_args()

    # Set environment variables for lifespan handler
    os.environ["MQTT_HOST"] = args.mqtt_host
    os.environ["MQTT_PORT"] = str(args.mqtt_port)
    os.environ["DB_PATH"] = args.db_path

    logger.info(f"Starting History Collector Service")
    logger.info(f"  MQTT: {args.mqtt_host}:{args.mqtt_port}")
    logger.info(f"  API: http://0.0.0.0:{args.api_port}")
    logger.info(f"  Database: {args.db_path}")

    uvicorn.run(app, host="0.0.0.0", port=args.api_port, log_level="info")


if __name__ == "__main__":
    main()

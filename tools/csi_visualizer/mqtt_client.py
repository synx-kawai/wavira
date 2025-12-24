#!/usr/bin/env python3
"""
MQTT Client for Wavira CSI Data Reception

Handles MQTT subscription and message processing for CSI data from ESP32 devices.
"""

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

try:
    import paho.mqtt.client as mqtt
    from paho.mqtt.enums import CallbackAPIVersion
    PAHO_V2 = True
except ImportError:
    import paho.mqtt.client as mqtt
    PAHO_V2 = False

logger = logging.getLogger(__name__)


@dataclass
class MQTTConfig:
    """MQTT client configuration"""

    broker_host: str = "localhost"
    broker_port: int = 1883
    username: Optional[str] = None
    password: Optional[str] = None
    client_id: str = "wavira_server"
    keepalive: int = 60
    reconnect_delay_min: float = 1.0
    reconnect_delay_max: float = 120.0
    qos: int = 1

    # TLS settings
    tls_enabled: bool = False
    tls_ca_certs: Optional[str] = None
    tls_certfile: Optional[str] = None
    tls_keyfile: Optional[str] = None

    @classmethod
    def from_env(cls) -> "MQTTConfig":
        """Create config from environment variables"""
        return cls(
            broker_host=os.environ.get("MQTT_BROKER_HOST", "localhost"),
            broker_port=int(os.environ.get("MQTT_BROKER_PORT", "1883")),
            username=os.environ.get("MQTT_USERNAME"),
            password=os.environ.get("MQTT_PASSWORD"),
            client_id=os.environ.get("MQTT_CLIENT_ID", "wavira_server"),
            keepalive=int(os.environ.get("MQTT_KEEPALIVE", "60")),
            qos=int(os.environ.get("MQTT_QOS", "1")),
            tls_enabled=os.environ.get("MQTT_TLS_ENABLED", "").lower() == "true",
            tls_ca_certs=os.environ.get("MQTT_TLS_CA_CERTS"),
            tls_certfile=os.environ.get("MQTT_TLS_CERTFILE"),
            tls_keyfile=os.environ.get("MQTT_TLS_KEYFILE"),
        )


@dataclass
class CSIMessage:
    """Parsed CSI message from MQTT"""

    device_id: str
    timestamp: int
    batch: List[Dict]
    raw_payload: bytes = field(default=b"", repr=False)

    @classmethod
    def from_payload(cls, device_id: str, payload: bytes) -> "CSIMessage":
        """Parse CSI message from MQTT payload"""
        data = json.loads(payload.decode("utf-8"))
        return cls(
            device_id=device_id,
            timestamp=data.get("timestamp", int(time.time() * 1000)),
            batch=data.get("batch", [data] if "csi_data" in data else []),
            raw_payload=payload,
        )


@dataclass
class DeviceStatusMessage:
    """Parsed device status message"""

    device_id: str
    status: str  # "online" or "offline"
    timestamp: int
    firmware_version: Optional[str] = None
    ip_address: Optional[str] = None
    uptime_seconds: Optional[int] = None
    reason: Optional[str] = None  # For will messages

    @classmethod
    def from_payload(cls, device_id: str, payload: bytes) -> "DeviceStatusMessage":
        """Parse status message from MQTT payload"""
        data = json.loads(payload.decode("utf-8"))
        return cls(
            device_id=device_id,
            status=data.get("status", "unknown"),
            timestamp=data.get("timestamp", int(time.time() * 1000)),
            firmware_version=data.get("firmware_version"),
            ip_address=data.get("ip_address"),
            uptime_seconds=data.get("uptime_seconds"),
            reason=data.get("reason"),
        )


# Callback type definitions
CSICallback = Callable[[CSIMessage], None]
StatusCallback = Callable[[DeviceStatusMessage], None]
ConnectionCallback = Callable[[bool], None]


class MQTTClient:
    """
    MQTT client for receiving CSI data from ESP32 devices.

    Topics subscribed:
    - wavira/device/+/csi/batch - CSI batch data
    - wavira/device/+/status - Device status updates
    - wavira/device/+/will - Last will messages (offline detection)
    """

    # Topic patterns
    TOPIC_CSI_BATCH = "wavira/device/+/csi/batch"
    TOPIC_STATUS = "wavira/device/+/status"
    TOPIC_WILL = "wavira/device/+/will"

    # Regex for extracting device_id from topic
    DEVICE_ID_PATTERN = re.compile(r"wavira/device/([^/]+)/")

    def __init__(self, config: Optional[MQTTConfig] = None):
        """Initialize MQTT client"""
        self.config = config or MQTTConfig.from_env()
        self._client: Optional[mqtt.Client] = None
        self._connected = False
        self._reconnect_delay = self.config.reconnect_delay_min
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Callbacks
        self._csi_callbacks: List[CSICallback] = []
        self._status_callbacks: List[StatusCallback] = []
        self._connection_callbacks: List[ConnectionCallback] = []

        # Statistics
        self._stats = {
            "messages_received": 0,
            "csi_messages": 0,
            "status_messages": 0,
            "errors": 0,
            "reconnects": 0,
        }

    @property
    def connected(self) -> bool:
        """Check if client is connected"""
        return self._connected

    @property
    def stats(self) -> Dict:
        """Get statistics"""
        return self._stats.copy()

    def on_csi(self, callback: CSICallback) -> None:
        """Register callback for CSI messages"""
        self._csi_callbacks.append(callback)

    def on_status(self, callback: StatusCallback) -> None:
        """Register callback for status messages"""
        self._status_callbacks.append(callback)

    def on_connection(self, callback: ConnectionCallback) -> None:
        """Register callback for connection state changes"""
        self._connection_callbacks.append(callback)

    def _create_client(self) -> mqtt.Client:
        """Create and configure MQTT client"""
        if PAHO_V2:
            client = mqtt.Client(
                callback_api_version=CallbackAPIVersion.VERSION2,
                client_id=self.config.client_id,
            )
        else:
            client = mqtt.Client(client_id=self.config.client_id)

        # Set credentials if provided
        if self.config.username:
            client.username_pw_set(
                self.config.username,
                self.config.password,
            )

        # Configure TLS if enabled
        if self.config.tls_enabled:
            client.tls_set(
                ca_certs=self.config.tls_ca_certs,
                certfile=self.config.tls_certfile,
                keyfile=self.config.tls_keyfile,
            )

        # Set callbacks
        client.on_connect = self._on_connect
        client.on_disconnect = self._on_disconnect
        client.on_message = self._on_message

        return client

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        """Handle connection established"""
        # Handle both paho v1 and v2 API
        if PAHO_V2:
            success = reason_code == 0 or (hasattr(reason_code, 'is_failure') and not reason_code.is_failure)
        else:
            success = reason_code == 0

        if success:
            logger.info(f"Connected to MQTT broker at {self.config.broker_host}:{self.config.broker_port}")
            self._connected = True
            self._reconnect_delay = self.config.reconnect_delay_min

            # Subscribe to topics
            topics = [
                (self.TOPIC_CSI_BATCH, self.config.qos),
                (self.TOPIC_STATUS, self.config.qos),
                (self.TOPIC_WILL, self.config.qos),
            ]
            for topic, qos in topics:
                client.subscribe(topic, qos)
                logger.debug(f"Subscribed to {topic} (QoS {qos})")

            # Notify callbacks
            for callback in self._connection_callbacks:
                try:
                    callback(True)
                except Exception as e:
                    logger.error(f"Connection callback error: {e}")
        else:
            logger.error(f"Connection failed with code: {reason_code}")
            self._connected = False

    def _on_disconnect(self, client, userdata, disconnect_flags=None, reason_code=None, properties=None):
        """Handle disconnection"""
        self._connected = False
        logger.warning(f"Disconnected from MQTT broker (reason: {reason_code})")

        # Notify callbacks
        for callback in self._connection_callbacks:
            try:
                callback(False)
            except Exception as e:
                logger.error(f"Connection callback error: {e}")

        self._stats["reconnects"] += 1

    def _on_message(self, client, userdata, msg):
        """Handle incoming message"""
        self._stats["messages_received"] += 1

        try:
            # Extract device_id from topic
            match = self.DEVICE_ID_PATTERN.search(msg.topic)
            if not match:
                logger.warning(f"Could not extract device_id from topic: {msg.topic}")
                return

            device_id = match.group(1)

            # Route message based on topic
            if "/csi/batch" in msg.topic:
                self._handle_csi_message(device_id, msg.payload)
            elif "/status" in msg.topic or "/will" in msg.topic:
                self._handle_status_message(device_id, msg.payload)
            else:
                logger.debug(f"Unknown topic: {msg.topic}")

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            self._stats["errors"] += 1

    def _handle_csi_message(self, device_id: str, payload: bytes) -> None:
        """Handle CSI data message"""
        try:
            message = CSIMessage.from_payload(device_id, payload)
            self._stats["csi_messages"] += 1

            logger.debug(
                f"CSI message from {device_id}: "
                f"{len(message.batch)} packets"
            )

            # Notify callbacks
            for callback in self._csi_callbacks:
                try:
                    callback(message)
                except Exception as e:
                    logger.error(f"CSI callback error: {e}")

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in CSI message from {device_id}: {e}")
            self._stats["errors"] += 1

    def _handle_status_message(self, device_id: str, payload: bytes) -> None:
        """Handle device status message"""
        try:
            message = DeviceStatusMessage.from_payload(device_id, payload)
            self._stats["status_messages"] += 1

            logger.info(
                f"Device {device_id} status: {message.status}"
                + (f" (reason: {message.reason})" if message.reason else "")
            )

            # Notify callbacks
            for callback in self._status_callbacks:
                try:
                    callback(message)
                except Exception as e:
                    logger.error(f"Status callback error: {e}")

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in status message from {device_id}: {e}")
            self._stats["errors"] += 1

    def connect(self) -> None:
        """Connect to MQTT broker (blocking)"""
        if self._client is None:
            self._client = self._create_client()

        logger.info(
            f"Connecting to MQTT broker at "
            f"{self.config.broker_host}:{self.config.broker_port}"
        )

        self._client.connect(
            self.config.broker_host,
            self.config.broker_port,
            self.config.keepalive,
        )

    def start(self) -> None:
        """Start the MQTT client loop (non-blocking)"""
        self.connect()
        self._client.loop_start()

    def stop(self) -> None:
        """Stop the MQTT client"""
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()
            self._connected = False
            logger.info("MQTT client stopped")

    async def start_async(self) -> None:
        """Start the MQTT client asynchronously"""
        self._loop = asyncio.get_event_loop()

        # Run blocking operations in executor
        await self._loop.run_in_executor(None, self.start)

    async def stop_async(self) -> None:
        """Stop the MQTT client asynchronously"""
        if self._loop:
            await self._loop.run_in_executor(None, self.stop)

    def publish(
        self,
        topic: str,
        payload: str | bytes | dict,
        qos: int = 1,
        retain: bool = False,
    ) -> None:
        """Publish a message to a topic"""
        if not self._client or not self._connected:
            raise RuntimeError("MQTT client not connected")

        if isinstance(payload, dict):
            payload = json.dumps(payload).encode("utf-8")
        elif isinstance(payload, str):
            payload = payload.encode("utf-8")

        self._client.publish(topic, payload, qos=qos, retain=retain)


class MQTTClientAsync:
    """
    Async wrapper for MQTTClient using asyncio.

    Provides an async interface for integrating with FastAPI/async applications.
    """

    def __init__(self, config: Optional[MQTTConfig] = None):
        self._client = MQTTClient(config)
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._status_queue: asyncio.Queue = asyncio.Queue()
        self._running = False

        # Set up internal callbacks to queue messages
        self._client.on_csi(self._queue_csi_message)
        self._client.on_status(self._queue_status_message)

    def _queue_csi_message(self, message: CSIMessage) -> None:
        """Queue CSI message for async processing"""
        try:
            self._message_queue.put_nowait(message)
        except asyncio.QueueFull:
            logger.warning("CSI message queue full, dropping message")

    def _queue_status_message(self, message: DeviceStatusMessage) -> None:
        """Queue status message for async processing"""
        try:
            self._status_queue.put_nowait(message)
        except asyncio.QueueFull:
            logger.warning("Status message queue full, dropping message")

    @property
    def connected(self) -> bool:
        return self._client.connected

    @property
    def stats(self) -> Dict:
        return self._client.stats

    def on_connection(self, callback: ConnectionCallback) -> None:
        """Register connection state callback"""
        self._client.on_connection(callback)

    async def start(self) -> None:
        """Start the async MQTT client"""
        self._running = True
        await self._client.start_async()

    async def stop(self) -> None:
        """Stop the async MQTT client"""
        self._running = False
        await self._client.stop_async()

    async def get_csi_message(self, timeout: Optional[float] = None) -> Optional[CSIMessage]:
        """Get next CSI message from queue"""
        try:
            if timeout:
                return await asyncio.wait_for(
                    self._message_queue.get(),
                    timeout=timeout,
                )
            else:
                return await self._message_queue.get()
        except asyncio.TimeoutError:
            return None

    async def get_status_message(self, timeout: Optional[float] = None) -> Optional[DeviceStatusMessage]:
        """Get next status message from queue"""
        try:
            if timeout:
                return await asyncio.wait_for(
                    self._status_queue.get(),
                    timeout=timeout,
                )
            else:
                return await self._status_queue.get()
        except asyncio.TimeoutError:
            return None

    async def process_messages(
        self,
        csi_handler: Callable[[CSIMessage], None],
        status_handler: Optional[Callable[[DeviceStatusMessage], None]] = None,
    ) -> None:
        """
        Process messages continuously.

        Args:
            csi_handler: Async function to handle CSI messages
            status_handler: Async function to handle status messages
        """
        async def process_csi():
            while self._running:
                message = await self.get_csi_message(timeout=1.0)
                if message:
                    try:
                        if asyncio.iscoroutinefunction(csi_handler):
                            await csi_handler(message)
                        else:
                            csi_handler(message)
                    except Exception as e:
                        logger.error(f"CSI handler error: {e}")

        async def process_status():
            while self._running:
                message = await self.get_status_message(timeout=1.0)
                if message and status_handler:
                    try:
                        if asyncio.iscoroutinefunction(status_handler):
                            await status_handler(message)
                        else:
                            status_handler(message)
                    except Exception as e:
                        logger.error(f"Status handler error: {e}")

        tasks = [asyncio.create_task(process_csi())]
        if status_handler:
            tasks.append(asyncio.create_task(process_status()))

        await asyncio.gather(*tasks, return_exceptions=True)


# Convenience function for creating configured client
def create_mqtt_client(config: Optional[MQTTConfig] = None) -> MQTTClientAsync:
    """Create an async MQTT client with optional configuration"""
    return MQTTClientAsync(config or MQTTConfig.from_env())

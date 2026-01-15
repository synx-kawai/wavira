"""
MQTT Device Simulator for Integration Testing

This module provides a simulated ESP32 device that publishes CSI data
via MQTT, enabling end-to-end testing without physical hardware.

Features:
- Multiple simulated devices with unique IDs
- Realistic CSI data generation (amplitude, subcarriers)
- Device status (online/offline) with Last Will support
- Configurable publish rates and QoS levels
- Presence/breathing pattern simulation
- Connection error injection for testing resilience

Usage:
    from wavira.utils.mqtt_simulator import MQTTDeviceSimulator, SimulatorConfig

    config = SimulatorConfig(broker_host="localhost")
    sim = MQTTDeviceSimulator(config)

    # Start a single device
    sim.start_device("device_001")

    # Or start multiple devices for load testing
    sim.start_devices(10)  # Start 10 devices

    # Stop all devices
    sim.stop_all()
"""

import json
import logging
import random
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any
from enum import Enum
import numpy as np

try:
    import paho.mqtt.client as mqtt
    PAHO_AVAILABLE = True
except ImportError:
    PAHO_AVAILABLE = False
    mqtt = None

logger = logging.getLogger(__name__)


class PresencePattern(Enum):
    """Presence simulation patterns."""
    ALWAYS_PRESENT = "always_present"
    ALWAYS_ABSENT = "always_absent"
    RANDOM = "random"
    PERIODIC = "periodic"  # Present for N seconds, absent for M seconds
    BREATHING = "breathing"  # Simulates breathing patterns when present


class DeviceState(Enum):
    """Device connection states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class SimulatorConfig:
    """Configuration for MQTT device simulator.

    Attributes:
        broker_host: MQTT broker hostname
        broker_port: MQTT broker port
        username: MQTT username (optional)
        password: MQTT password (optional)
        num_subcarriers: Number of CSI subcarriers to simulate
        publish_rate_hz: Data publish rate in Hz (messages per second)
        qos: MQTT QoS level (0, 1, or 2)
        clean_session: Whether to use clean session
        keepalive: MQTT keepalive interval in seconds
        base_amplitude: Base CSI amplitude value
        amplitude_noise: Amplitude noise standard deviation
        presence_pattern: Pattern for presence simulation
        presence_period_on: Seconds of presence (for PERIODIC pattern)
        presence_period_off: Seconds of absence (for PERIODIC pattern)
        breathing_rate: Breaths per minute (for BREATHING pattern)
        connect_timeout: Connection timeout in seconds
        auto_reconnect: Whether to auto-reconnect on disconnect
        reconnect_delay: Base delay for reconnection (exponential backoff)
        max_reconnect_delay: Maximum reconnection delay
    """
    broker_host: str = "localhost"
    broker_port: int = 1883
    username: Optional[str] = None
    password: Optional[str] = None
    num_subcarriers: int = 52
    publish_rate_hz: float = 10.0
    qos: int = 1
    clean_session: bool = True
    keepalive: int = 60
    base_amplitude: float = 50.0
    amplitude_noise: float = 5.0
    presence_pattern: PresencePattern = PresencePattern.RANDOM
    presence_period_on: float = 30.0
    presence_period_off: float = 30.0
    breathing_rate: float = 15.0  # breaths per minute
    connect_timeout: float = 10.0
    auto_reconnect: bool = True
    reconnect_delay: float = 1.0
    max_reconnect_delay: float = 30.0


@dataclass
class DeviceStats:
    """Statistics for a simulated device.

    Attributes:
        device_id: Device identifier
        messages_sent: Total messages sent
        messages_failed: Total failed sends
        connect_time: Time of last connection
        disconnect_time: Time of last disconnection
        total_connect_time: Total connected time in seconds
        reconnect_count: Number of reconnection attempts
        last_error: Last error message
        state: Current device state
    """
    device_id: str
    messages_sent: int = 0
    messages_failed: int = 0
    connect_time: Optional[float] = None
    disconnect_time: Optional[float] = None
    total_connect_time: float = 0.0
    reconnect_count: int = 0
    last_error: Optional[str] = None
    state: DeviceState = DeviceState.DISCONNECTED

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "device_id": self.device_id,
            "messages_sent": self.messages_sent,
            "messages_failed": self.messages_failed,
            "connect_time": self.connect_time,
            "disconnect_time": self.disconnect_time,
            "total_connect_time": self.total_connect_time,
            "reconnect_count": self.reconnect_count,
            "last_error": self.last_error,
            "state": self.state.value,
        }


class SimulatedDevice:
    """A single simulated ESP32 device.

    This class simulates an ESP32 device that collects CSI data and
    publishes it via MQTT. It handles connection management, data
    generation, and Last Will messages for offline detection.
    """

    def __init__(
        self,
        device_id: str,
        config: SimulatorConfig,
        on_message_callback: Optional[Callable[[str, dict], None]] = None,
    ):
        """Initialize simulated device.

        Args:
            device_id: Unique device identifier
            config: Simulator configuration
            on_message_callback: Optional callback for sent messages
        """
        if not PAHO_AVAILABLE:
            raise RuntimeError("paho-mqtt is required. Install with: pip install paho-mqtt")

        self.device_id = device_id
        self.config = config
        self.on_message_callback = on_message_callback

        self.stats = DeviceStats(device_id=device_id)
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._client: Optional[mqtt.Client] = None
        self._lock = threading.Lock()

        # State tracking
        self._start_time = 0.0
        self._last_publish_time = 0.0
        self._presence_start_time = 0.0
        self._is_present = False

        # Topics
        self._csi_topic = f"wavira/csi/{device_id}"
        self._analysis_topic = f"wavira/analysis/{device_id}"
        self._status_topic = f"wavira/device/{device_id}/status"
        self._will_topic = f"wavira/device/{device_id}/will"

    def _create_client(self) -> mqtt.Client:
        """Create and configure MQTT client."""
        client_id = f"wavira_sim_{self.device_id}_{uuid.uuid4().hex[:8]}"
        client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=client_id,
            clean_session=self.config.clean_session,
        )

        if self.config.username:
            client.username_pw_set(self.config.username, self.config.password)

        # Set Last Will message
        will_payload = json.dumps({
            "device_id": self.device_id,
            "status": "offline",
            "reason": "unexpected_disconnect",
            "timestamp": time.time(),
        })
        client.will_set(
            self._will_topic,
            payload=will_payload,
            qos=1,
            retain=True,
        )

        client.on_connect = self._on_connect
        client.on_disconnect = self._on_disconnect

        return client

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        """Handle MQTT connection."""
        if reason_code == 0:
            logger.info(f"Device {self.device_id} connected to MQTT broker")
            with self._lock:
                self.stats.state = DeviceState.CONNECTED
                self.stats.connect_time = time.time()

            # Publish online status
            self._publish_status("online")
        else:
            logger.error(f"Device {self.device_id} connection failed: {reason_code}")
            with self._lock:
                self.stats.state = DeviceState.ERROR
                self.stats.last_error = f"Connection failed: {reason_code}"

    def _on_disconnect(self, client, userdata, flags, reason_code, properties):
        """Handle MQTT disconnection."""
        logger.warning(f"Device {self.device_id} disconnected: {reason_code}")
        with self._lock:
            if self.stats.connect_time:
                self.stats.total_connect_time += time.time() - self.stats.connect_time
            self.stats.disconnect_time = time.time()
            self.stats.state = DeviceState.DISCONNECTED

    def _publish_status(self, status: str):
        """Publish device status message."""
        if not self._client or not self._client.is_connected():
            return

        payload = json.dumps({
            "device_id": self.device_id,
            "status": status,
            "timestamp": time.time(),
        })

        self._client.publish(
            self._status_topic,
            payload=payload,
            qos=1,
            retain=True,
        )

    def _generate_csi_data(self, timestamp: float) -> Dict[str, Any]:
        """Generate simulated CSI data.

        Args:
            timestamp: Current timestamp

        Returns:
            Dictionary with CSI data
        """
        # Determine presence based on pattern
        is_present = self._determine_presence(timestamp)

        # Base amplitudes with noise
        amplitudes = (
            self.config.base_amplitude +
            np.random.randn(self.config.num_subcarriers) * self.config.amplitude_noise
        )

        # Add presence effect (higher variance when present)
        if is_present:
            presence_variance = 10.0
            amplitudes += np.random.randn(self.config.num_subcarriers) * presence_variance

            # Add breathing pattern if configured
            if self.config.presence_pattern == PresencePattern.BREATHING:
                breath_freq = self.config.breathing_rate / 60.0  # Convert to Hz
                breath_signal = 5.0 * np.sin(2 * np.pi * breath_freq * timestamp)
                amplitudes += breath_signal

        # Calculate derived metrics
        avg_amplitude = float(np.mean(amplitudes))
        variance = float(np.var(amplitudes))

        # Simulate breathing detection
        breath_ratio = 0.0
        if is_present and self.config.presence_pattern == PresencePattern.BREATHING:
            breath_ratio = 0.3 + random.uniform(-0.1, 0.1)

        return {
            "device_id": self.device_id,
            "timestamp": timestamp,
            "rssi": random.randint(-70, -40),
            "amplitudes": amplitudes.tolist(),
            "avg_amplitude": avg_amplitude,
            "variance": variance,
            "present": is_present,
            "breath_ratio": breath_ratio,
            "breathing": is_present and breath_ratio > 0.2,
            "rate": 54,  # 54 Mbps
            "noise_floor": random.randint(-95, -85),
        }

    def _determine_presence(self, timestamp: float) -> bool:
        """Determine if presence should be simulated.

        Args:
            timestamp: Current timestamp

        Returns:
            True if presence should be simulated
        """
        pattern = self.config.presence_pattern

        if pattern == PresencePattern.ALWAYS_PRESENT:
            return True
        elif pattern == PresencePattern.ALWAYS_ABSENT:
            return False
        elif pattern == PresencePattern.RANDOM:
            return random.random() > 0.3  # 70% chance of presence
        elif pattern in (PresencePattern.PERIODIC, PresencePattern.BREATHING):
            elapsed = timestamp - self._start_time
            period = self.config.presence_period_on + self.config.presence_period_off
            position = elapsed % period
            return position < self.config.presence_period_on
        else:
            return False

    def _publish_csi(self, data: Dict[str, Any]):
        """Publish CSI data to MQTT.

        Args:
            data: CSI data dictionary
        """
        if not self._client or not self._client.is_connected():
            with self._lock:
                self.stats.messages_failed += 1
            return

        # Publish to analysis topic (without raw amplitudes for dashboard)
        analysis_data = {k: v for k, v in data.items() if k != "amplitudes"}
        payload = json.dumps(analysis_data)

        result = self._client.publish(
            self._analysis_topic,
            payload=payload,
            qos=self.config.qos,
        )

        with self._lock:
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                self.stats.messages_sent += 1
            else:
                self.stats.messages_failed += 1

        if self.on_message_callback:
            self.on_message_callback(self._analysis_topic, data)

    def _run_loop(self):
        """Main device loop."""
        self._start_time = time.time()
        publish_interval = 1.0 / self.config.publish_rate_hz

        while self.running:
            try:
                current_time = time.time()

                # Publish at configured rate
                if current_time - self._last_publish_time >= publish_interval:
                    data = self._generate_csi_data(current_time)
                    self._publish_csi(data)
                    self._last_publish_time = current_time

                # Small sleep to prevent busy loop
                time.sleep(0.01)

            except Exception as e:
                logger.error(f"Device {self.device_id} error: {e}")
                with self._lock:
                    self.stats.last_error = str(e)
                time.sleep(1.0)

    def start(self) -> bool:
        """Start the simulated device.

        Returns:
            True if started successfully
        """
        if self.running:
            return True

        try:
            with self._lock:
                self.stats.state = DeviceState.CONNECTING

            self._client = self._create_client()
            self._client.connect(
                self.config.broker_host,
                self.config.broker_port,
                keepalive=self.config.keepalive,
            )
            self._client.loop_start()

            # Wait for connection
            timeout = self.config.connect_timeout
            start = time.time()
            while time.time() - start < timeout:
                if self.stats.state == DeviceState.CONNECTED:
                    break
                time.sleep(0.1)

            if self.stats.state != DeviceState.CONNECTED:
                raise ConnectionError("Failed to connect within timeout")

            self.running = True
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()

            logger.info(f"Device {self.device_id} started")
            return True

        except Exception as e:
            logger.error(f"Failed to start device {self.device_id}: {e}")
            with self._lock:
                self.stats.state = DeviceState.ERROR
                self.stats.last_error = str(e)
            return False

    def stop(self):
        """Stop the simulated device."""
        self.running = False

        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None

        if self._client:
            # Publish offline status before disconnecting
            self._publish_status("offline")
            time.sleep(0.1)  # Allow message to be sent

            self._client.loop_stop()
            self._client.disconnect()
            self._client = None

        with self._lock:
            self.stats.state = DeviceState.DISCONNECTED

        logger.info(f"Device {self.device_id} stopped")

    def get_stats(self) -> DeviceStats:
        """Get device statistics.

        Returns:
            Copy of device statistics
        """
        with self._lock:
            stats = DeviceStats(
                device_id=self.stats.device_id,
                messages_sent=self.stats.messages_sent,
                messages_failed=self.stats.messages_failed,
                connect_time=self.stats.connect_time,
                disconnect_time=self.stats.disconnect_time,
                total_connect_time=self.stats.total_connect_time,
                reconnect_count=self.stats.reconnect_count,
                last_error=self.stats.last_error,
                state=self.stats.state,
            )
            # Update total connect time if still connected
            if stats.state == DeviceState.CONNECTED and stats.connect_time:
                stats.total_connect_time += time.time() - stats.connect_time
            return stats


class MQTTDeviceSimulator:
    """Manages multiple simulated MQTT devices.

    This class provides a high-level interface for creating and managing
    multiple simulated devices for load testing and integration testing.

    Usage:
        sim = MQTTDeviceSimulator(config)

        # Start specific devices
        sim.start_device("device_001")
        sim.start_device("device_002")

        # Or start multiple devices at once
        sim.start_devices(count=10, prefix="test_")

        # Get statistics
        stats = sim.get_all_stats()
        print(f"Total messages: {stats['total_messages_sent']}")

        # Stop all devices
        sim.stop_all()
    """

    def __init__(
        self,
        config: Optional[SimulatorConfig] = None,
        on_message_callback: Optional[Callable[[str, str, dict], None]] = None,
    ):
        """Initialize the simulator.

        Args:
            config: Simulator configuration (uses defaults if not provided)
            on_message_callback: Optional callback (device_id, topic, data)
        """
        self.config = config or SimulatorConfig()
        self.on_message_callback = on_message_callback
        self._devices: Dict[str, SimulatedDevice] = {}
        self._lock = threading.Lock()

    def _device_callback(self, device_id: str):
        """Create a callback wrapper for a specific device."""
        def callback(topic: str, data: dict):
            if self.on_message_callback:
                self.on_message_callback(device_id, topic, data)
        return callback

    def start_device(
        self,
        device_id: str,
        config: Optional[SimulatorConfig] = None,
    ) -> bool:
        """Start a single simulated device.

        Args:
            device_id: Unique device identifier
            config: Optional per-device configuration override

        Returns:
            True if device started successfully
        """
        with self._lock:
            if device_id in self._devices:
                logger.warning(f"Device {device_id} already exists")
                return False

        device_config = config or self.config
        device = SimulatedDevice(
            device_id=device_id,
            config=device_config,
            on_message_callback=self._device_callback(device_id) if self.on_message_callback else None,
        )

        if device.start():
            with self._lock:
                self._devices[device_id] = device
            return True
        return False

    def start_devices(
        self,
        count: int,
        prefix: str = "sim_device_",
        stagger_delay: float = 0.1,
    ) -> List[str]:
        """Start multiple simulated devices.

        Args:
            count: Number of devices to start
            prefix: Device ID prefix
            stagger_delay: Delay between starting devices (to avoid thundering herd)

        Returns:
            List of successfully started device IDs
        """
        started = []
        for i in range(count):
            device_id = f"{prefix}{i:03d}"
            if self.start_device(device_id):
                started.append(device_id)
            if i < count - 1 and stagger_delay > 0:
                time.sleep(stagger_delay)
        return started

    def stop_device(self, device_id: str):
        """Stop a specific device.

        Args:
            device_id: Device to stop
        """
        with self._lock:
            device = self._devices.pop(device_id, None)

        if device:
            device.stop()

    def stop_all(self):
        """Stop all simulated devices."""
        with self._lock:
            devices = list(self._devices.values())
            self._devices.clear()

        for device in devices:
            device.stop()

        logger.info(f"Stopped {len(devices)} devices")

    def get_device_stats(self, device_id: str) -> Optional[DeviceStats]:
        """Get statistics for a specific device.

        Args:
            device_id: Device to query

        Returns:
            DeviceStats or None if device not found
        """
        with self._lock:
            device = self._devices.get(device_id)
        return device.get_stats() if device else None

    def get_all_stats(self) -> Dict[str, Any]:
        """Get aggregated statistics for all devices.

        Returns:
            Dictionary with aggregated statistics
        """
        with self._lock:
            devices = list(self._devices.values())

        device_stats = [d.get_stats() for d in devices]

        total_sent = sum(s.messages_sent for s in device_stats)
        total_failed = sum(s.messages_failed for s in device_stats)
        total_connected = sum(1 for s in device_stats if s.state == DeviceState.CONNECTED)

        return {
            "device_count": len(devices),
            "connected_count": total_connected,
            "total_messages_sent": total_sent,
            "total_messages_failed": total_failed,
            "success_rate": total_sent / max(1, total_sent + total_failed),
            "devices": {s.device_id: s.to_dict() for s in device_stats},
        }

    def wait_for_messages(
        self,
        target_count: int,
        timeout: float = 30.0,
        check_interval: float = 0.5,
    ) -> bool:
        """Wait until target number of messages have been sent.

        Args:
            target_count: Target total message count
            timeout: Maximum wait time in seconds
            check_interval: Check interval in seconds

        Returns:
            True if target was reached, False if timeout
        """
        start = time.time()
        while time.time() - start < timeout:
            stats = self.get_all_stats()
            if stats["total_messages_sent"] >= target_count:
                return True
            time.sleep(check_interval)
        return False

    def inject_disconnect(self, device_id: str) -> bool:
        """Simulate a network disconnect for a device.

        Args:
            device_id: Device to disconnect

        Returns:
            True if device was disconnected
        """
        with self._lock:
            device = self._devices.get(device_id)

        if device and device._client:
            device._client.disconnect()
            return True
        return False

    @property
    def device_count(self) -> int:
        """Get the number of active devices."""
        with self._lock:
            return len(self._devices)

    @property
    def connected_count(self) -> int:
        """Get the number of connected devices."""
        with self._lock:
            devices = list(self._devices.values())
        return sum(1 for d in devices if d.stats.state == DeviceState.CONNECTED)


def run_load_test(
    broker_host: str = "localhost",
    broker_port: int = 1883,
    device_count: int = 10,
    duration_seconds: float = 60.0,
    publish_rate_hz: float = 10.0,
) -> Dict[str, Any]:
    """Run a load test with multiple simulated devices.

    Args:
        broker_host: MQTT broker hostname
        broker_port: MQTT broker port
        device_count: Number of devices to simulate
        duration_seconds: Test duration in seconds
        publish_rate_hz: Publish rate per device

    Returns:
        Dictionary with test results
    """
    config = SimulatorConfig(
        broker_host=broker_host,
        broker_port=broker_port,
        publish_rate_hz=publish_rate_hz,
    )

    simulator = MQTTDeviceSimulator(config)

    logger.info(f"Starting load test: {device_count} devices, {duration_seconds}s")

    # Start devices
    start_time = time.time()
    started = simulator.start_devices(device_count)

    if len(started) < device_count:
        logger.warning(f"Only {len(started)}/{device_count} devices started")

    # Run for specified duration
    time.sleep(duration_seconds)

    # Collect results
    stats = simulator.get_all_stats()
    elapsed = time.time() - start_time

    results = {
        "duration_seconds": elapsed,
        "device_count": len(started),
        "connected_count": stats["connected_count"],
        "total_messages_sent": stats["total_messages_sent"],
        "total_messages_failed": stats["total_messages_failed"],
        "success_rate": stats["success_rate"],
        "messages_per_second": stats["total_messages_sent"] / elapsed,
        "expected_messages": int(device_count * publish_rate_hz * duration_seconds),
    }

    # Calculate message delivery rate
    results["delivery_rate"] = (
        results["total_messages_sent"] / max(1, results["expected_messages"])
    )

    # Stop all devices
    simulator.stop_all()

    logger.info(f"Load test complete: {results['messages_per_second']:.1f} msg/s")

    return results

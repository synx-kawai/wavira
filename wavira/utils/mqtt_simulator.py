#!/usr/bin/env python3
"""
MQTT Simulator for Wavira CSI System

Simulates ESP32 devices sending CSI data via MQTT for testing purposes.

Usage:
    python mqtt_simulator.py --device-id esp32-test-001 --broker localhost
    python mqtt_simulator.py --device-count 10 --rate 10
"""

import argparse
import json
import logging
import random
import signal
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

try:
    import paho.mqtt.client as mqtt
    from paho.mqtt.enums import CallbackAPIVersion
    PAHO_V2 = True
except ImportError:
    try:
        import paho.mqtt.client as mqtt
        PAHO_V2 = False
    except ImportError:
        print("Error: paho-mqtt not installed. Run: pip install paho-mqtt")
        sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class SimulatorConfig:
    """Simulator configuration"""
    broker_host: str = "localhost"
    broker_port: int = 1883
    username: Optional[str] = None
    password: Optional[str] = None
    device_id: str = "esp32-sim-001"
    device_count: int = 1
    rate_hz: float = 10.0
    duration_seconds: int = 0  # 0 = infinite
    qos: int = 1


class CSIDataGenerator:
    """Generates simulated CSI data"""

    def __init__(self, device_id: str):
        self.device_id = device_id
        self.seq = 0
        self.start_time = time.time()

    def generate_packet(self) -> dict:
        """Generate a single CSI packet"""
        self.seq += 1
        timestamp = int(time.time() * 1000)

        # Simulate CSI data (64 subcarriers)
        csi_data = [
            random.uniform(-30, 30) for _ in range(64)
        ]

        return {
            "device_id": self.device_id,
            "seq": self.seq,
            "timestamp": timestamp,
            "mac": f"aa:bb:cc:dd:ee:{random.randint(0, 255):02x}",
            "rssi": random.randint(-80, -30),
            "rate": 11,
            "channel": 6,
            "noise_floor": -95,
            "csi_data": csi_data,
            "metadata": {
                "firmware_version": "1.0.0-sim",
                "uptime_ms": int((time.time() - self.start_time) * 1000),
            },
        }

    def generate_batch(self, count: int = 10) -> dict:
        """Generate a batch of CSI packets"""
        return {
            "device_id": self.device_id,
            "timestamp": int(time.time() * 1000),
            "batch": [self.generate_packet() for _ in range(count)],
        }


class DeviceSimulator:
    """Simulates a single ESP32 device"""

    def __init__(self, config: SimulatorConfig, device_id: str):
        self.config = config
        self.device_id = device_id
        self.generator = CSIDataGenerator(device_id)
        self.client: Optional[mqtt.Client] = None
        self.connected = False
        self.running = False
        self.packets_sent = 0

        # Topics
        self.csi_topic = f"wavira/device/{device_id}/csi/batch"
        self.status_topic = f"wavira/device/{device_id}/status"
        self.will_topic = f"wavira/device/{device_id}/will"

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        """Handle connection"""
        if PAHO_V2:
            success = reason_code == 0 or (hasattr(reason_code, 'is_failure') and not reason_code.is_failure)
        else:
            success = reason_code == 0

        if success:
            logger.info(f"[{self.device_id}] Connected to broker")
            self.connected = True

            # Publish online status
            status = {
                "device_id": self.device_id,
                "status": "online",
                "timestamp": int(time.time() * 1000),
                "firmware_version": "1.0.0-sim",
            }
            client.publish(self.status_topic, json.dumps(status), qos=1, retain=True)
        else:
            logger.error(f"[{self.device_id}] Connection failed: {reason_code}")

    def _on_disconnect(self, client, userdata, disconnect_flags=None, reason_code=None, properties=None):
        """Handle disconnection"""
        self.connected = False
        logger.warning(f"[{self.device_id}] Disconnected")

    def start(self):
        """Start the simulator"""
        # Create client
        client_id = f"wavira_{self.device_id}"
        if PAHO_V2:
            self.client = mqtt.Client(
                callback_api_version=CallbackAPIVersion.VERSION2,
                client_id=client_id,
            )
        else:
            self.client = mqtt.Client(client_id=client_id)

        # Set callbacks
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect

        # Set credentials
        if self.config.username:
            self.client.username_pw_set(self.config.username, self.config.password)

        # Set last will
        will_msg = json.dumps({
            "device_id": self.device_id,
            "status": "offline",
            "timestamp": int(time.time() * 1000),
            "reason": "unexpected_disconnect",
        })
        self.client.will_set(self.will_topic, will_msg, qos=1, retain=True)

        # Connect
        logger.info(f"[{self.device_id}] Connecting to {self.config.broker_host}:{self.config.broker_port}")
        self.client.connect(self.config.broker_host, self.config.broker_port, 60)
        self.client.loop_start()
        self.running = True

    def stop(self):
        """Stop the simulator"""
        self.running = False
        if self.client:
            # Publish offline status
            status = {
                "device_id": self.device_id,
                "status": "offline",
                "timestamp": int(time.time() * 1000),
                "reason": "graceful_shutdown",
            }
            self.client.publish(self.status_topic, json.dumps(status), qos=1, retain=True)
            time.sleep(0.1)

            self.client.loop_stop()
            self.client.disconnect()
        logger.info(f"[{self.device_id}] Stopped. Sent {self.packets_sent} packets")

    def send_batch(self, batch_size: int = 10):
        """Send a batch of CSI data"""
        if not self.connected or not self.client:
            return False

        batch = self.generator.generate_batch(batch_size)
        payload = json.dumps(batch)

        result = self.client.publish(
            self.csi_topic,
            payload,
            qos=self.config.qos,
        )

        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            self.packets_sent += batch_size
            return True
        return False


def run_simulator(config: SimulatorConfig):
    """Run the MQTT simulator"""
    devices: List[DeviceSimulator] = []

    # Create device simulators
    for i in range(config.device_count):
        if config.device_count == 1:
            device_id = config.device_id
        else:
            device_id = f"{config.device_id}-{i+1:03d}"
        device = DeviceSimulator(config, device_id)
        devices.append(device)

    # Handle shutdown
    shutdown = False

    def signal_handler(sig, frame):
        nonlocal shutdown
        logger.info("Shutdown signal received...")
        shutdown = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start all devices
    for device in devices:
        device.start()
        time.sleep(0.1)  # Stagger connections

    # Wait for connections
    time.sleep(1)

    # Calculate interval
    interval = 1.0 / config.rate_hz
    batch_size = max(1, int(config.rate_hz / 10))  # Batch at 10 Hz

    start_time = time.time()
    last_stats_time = start_time

    logger.info(f"Starting simulation: {config.device_count} devices, {config.rate_hz} Hz")

    try:
        while not shutdown:
            loop_start = time.time()

            # Check duration limit
            if config.duration_seconds > 0:
                if time.time() - start_time > config.duration_seconds:
                    logger.info("Duration limit reached")
                    break

            # Send data from all devices
            for device in devices:
                device.send_batch(batch_size)

            # Print stats periodically
            if time.time() - last_stats_time > 10:
                total_sent = sum(d.packets_sent for d in devices)
                elapsed = time.time() - start_time
                rate = total_sent / elapsed if elapsed > 0 else 0
                logger.info(f"Stats: {total_sent} packets sent, {rate:.1f} packets/s")
                last_stats_time = time.time()

            # Sleep to maintain rate
            elapsed = time.time() - loop_start
            sleep_time = max(0, interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        # Stop all devices
        for device in devices:
            device.stop()

    # Final stats
    total_sent = sum(d.packets_sent for d in devices)
    elapsed = time.time() - start_time
    logger.info(f"Simulation complete: {total_sent} packets in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="MQTT Simulator for Wavira CSI")
    parser.add_argument("--broker", "-b", default="localhost", help="MQTT broker host")
    parser.add_argument("--port", "-p", type=int, default=1883, help="MQTT broker port")
    parser.add_argument("--username", "-u", help="MQTT username")
    parser.add_argument("--password", "-P", help="MQTT password")
    parser.add_argument("--device-id", "-d", default="esp32-sim-001", help="Device ID prefix")
    parser.add_argument("--device-count", "-n", type=int, default=1, help="Number of devices to simulate")
    parser.add_argument("--rate", "-r", type=float, default=10.0, help="Packets per second per device")
    parser.add_argument("--duration", "-t", type=int, default=0, help="Duration in seconds (0 = infinite)")
    parser.add_argument("--qos", type=int, default=1, choices=[0, 1, 2], help="MQTT QoS level")
    args = parser.parse_args()

    config = SimulatorConfig(
        broker_host=args.broker,
        broker_port=args.port,
        username=args.username,
        password=args.password,
        device_id=args.device_id,
        device_count=args.device_count,
        rate_hz=args.rate,
        duration_seconds=args.duration,
        qos=args.qos,
    )

    run_simulator(config)


if __name__ == "__main__":
    main()

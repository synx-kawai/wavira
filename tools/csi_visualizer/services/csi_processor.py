#!/usr/bin/env python3
"""
CSI Processor Service for Wavira

Subscribes to CSI data from ESP32 devices via MQTT, performs analysis
(presence detection, breathing detection), and publishes results back to MQTT.

MQTT Topics:
    Subscribe: wavira/csi/{device_id}
    Publish:   wavira/analysis/{device_id}
               wavira/analysis/{device_id}/presence
               wavira/analysis/{device_id}/breathing
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import paho.mqtt.client as mqtt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class DeviceAnalysis:
    """Device analysis state."""
    device_id: str
    # Recent amplitude history for analysis
    amp_history: deque = field(default_factory=lambda: deque(maxlen=100))
    # Analysis results
    present: bool = False
    breath_ratio: float = 0.0
    breathing_detected: bool = False
    breathing_rate: float = 0.0
    last_update: float = 0.0
    # Statistics
    avg_amplitude: float = 0.0
    amplitude_variance: float = 0.0


class CSIProcessor:
    """CSI data processor with MQTT pub/sub."""

    # Presence detection threshold (amplitude variance)
    PRESENCE_VARIANCE_THRESHOLD = 50
    # Breathing detection parameters
    BREATHING_VARIANCE_MIN = 10
    BREATHING_VARIANCE_MAX = 500
    # Publish rate limit (seconds)
    MIN_PUBLISH_INTERVAL = 0.1

    def __init__(self, mqtt_host: str, mqtt_port: int):
        self.mqtt_host = mqtt_host
        self.mqtt_port = mqtt_port
        self.running = False

        # Device analysis states
        self.devices: Dict[str, DeviceAnalysis] = {}

        # MQTT client
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message

        # Last publish timestamps per device
        self._last_publish: Dict[str, float] = {}

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        """MQTT connection callback."""
        if reason_code == 0:
            logger.info(f"Connected to MQTT broker at {self.mqtt_host}:{self.mqtt_port}")
            # Subscribe to CSI data from all devices
            client.subscribe("wavira/csi/#", qos=0)
            logger.info("Subscribed to wavira/csi/#")
        else:
            logger.error(f"MQTT connection failed: {reason_code}")

    def _on_disconnect(self, client, userdata, flags, reason_code, properties):
        """MQTT disconnection callback."""
        logger.warning(f"MQTT disconnected: {reason_code}")
        if self.running:
            logger.info("Attempting to reconnect...")

    def _on_message(self, client, userdata, msg):
        """MQTT message callback."""
        try:
            topic = msg.topic
            if not topic.startswith("wavira/csi/"):
                return

            # Parse device ID from topic
            device_id = topic.split("/")[-1]

            # Parse payload
            payload = json.loads(msg.payload.decode())
            self._process_csi(device_id, payload)

        except json.JSONDecodeError as e:
            logger.debug(f"Invalid JSON: {e}")
        except Exception as e:
            logger.error(f"Error processing message: {e}")

    def _process_csi(self, device_id: str, data: dict):
        """Process CSI data and perform analysis."""
        now = time.time()

        # Get or create device analysis state
        if device_id not in self.devices:
            self.devices[device_id] = DeviceAnalysis(device_id=device_id)
            logger.info(f"New device registered: {device_id}")

        device = self.devices[device_id]
        device.last_update = now

        # Extract CSI data (I/Q pairs)
        csi_data = data.get("data", [])
        rssi = data.get("rssi", -100)

        if not csi_data:
            return

        # Calculate amplitudes from I/Q pairs
        amplitudes = []
        for i in range(0, len(csi_data) - 1, 2):
            i_val = csi_data[i]
            q_val = csi_data[i + 1] if i + 1 < len(csi_data) else 0
            amp = np.sqrt(i_val ** 2 + q_val ** 2)
            amplitudes.append(amp)

        if not amplitudes:
            return

        # Calculate statistics
        amplitudes = np.array(amplitudes)
        avg_amp = float(np.mean(amplitudes))
        variance = float(np.var(amplitudes))

        # Update device state
        device.avg_amplitude = avg_amp
        device.amplitude_variance = variance
        device.amp_history.append(avg_amp)

        # Presence detection (simple variance-based)
        device.present = variance > self.PRESENCE_VARIANCE_THRESHOLD
        device.breath_ratio = min(variance / 500.0, 1.0) if variance > 0 else 0.0

        # Breathing detection (amplitude variation pattern)
        if len(device.amp_history) >= 20:
            history = np.array(list(device.amp_history))
            history_variance = float(np.var(history))
            device.breathing_detected = (
                self.BREATHING_VARIANCE_MIN < history_variance < self.BREATHING_VARIANCE_MAX
                and device.present
            )
            # Estimate breathing rate from oscillation frequency (simplified)
            if device.breathing_detected:
                # Count zero-crossings of de-meaned signal
                centered = history - np.mean(history)
                zero_crossings = int(np.sum(np.diff(np.sign(centered)) != 0))
                # Assuming ~10Hz sampling rate, convert to breaths per minute
                device.breathing_rate = float(zero_crossings * 3)  # rough estimate
            else:
                device.breathing_rate = 0.0
        else:
            device.breathing_detected = False
            device.breathing_rate = 0.0

        # Publish analysis results (rate limited)
        self._publish_analysis(device, rssi, amplitudes.tolist())

    def _publish_analysis(self, device: DeviceAnalysis, rssi: int, amplitudes: list):
        """Publish analysis results to MQTT."""
        now = time.time()
        device_id = device.device_id

        # Rate limit publishing
        last = self._last_publish.get(device_id, 0)
        if now - last < self.MIN_PUBLISH_INTERVAL:
            return
        self._last_publish[device_id] = now

        # Main analysis result - ensure all values are JSON serializable
        analysis = {
            "device_id": device_id,
            "timestamp": now,
            "rssi": int(rssi),
            "amps": [float(a) for a in amplitudes[:64]],  # Limit size, convert to float
            "avg_amplitude": float(round(device.avg_amplitude, 2)),
            "variance": float(round(device.amplitude_variance, 2)),
            "present": bool(device.present),
            "breath_ratio": float(round(device.breath_ratio, 3)),
            "breathing": bool(device.breathing_detected),
            "breathing_rate": float(round(device.breathing_rate, 1)),
        }

        # Publish to main analysis topic
        topic = f"wavira/analysis/{device_id}"
        self.client.publish(topic, json.dumps(analysis), qos=0)

        # Publish presence status (retained for state tracking)
        presence_topic = f"wavira/analysis/{device_id}/presence"
        presence_payload = {
            "device_id": device_id,
            "timestamp": now,
            "present": device.present,
            "confidence": device.breath_ratio,
        }
        self.client.publish(presence_topic, json.dumps(presence_payload), qos=1, retain=True)

        # Publish breathing status if detected
        if device.breathing_detected:
            breathing_topic = f"wavira/analysis/{device_id}/breathing"
            breathing_payload = {
                "device_id": device_id,
                "timestamp": now,
                "breathing": True,
                "rate": device.breathing_rate,
            }
            self.client.publish(breathing_topic, json.dumps(breathing_payload), qos=0)

    def run(self):
        """Run the processor."""
        self.running = True
        logger.info(f"Starting CSI Processor")
        logger.info(f"  MQTT: {self.mqtt_host}:{self.mqtt_port}")

        # Connect to MQTT broker
        try:
            self.client.connect(self.mqtt_host, self.mqtt_port, keepalive=60)
        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {e}")
            return

        # Run MQTT loop (blocking)
        try:
            self.client.loop_forever()
        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            self.client.disconnect()
            logger.info("CSI Processor stopped")

    def stop(self):
        """Stop the processor."""
        self.running = False
        self.client.disconnect()


def main():
    parser = argparse.ArgumentParser(description="CSI Processor Service")
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
    args = parser.parse_args()

    processor = CSIProcessor(args.mqtt_host, args.mqtt_port)

    def signal_handler(sig, frame):
        logger.info("Shutting down...")
        processor.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    processor.run()


if __name__ == "__main__":
    main()

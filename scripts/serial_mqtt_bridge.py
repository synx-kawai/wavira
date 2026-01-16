#!/usr/bin/env python3
"""
Serial-to-MQTT bridge for ESP32 CSI data.

Reads CSI data from ESP32 serial port and publishes to MQTT broker.

Usage:
    python scripts/serial_mqtt_bridge.py --port /dev/cu.usbmodem21201
    python scripts/serial_mqtt_bridge.py --port /dev/cu.usbmodem21201 --mqtt-host localhost
"""

import argparse
import json
import logging
import signal
import sys
import time
from typing import Optional

import serial
import paho.mqtt.client as mqtt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class SerialMQTTBridge:
    """Bridge between ESP32 serial port and MQTT broker."""

    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        mqtt_host: str = "localhost",
        mqtt_port: int = 1883,
        device_id: Optional[str] = None,
    ):
        self.port = port
        self.baudrate = baudrate
        self.mqtt_host = mqtt_host
        self.mqtt_port = mqtt_port
        self.device_id = device_id or f"esp32-{port.split('/')[-1]}"

        self.running = False
        self.serial: Optional[serial.Serial] = None
        self.mqtt_client: Optional[mqtt.Client] = None

        self.packet_count = 0
        self.last_report_time = time.time()

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        logger.info("Shutdown signal received")
        self.running = False

    def _parse_csi_line(self, line: str) -> Optional[dict]:
        """Parse CSI_DATA line from ESP32.

        Format: CSI_DATA,seq,mac,rssi,rate,noise,<extra_fields>,"[I0,Q0,I1,Q1,...]"
        The actual CSI data array is enclosed in quotes and brackets at the end.
        """
        if not line.startswith("CSI_DATA,"):
            return None

        try:
            # Find the JSON array at the end (starts with "[ and ends with ]")
            array_start = line.find('"[')
            if array_start == -1:
                return None

            # Extract JSON array
            data_str = line[array_start:].strip().strip('"')
            data = json.loads(data_str)

            # Parse header fields (before the extra metadata)
            header_part = line[:array_start]
            parts = header_part.split(",")
            if len(parts) < 6:
                return None

            seq = int(parts[1])
            mac = parts[2]
            rssi = int(parts[3])
            rate = int(parts[4])
            noise = int(parts[5])

            return {
                "timestamp": time.time(),
                "device_id": self.device_id,
                "seq": seq,
                "mac": mac,
                "rssi": rssi,
                "rate": rate,
                "noise_floor": noise,
                "data": data,
            }
        except (ValueError, json.JSONDecodeError) as e:
            logger.debug(f"Failed to parse CSI line: {e}")
            return None

    def connect(self):
        """Connect to serial port and MQTT broker."""
        # Connect to serial port
        logger.info(f"Connecting to serial port: {self.port}")
        self.serial = serial.Serial(self.port, self.baudrate, timeout=0.5)
        self.serial.reset_input_buffer()

        # Connect to MQTT broker
        logger.info(f"Connecting to MQTT broker: {self.mqtt_host}:{self.mqtt_port}")
        self.mqtt_client = mqtt.Client(client_id=f"wavira-bridge-{self.device_id}")
        self.mqtt_client.connect(self.mqtt_host, self.mqtt_port, 60)
        self.mqtt_client.loop_start()

        logger.info(f"Bridge ready: {self.port} -> wavira/csi/{self.device_id}")

    def run(self):
        """Main bridge loop."""
        self.running = True
        self.connect()

        topic = f"wavira/csi/{self.device_id}"

        try:
            while self.running:
                if self.serial.in_waiting:
                    line = self.serial.readline().decode('utf-8', errors='ignore').strip()

                    if line:
                        csi_data = self._parse_csi_line(line)

                        if csi_data:
                            # Publish to MQTT
                            payload = json.dumps(csi_data)
                            self.mqtt_client.publish(topic, payload, qos=0)
                            self.packet_count += 1

                            # Report stats periodically
                            now = time.time()
                            if now - self.last_report_time >= 5:
                                rate = self.packet_count / (now - self.last_report_time)
                                logger.info(f"Published {self.packet_count} packets ({rate:.1f}/sec)")
                                self.packet_count = 0
                                self.last_report_time = now
                else:
                    time.sleep(0.001)

        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        if self.mqtt_client:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
        if self.serial:
            self.serial.close()
        logger.info("Bridge stopped")


def main():
    parser = argparse.ArgumentParser(description="Serial-to-MQTT bridge for ESP32 CSI data")
    parser.add_argument("--port", "-p", required=True, help="Serial port path")
    parser.add_argument("--baudrate", "-b", type=int, default=115200, help="Serial baudrate")
    parser.add_argument("--mqtt-host", default="localhost", help="MQTT broker host")
    parser.add_argument("--mqtt-port", type=int, default=1883, help="MQTT broker port")
    parser.add_argument("--device-id", help="Device ID for MQTT topic")

    args = parser.parse_args()

    bridge = SerialMQTTBridge(
        port=args.port,
        baudrate=args.baudrate,
        mqtt_host=args.mqtt_host,
        mqtt_port=args.mqtt_port,
        device_id=args.device_id,
    )

    bridge.run()


if __name__ == "__main__":
    main()

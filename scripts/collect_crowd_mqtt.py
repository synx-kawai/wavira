#!/usr/bin/env python3
"""
MQTT-based CSI data collection for crowd estimation.

Usage:
    python scripts/collect_crowd_mqtt.py --level 0 --location office --samples 100
    python scripts/collect_crowd_mqtt.py --level 1 --mqtt-host 13.115.255.111

Crowd levels:
    0: Empty (0-1 people)
    1: Moderate (2-5 people)
    2: Crowded (6+ people)
"""

import sys
import os
import time
import json
import numpy as np
import h5py
import argparse
import logging
import signal
from datetime import datetime
from typing import Optional, List, Dict
from collections import deque

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import paho.mqtt.client as mqtt
except ImportError:
    print("Please install paho-mqtt: pip install paho-mqtt")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


CROWD_LEVELS = {
    0: "empty",      # 0-1 people
    1: "moderate",   # 2-5 people
    2: "crowded",    # 6+ people
}

DEFAULT_NUM_PEOPLE = {
    0: 0,
    1: 3,
    2: 8,
}


class MQTTCrowdCollector:
    """MQTT-based CSI data collector for crowd level estimation."""

    def __init__(
        self,
        mqtt_host: str,
        mqtt_port: int,
        output_dir: str,
        level: int,
        location: str,
        device_id: Optional[str] = None,
        num_people: Optional[int] = None,
        samples_per_file: int = 100,
        num_files: int = 10,
        timeout_sec: int = 300,
    ):
        self.mqtt_host = mqtt_host
        self.mqtt_port = mqtt_port
        self.level = level
        self.level_name = CROWD_LEVELS.get(level, f"level_{level}")
        self.location = location
        self.device_id = device_id
        self.num_people = num_people if num_people is not None else DEFAULT_NUM_PEOPLE.get(level, 0)
        self.output_dir = output_dir
        self.samples_per_file = samples_per_file
        self.num_files = num_files
        self.timeout_sec = timeout_sec

        # Data buffers
        self.csi_buffer: List[np.ndarray] = []
        self.rssi_buffer: List[int] = []
        self.collected_files = 0

        # Control flags
        self.running = False
        self.connected = False

        # MQTT client
        self.client = mqtt.Client(client_id=f"wavira-collector-{int(time.time())}")
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info("Received shutdown signal, stopping...")
        self.running = False

    def _on_connect(self, client, userdata, flags, rc):
        """MQTT connect callback."""
        if rc == 0:
            self.connected = True
            topic = f"wavira/csi/{self.device_id}" if self.device_id else "wavira/csi/#"
            client.subscribe(topic)
            logger.info(f"Connected to MQTT, subscribed to {topic}")
        else:
            logger.error(f"MQTT connection failed: rc={rc}")

    def _on_disconnect(self, client, userdata, rc):
        """MQTT disconnect callback."""
        self.connected = False
        if rc != 0:
            logger.warning(f"Unexpected MQTT disconnect: rc={rc}")

    def _on_message(self, client, userdata, msg):
        """Process incoming CSI message."""
        try:
            data = json.loads(msg.payload.decode())

            # Extract CSI data
            csi_raw = data.get("data", [])
            rssi = data.get("rssi", -100)

            if not csi_raw:
                return

            # Convert to numpy array and compute amplitude
            csi_array = np.array(csi_raw, dtype=np.float32)

            # CSI data is I/Q pairs: [I0, Q0, I1, Q1, ...]
            if len(csi_array) % 2 == 0:
                n_subcarriers = len(csi_array) // 2
                csi_complex = csi_array[0::2] + 1j * csi_array[1::2]
                amplitude = np.abs(csi_complex)
            else:
                amplitude = csi_array

            self.csi_buffer.append(amplitude)
            self.rssi_buffer.append(rssi)

        except json.JSONDecodeError:
            pass
        except Exception as e:
            logger.debug(f"Error processing message: {e}")

    def _save_file(self) -> bool:
        """Save collected samples to HDF5 file."""
        if len(self.csi_buffer) < self.samples_per_file:
            return False

        # Take samples for this file
        samples = self.csi_buffer[:self.samples_per_file]
        rssi_samples = self.rssi_buffer[:self.samples_per_file]

        # Remove used samples
        self.csi_buffer = self.csi_buffer[self.samples_per_file:]
        self.rssi_buffer = self.rssi_buffer[self.samples_per_file:]

        # Pad/truncate to uniform shape
        max_len = max(len(s) for s in samples)
        padded = np.zeros((len(samples), max_len), dtype=np.float32)
        for i, s in enumerate(samples):
            padded[i, :len(s)] = s

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.level_name}_{self.location}_{timestamp}_{self.collected_files:04d}.h5"
        filepath = os.path.join(self.output_dir, filename)

        # Save to HDF5
        with h5py.File(filepath, 'w') as f:
            f.create_dataset('amplitudes', data=padded, compression='gzip')
            f.create_dataset('rssi', data=np.array(rssi_samples, dtype=np.int16))
            f.attrs['level'] = self.level
            f.attrs['level_name'] = self.level_name
            f.attrs['num_people'] = self.num_people
            f.attrs['location'] = self.location
            f.attrs['timestamp'] = timestamp
            f.attrs['samples'] = len(samples)
            f.attrs['n_subcarriers'] = max_len

        self.collected_files += 1
        logger.info(f"Saved {filepath} ({self.collected_files}/{self.num_files})")

        return True

    def collect(self):
        """Run the collection loop."""
        logger.info(f"Starting collection: level={self.level} ({self.level_name}), "
                   f"num_people={self.num_people}, location={self.location}")
        logger.info(f"Target: {self.num_files} files x {self.samples_per_file} samples")
        logger.info(f"MQTT: {self.mqtt_host}:{self.mqtt_port}")

        self.running = True
        start_time = time.time()

        # Connect to MQTT
        try:
            self.client.connect(self.mqtt_host, self.mqtt_port, 60)
            self.client.loop_start()
        except Exception as e:
            logger.error(f"Failed to connect to MQTT: {e}")
            return

        try:
            while self.running and self.collected_files < self.num_files:
                # Check timeout
                if time.time() - start_time > self.timeout_sec:
                    logger.warning(f"Timeout after {self.timeout_sec}s")
                    break

                # Try to save a file if we have enough samples
                if len(self.csi_buffer) >= self.samples_per_file:
                    self._save_file()

                # Progress update
                if len(self.csi_buffer) > 0 and len(self.csi_buffer) % 10 == 0:
                    logger.info(f"Buffer: {len(self.csi_buffer)}/{self.samples_per_file} samples")

                time.sleep(0.1)

        finally:
            self.client.loop_stop()
            self.client.disconnect()

        logger.info(f"Collection complete: {self.collected_files} files saved to {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect CSI data via MQTT for crowd level estimation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Crowd Levels:
  0: Empty (0-1 people)
  1: Moderate (2-5 people)
  2: Crowded (6+ people)

Example:
  python scripts/collect_crowd_mqtt.py --level 0 --location office
  python scripts/collect_crowd_mqtt.py --level 1 --mqtt-host 13.115.255.111
        """
    )

    parser.add_argument('-l', '--level', type=int, required=True, choices=[0, 1, 2],
                       help='Crowd level (0=empty, 1=moderate, 2=crowded)')
    parser.add_argument('--num-people', type=int, default=None,
                       help='Exact number of people (overrides level default)')
    parser.add_argument('--location', type=str, default='room1',
                       help='Location identifier')
    parser.add_argument('--device-id', type=str, default=None,
                       help='Specific device ID to collect from (default: all)')
    parser.add_argument('--mqtt-host', type=str, default='localhost',
                       help='MQTT broker host')
    parser.add_argument('--mqtt-port', type=int, default=1883,
                       help='MQTT broker port')
    parser.add_argument('-o', '--output', type=str, default='data/crowd',
                       help='Output directory')
    parser.add_argument('-n', '--num-files', type=int, default=10,
                       help='Number of files to collect')
    parser.add_argument('-s', '--samples', type=int, default=100,
                       help='Samples per file')
    parser.add_argument('-t', '--timeout', type=int, default=300,
                       help='Timeout in seconds')

    args = parser.parse_args()

    collector = MQTTCrowdCollector(
        mqtt_host=args.mqtt_host,
        mqtt_port=args.mqtt_port,
        output_dir=args.output,
        level=args.level,
        location=args.location,
        device_id=args.device_id,
        num_people=args.num_people,
        samples_per_file=args.samples,
        num_files=args.num_files,
        timeout_sec=args.timeout,
    )

    collector.collect()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Crowd level data collection script for CSI-based crowd estimation.

Usage:
    python scripts/collect_crowd.py --level 0 --location office --samples 100
    python scripts/collect_crowd.py --level 1 --location office --samples 100
    python scripts/collect_crowd.py --level 2 --location office --samples 100

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
import argparse
import logging
import signal
from datetime import datetime
from typing import Optional, List

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wavira.utils.esp32_serial import ESP32Serial, CSIPacket

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


class CrowdDataCollector:
    """
    Robust CSI data collector for crowd level estimation.

    Features:
    - Automatic reconnection on connection loss
    - Progress tracking
    - Graceful shutdown
    - Timeout handling
    """

    def __init__(
        self,
        port: Optional[str],
        baud_rate: int,
        output_dir: str,
        level: int,
        location: str,
        samples_per_file: int = 100,
        num_files: int = 10,
        timeout_sec: int = 300,
    ):
        self.port = port
        self.baud_rate = baud_rate
        self.level = level
        self.level_name = CROWD_LEVELS.get(level, f"level_{level}")
        self.location = location
        self.samples_per_file = samples_per_file
        self.num_files = num_files
        self.timeout_sec = timeout_sec

        # Create output directory
        self.save_dir = os.path.join(output_dir, location, self.level_name)
        os.makedirs(self.save_dir, exist_ok=True)

        # Session ID
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Collection state
        self._buffer: List[List[complex]] = []
        self._file_count = 0
        self._total_packets = 0
        self._collecting = False
        self._start_time = 0
        self._last_packet_time = 0

        # ESP32 handler
        self._esp32: Optional[ESP32Serial] = None
        self._shutdown_requested = False

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info("Shutdown requested...")
        self._shutdown_requested = True
        self._collecting = False

    def _on_csi_packet(self, packet: CSIPacket):
        """Handle incoming CSI packet."""
        if not self._collecting:
            return

        self._buffer.append(packet.csi_data)
        self._total_packets += 1
        self._last_packet_time = time.time()

        # Progress display
        progress = len(self._buffer) / self.samples_per_file * 100
        overall = (self._file_count * self.samples_per_file + len(self._buffer)) / \
                  (self.num_files * self.samples_per_file) * 100
        print(f"\rFile {self._file_count + 1}/{self.num_files} | "
              f"Packets: {len(self._buffer)}/{self.samples_per_file} | "
              f"Overall: {overall:.1f}%", end="", flush=True)

        # Save when buffer is full
        if len(self._buffer) >= self.samples_per_file:
            self._save_buffer()

    def _on_connection_state(self, state):
        """Handle connection state changes."""
        logger.info(f"Connection state: {state.value}")

    def _on_error(self, error: Exception):
        """Handle errors."""
        logger.error(f"Error: {error}")

    def _save_buffer(self):
        """Save current buffer to file."""
        if not self._buffer:
            return

        try:
            # Convert to numpy array
            data_array = np.array(self._buffer)
            data_array = np.transpose(data_array, (1, 0))
            data_array = np.expand_dims(data_array, axis=0)

            # Save data
            filename = f"{self.session_id}_{self._file_count:04d}.npy"
            filepath = os.path.join(self.save_dir, filename)
            np.save(filepath, data_array)

            # Save metadata
            meta = {
                "level": self.level,
                "level_name": self.level_name,
                "location": self.location,
                "session_id": self.session_id,
                "file_index": self._file_count,
                "samples": len(self._buffer),
                "timestamp": datetime.now().isoformat(),
            }
            meta_path = filepath.replace(".npy", ".json")
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)

            print(f"\n  Saved: {filename}")

            self._buffer = []
            self._file_count += 1

        except Exception as e:
            logger.error(f"Save error: {e}")

    def collect(self) -> dict:
        """
        Run data collection.

        Returns:
            Collection statistics
        """
        # Print header
        print("=" * 60)
        print("Crowd Level Data Collection")
        print("=" * 60)
        print(f"Location:    {self.location}")
        print(f"Level:       {self.level} ({self.level_name})")
        print(f"Output:      {self.save_dir}")
        print(f"Target:      {self.num_files} files x {self.samples_per_file} samples")
        print(f"Session:     {self.session_id}")
        print(f"Timeout:     {self.timeout_sec}s")
        print("=" * 60)

        # Confirm with user
        try:
            input(f"\nPress ENTER to start collecting '{self.level_name}' data...")
        except EOFError:
            pass

        # Create ESP32 handler
        self._esp32 = ESP32Serial(
            port=self.port,
            baud_rate=self.baud_rate,
            timeout=1.0,
            auto_reconnect=True,
            max_reconnect_attempts=10,
            reconnect_delay=2.0,
        )

        # Register callbacks
        self._esp32.add_csi_callback(self._on_csi_packet)
        self._esp32.add_state_callback(self._on_connection_state)
        self._esp32.add_error_callback(self._on_error)

        # Connect
        print(f"\nConnecting to {self.port or 'auto-detect'}...")

        if not self._esp32.connect():
            logger.error("Failed to connect to ESP32")
            return self._get_stats()

        print("Connected! Collecting CSI data...")
        print("Press Ctrl+C to stop.\n")

        # Start collection
        self._collecting = True
        self._start_time = time.time()
        self._last_packet_time = time.time()
        self._esp32.start()

        # Wait for collection to complete
        try:
            while self._collecting and not self._shutdown_requested:
                time.sleep(0.1)

                # Check completion
                if self._file_count >= self.num_files:
                    logger.info("Collection target reached")
                    break

                # Check timeout
                elapsed = time.time() - self._start_time
                if elapsed > self.timeout_sec:
                    logger.warning(f"Timeout reached ({self.timeout_sec}s)")
                    break

                # Check data timeout (no data for 30s)
                if time.time() - self._last_packet_time > 30.0:
                    logger.warning("No data received for 30s, checking connection...")
                    self._last_packet_time = time.time()

        except KeyboardInterrupt:
            print("\n\nCollection interrupted by user.")

        finally:
            # Stop and cleanup
            self._collecting = False
            if self._esp32:
                self._esp32.stop()

            # Save remaining buffer
            if self._buffer:
                self._save_buffer()

        # Print summary
        stats = self._get_stats()
        print("\n" + "=" * 60)
        print("Collection Complete")
        print(f"  Files saved: {stats['files_saved']}")
        print(f"  Total packets: {stats['total_packets']}")
        print(f"  Duration: {stats['duration']:.1f}s")
        print(f"  Location: {self.save_dir}")
        print("=" * 60)

        return stats

    def _get_stats(self) -> dict:
        """Get collection statistics."""
        return {
            "files_saved": self._file_count,
            "total_packets": self._total_packets,
            "duration": time.time() - self._start_time if self._start_time else 0,
            "output_dir": self.save_dir,
            "session_id": self.session_id,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Collect CSI data for crowd level estimation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Crowd Levels:
  0: Empty (0-1 people)
  1: Moderate (2-5 people)
  2: Crowded (6+ people)

Example:
  python scripts/collect_crowd.py --level 0 --location office
  python scripts/collect_crowd.py --level 1 --location office
  python scripts/collect_crowd.py --level 2 --location office
        """
    )

    parser.add_argument(
        "-l", "--level",
        type=int,
        required=True,
        choices=[0, 1, 2],
        help="Crowd level (0=empty, 1=moderate, 2=crowded)"
    )
    parser.add_argument(
        "--location",
        type=str,
        default="room1",
        help="Location identifier (default: room1)"
    )
    parser.add_argument(
        "-p", "--port",
        default=None,
        help="Serial port (auto-detect if not specified)"
    )
    parser.add_argument(
        "-b", "--baud",
        type=int,
        default=115200,
        help="Baud rate (default: 115200)"
    )
    parser.add_argument(
        "-o", "--output",
        default="data/crowd",
        help="Output directory (default: data/crowd)"
    )
    parser.add_argument(
        "-n", "--num-files",
        type=int,
        default=10,
        help="Number of files to collect (default: 10)"
    )
    parser.add_argument(
        "-s", "--samples",
        type=int,
        default=100,
        help="Samples per file (default: 100)"
    )
    parser.add_argument(
        "-t", "--timeout",
        type=int,
        default=300,
        help="Timeout in seconds (default: 300)"
    )

    args = parser.parse_args()

    collector = CrowdDataCollector(
        port=args.port,
        baud_rate=args.baud,
        output_dir=args.output,
        level=args.level,
        location=args.location,
        samples_per_file=args.samples,
        num_files=args.num_files,
        timeout_sec=args.timeout,
    )

    collector.collect()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
CSI Data Collector for ESP32

Collects Channel State Information (CSI) data from ESP32 via serial port
and saves it in a format compatible with the Wavira pipeline.
"""

import argparse
import serial
import time
import numpy as np
import re
import sys
from datetime import datetime
from typing import Optional, List, Tuple
import json


class CSICollector:
    """Collects CSI data from ESP32 via serial port."""

    # CSI data format for ESP32 (non-C5/C6)
    CSI_COLUMNS = [
        'type', 'id', 'mac', 'rssi', 'rate', 'sig_mode', 'mcs', 'bandwidth',
        'smoothing', 'not_sounding', 'aggregation', 'stbc', 'fec_coding',
        'sgi', 'noise_floor', 'ampdu_cnt', 'channel', 'secondary_channel',
        'local_timestamp', 'ant', 'sig_len', 'rx_state', 'len', 'first_word', 'data'
    ]

    def __init__(
        self,
        port: str = '/dev/cu.usbserial-5AE90127161',
        baudrate: int = 921600,
        timeout: float = 1.0
    ):
        """
        Initialize CSI Collector.

        Args:
            port: Serial port path
            baudrate: Serial communication baud rate
            timeout: Read timeout in seconds
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial: Optional[serial.Serial] = None
        self.csi_data: List[dict] = []

    def connect(self) -> bool:
        """Connect to ESP32 via serial port."""
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                rtscts=False,
                dsrdtr=False
            )
            # Clear buffers
            self.serial.reset_input_buffer()
            self.serial.reset_output_buffer()
            print(f"Connected to {self.port} at {self.baudrate} baud")
            return True
        except Exception as e:
            print(f"Failed to connect: {e}")
            return False

    def disconnect(self):
        """Disconnect from serial port."""
        if self.serial and self.serial.is_open:
            self.serial.close()
            print("Disconnected")

    def reset_esp32(self):
        """Reset ESP32 using RTS pin."""
        if self.serial:
            self.serial.setRTS(True)
            time.sleep(0.1)
            self.serial.setRTS(False)
            time.sleep(1.0)  # Wait for boot
            self.serial.reset_input_buffer()
            print("ESP32 reset")

    def parse_csi_line(self, line: str) -> Optional[dict]:
        """
        Parse a line of CSI data.

        Args:
            line: Raw CSI data line from serial

        Returns:
            Parsed CSI data dictionary or None if parsing fails
        """
        if not line.startswith('CSI_DATA'):
            return None

        try:
            # Extract the data array from the line
            # Format: CSI_DATA,id,mac,rssi,...,len,first_word,"[data,...]"

            # Find the data array
            data_match = re.search(r'\"\[([-\d,\s]+)\]\"', line)
            if not data_match:
                return None

            # Parse the raw CSI values
            data_str = data_match.group(1)
            raw_values = [int(x.strip()) for x in data_str.split(',')]

            # Parse metadata (everything before the data array)
            metadata_str = line[:data_match.start()].rstrip(',')
            parts = metadata_str.split(',')

            if len(parts) < 24:  # Minimum expected fields
                return None

            # Convert raw CSI values to complex numbers
            # Format: [imag1, real1, imag2, real2, ...]
            n_subcarriers = len(raw_values) // 2
            csi_complex = np.zeros(n_subcarriers, dtype=np.complex64)

            for i in range(n_subcarriers):
                imag = raw_values[2 * i]
                real = raw_values[2 * i + 1]
                csi_complex[i] = complex(real, imag)

            return {
                'type': parts[0],
                'id': int(parts[1]),
                'mac': parts[2],
                'rssi': int(parts[3]),
                'rate': int(parts[4]),
                'channel': int(parts[16]) if len(parts) > 16 else 0,
                'timestamp': int(parts[18]) if len(parts) > 18 else 0,
                'n_subcarriers': n_subcarriers,
                'csi_complex': csi_complex,
                'csi_amplitude': np.abs(csi_complex),
                'csi_phase': np.angle(csi_complex),
            }

        except Exception as e:
            # Silently skip malformed lines
            return None

    def collect(
        self,
        duration: float = 10.0,
        max_frames: int = 1000,
        verbose: bool = True
    ) -> List[dict]:
        """
        Collect CSI data for specified duration.

        Args:
            duration: Collection duration in seconds
            max_frames: Maximum number of frames to collect
            verbose: Print progress information

        Returns:
            List of parsed CSI data frames
        """
        if not self.serial or not self.serial.is_open:
            print("Not connected")
            return []

        self.csi_data = []
        start_time = time.time()
        frame_count = 0
        line_count = 0

        if verbose:
            print(f"Collecting CSI data for {duration}s (max {max_frames} frames)...")
            print("Waiting for data...")

        while (time.time() - start_time) < duration and frame_count < max_frames:
            try:
                if self.serial.in_waiting > 0:
                    line = self.serial.readline().decode('utf-8', errors='ignore').strip()
                    line_count += 1

                    if line:
                        # Print non-CSI lines for debugging
                        if verbose and not line.startswith('CSI_DATA') and len(line) > 5:
                            # Filter out garbled boot messages
                            if all(ord(c) < 128 for c in line[:min(10, len(line))]):
                                print(f"[ESP32] {line[:100]}")

                        parsed = self.parse_csi_line(line)
                        if parsed:
                            self.csi_data.append(parsed)
                            frame_count += 1
                            if verbose and frame_count % 10 == 0:
                                print(f"  Collected {frame_count} CSI frames...")
                else:
                    time.sleep(0.01)

            except Exception as e:
                if verbose:
                    print(f"Read error: {e}")
                break

        if verbose:
            print(f"\nCollection complete:")
            print(f"  Total lines read: {line_count}")
            print(f"  CSI frames collected: {frame_count}")
            if self.csi_data:
                print(f"  Subcarriers per frame: {self.csi_data[0]['n_subcarriers']}")

        return self.csi_data

    def to_numpy(self) -> Tuple[np.ndarray, dict]:
        """
        Convert collected CSI data to numpy array.

        Returns:
            Tuple of (csi_array, metadata) where csi_array has shape
            (n_frames, n_subcarriers) containing complex CSI values
        """
        if not self.csi_data:
            return np.array([]), {}

        n_frames = len(self.csi_data)
        n_subcarriers = self.csi_data[0]['n_subcarriers']

        csi_array = np.zeros((n_frames, n_subcarriers), dtype=np.complex64)
        rssi_array = np.zeros(n_frames)
        timestamps = np.zeros(n_frames)

        for i, frame in enumerate(self.csi_data):
            csi_array[i, :len(frame['csi_complex'])] = frame['csi_complex']
            rssi_array[i] = frame['rssi']
            timestamps[i] = frame['timestamp']

        metadata = {
            'n_frames': n_frames,
            'n_subcarriers': n_subcarriers,
            'mac': self.csi_data[0]['mac'] if self.csi_data else '',
            'channel': self.csi_data[0]['channel'] if self.csi_data else 0,
            'rssi_mean': float(np.mean(rssi_array)),
            'rssi_std': float(np.std(rssi_array)),
        }

        return csi_array, metadata

    def save(self, filename: str):
        """
        Save collected CSI data to file.

        Args:
            filename: Output filename (.npz for numpy format)
        """
        csi_array, metadata = self.to_numpy()

        if csi_array.size == 0:
            print("No data to save")
            return

        # Save as numpy compressed file
        np.savez_compressed(
            filename,
            csi_complex=csi_array,
            csi_amplitude=np.abs(csi_array),
            csi_phase=np.angle(csi_array),
            metadata=json.dumps(metadata)
        )
        print(f"Saved {len(self.csi_data)} frames to {filename}")


def main():
    parser = argparse.ArgumentParser(description='Collect CSI data from ESP32')
    parser.add_argument(
        '-p', '--port',
        default='/dev/cu.usbserial-5AE90127161',
        help='Serial port'
    )
    parser.add_argument(
        '-b', '--baudrate',
        type=int,
        default=921600,
        help='Baud rate'
    )
    parser.add_argument(
        '-d', '--duration',
        type=float,
        default=10.0,
        help='Collection duration in seconds'
    )
    parser.add_argument(
        '-n', '--max-frames',
        type=int,
        default=1000,
        help='Maximum number of frames to collect'
    )
    parser.add_argument(
        '-o', '--output',
        default=None,
        help='Output filename (default: csi_data_TIMESTAMP.npz)'
    )
    parser.add_argument(
        '-r', '--reset',
        action='store_true',
        help='Reset ESP32 before collecting'
    )

    args = parser.parse_args()

    collector = CSICollector(
        port=args.port,
        baudrate=args.baudrate
    )

    if not collector.connect():
        sys.exit(1)

    try:
        if args.reset:
            collector.reset_esp32()
            time.sleep(3)  # Wait for Wi-Fi connection

        data = collector.collect(
            duration=args.duration,
            max_frames=args.max_frames
        )

        if data:
            if args.output:
                filename = args.output
            else:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'csi_data_{timestamp}.npz'

            collector.save(filename)
        else:
            print("No CSI data collected")
            print("\nTroubleshooting:")
            print("  1. Check if ESP32 is connected to Wi-Fi")
            print("  2. Verify SSID and password are correct")
            print("  3. Try resetting ESP32 with -r flag")
            print("  4. Check if router is reachable")

    finally:
        collector.disconnect()


if __name__ == '__main__':
    main()

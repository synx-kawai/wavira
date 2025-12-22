#!/usr/bin/env python3
"""
ESP32 Serial Communication Module

Provides robust serial communication with ESP32 for CSI data collection.
Handles connection issues, timeouts, and automatic reconnection.
"""

import serial
import serial.tools.list_ports
import time
import json
import threading
import logging
from typing import Optional, Callable, List, Dict, Any
from dataclasses import dataclass
from enum import Enum
import select
import sys

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class CSIPacket:
    """Parsed CSI data packet."""
    sequence_id: int
    mac: str
    rssi: int
    channel: int
    timestamp: int
    csi_data: List[complex]
    raw_line: str


class ESP32Serial:
    """
    Robust ESP32 serial communication handler.

    Features:
    - Automatic port detection
    - Connection monitoring with heartbeat
    - Automatic reconnection
    - Buffer management
    - Graceful shutdown
    """

    def __init__(
        self,
        port: Optional[str] = None,
        baud_rate: int = 115200,
        timeout: float = 1.0,
        auto_reconnect: bool = True,
        max_reconnect_attempts: int = 5,
        reconnect_delay: float = 2.0,
    ):
        """
        Initialize ESP32 serial handler.

        Args:
            port: Serial port (auto-detect if None)
            baud_rate: Baud rate (default: 115200)
            timeout: Read timeout in seconds
            auto_reconnect: Enable automatic reconnection
            max_reconnect_attempts: Max reconnection attempts
            reconnect_delay: Delay between reconnection attempts
        """
        self.port = port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.auto_reconnect = auto_reconnect
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay = reconnect_delay

        self._serial: Optional[serial.Serial] = None
        self._state = ConnectionState.DISCONNECTED
        self._running = False
        self._read_thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable[[CSIPacket], None]] = []
        self._error_callbacks: List[Callable[[Exception], None]] = []
        self._state_callbacks: List[Callable[[ConnectionState], None]] = []
        self._last_data_time = 0
        self._reconnect_count = 0
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

    @property
    def state(self) -> ConnectionState:
        """Get current connection state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._state == ConnectionState.CONNECTED

    def _set_state(self, state: ConnectionState):
        """Update connection state and notify callbacks."""
        if self._state != state:
            self._state = state
            for callback in self._state_callbacks:
                try:
                    callback(state)
                except Exception as e:
                    logger.error(f"State callback error: {e}")

    def add_csi_callback(self, callback: Callable[[CSIPacket], None]):
        """Add callback for CSI data."""
        self._callbacks.append(callback)

    def add_error_callback(self, callback: Callable[[Exception], None]):
        """Add callback for errors."""
        self._error_callbacks.append(callback)

    def add_state_callback(self, callback: Callable[[ConnectionState], None]):
        """Add callback for state changes."""
        self._state_callbacks.append(callback)

    @staticmethod
    def find_esp32_ports() -> List[str]:
        """Find available ESP32 serial ports."""
        ports = []
        for port in serial.tools.list_ports.comports():
            # Common ESP32 identifiers
            if any(x in port.description.lower() for x in ['cp210', 'ch340', 'ftdi', 'usb']):
                ports.append(port.device)
            elif 'usbserial' in port.device.lower():
                ports.append(port.device)
        return ports

    def connect(self) -> bool:
        """
        Connect to ESP32.

        Returns:
            True if connected successfully
        """
        self._set_state(ConnectionState.CONNECTING)

        # Auto-detect port if not specified
        if not self.port:
            ports = self.find_esp32_ports()
            if not ports:
                logger.error("No ESP32 ports found")
                self._set_state(ConnectionState.ERROR)
                return False
            self.port = ports[0]
            logger.info(f"Auto-detected port: {self.port}")

        try:
            # Close existing connection
            if self._serial and self._serial.is_open:
                self._serial.close()

            # Open new connection
            self._serial = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                timeout=self.timeout,
                write_timeout=self.timeout,
            )

            # Reset ESP32
            self._reset_esp32()

            # Clear buffers
            self._serial.reset_input_buffer()
            self._serial.reset_output_buffer()

            self._set_state(ConnectionState.CONNECTED)
            self._last_data_time = time.time()
            self._reconnect_count = 0
            logger.info(f"Connected to {self.port}")
            return True

        except serial.SerialException as e:
            logger.error(f"Connection failed: {e}")
            self._set_state(ConnectionState.ERROR)
            self._notify_error(e)
            return False

    def _reset_esp32(self):
        """Perform soft reset on ESP32."""
        if self._serial and self._serial.is_open:
            self._serial.setDTR(False)
            self._serial.setRTS(True)
            time.sleep(0.1)
            self._serial.setRTS(False)
            time.sleep(0.5)
            logger.debug("ESP32 reset complete")

    def disconnect(self):
        """Disconnect from ESP32."""
        with self._lock:
            if self._serial and self._serial.is_open:
                try:
                    self._serial.close()
                except Exception as e:
                    logger.warning(f"Error closing serial: {e}")
            self._set_state(ConnectionState.DISCONNECTED)
            logger.info("Disconnected")

    def start(self):
        """Start reading data in background thread."""
        if self._running:
            return

        if not self.is_connected:
            if not self.connect():
                return

        self._running = True
        self._stop_event.clear()
        self._read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._read_thread.start()
        logger.info("Started reading")

    def stop(self):
        """Stop reading and disconnect."""
        self._running = False
        self._stop_event.set()
        if self._read_thread:
            self._read_thread.join(timeout=3.0)
            if self._read_thread.is_alive():
                logger.warning("Read thread did not terminate cleanly")
        self.disconnect()
        logger.info("Stopped")

    def _read_loop(self):
        """Main reading loop with non-blocking reads for responsive shutdown."""
        buffer = b""

        while self._running and not self._stop_event.is_set():
            try:
                if not self._serial or not self._serial.is_open:
                    if self.auto_reconnect:
                        self._try_reconnect()
                    else:
                        break
                    continue

                # Check for data timeout (possible disconnect)
                if time.time() - self._last_data_time > 10.0:
                    logger.warning("No data received for 10s, checking connection...")
                    if not self._check_connection():
                        continue

                # Non-blocking read: check if data is available first
                try:
                    waiting = self._serial.in_waiting
                except (serial.SerialException, OSError) as e:
                    logger.error(f"Port error: {e}")
                    self._set_state(ConnectionState.ERROR)
                    if self.auto_reconnect:
                        self._try_reconnect()
                    continue

                if waiting == 0:
                    # No data available, wait briefly and check shutdown flag
                    if self._stop_event.wait(timeout=0.05):
                        break
                    continue

                # Read available data
                try:
                    data = self._serial.read(waiting)
                    if not data:
                        continue
                    buffer += data
                except serial.SerialException as e:
                    logger.error(f"Read error: {e}")
                    self._set_state(ConnectionState.ERROR)
                    if self.auto_reconnect:
                        self._try_reconnect()
                    continue

                self._last_data_time = time.time()

                # Process complete lines from buffer
                while b'\n' in buffer:
                    line, buffer = buffer.split(b'\n', 1)
                    try:
                        decoded = line.decode('utf-8', errors='ignore').strip()
                        if decoded and "CSI_DATA" in decoded:
                            packet = self._parse_csi_line(decoded)
                            if packet:
                                self._notify_csi(packet)
                    except Exception as e:
                        logger.debug(f"Parse error: {e}")

            except Exception as e:
                logger.error(f"Read loop error: {e}")
                self._notify_error(e)
                if self._stop_event.wait(timeout=0.1):
                    break

    def _check_connection(self) -> bool:
        """Check if connection is still alive."""
        try:
            if self._serial and self._serial.is_open:
                # Try to get port status
                self._serial.in_waiting
                return True
        except Exception:
            pass

        self._set_state(ConnectionState.ERROR)
        return False

    def _try_reconnect(self):
        """Attempt to reconnect."""
        if self._reconnect_count >= self.max_reconnect_attempts:
            logger.error("Max reconnection attempts reached")
            self._running = False
            return

        self._reconnect_count += 1
        logger.info(f"Reconnection attempt {self._reconnect_count}/{self.max_reconnect_attempts}")

        time.sleep(self.reconnect_delay)

        if self.connect():
            logger.info("Reconnected successfully")
        else:
            logger.warning("Reconnection failed")

    def _parse_csi_line(self, line: str) -> Optional[CSIPacket]:
        """Parse a CSI data line."""
        try:
            idx = line.find("CSI_DATA")
            if idx < 0:
                return None

            csv_part = line[idx:]
            parts = csv_part.split(',')

            if len(parts) < 25:
                return None

            # Extract JSON data
            json_str = parts[-1].strip()
            if json_str.startswith('"'):
                json_str = json_str[1:]
            if json_str.endswith('"'):
                json_str = json_str[:-1]

            csi_raw = json.loads(json_str)
            csi_len = int(parts[-3])

            if len(csi_raw) != csi_len:
                return None

            # Convert to complex
            csi_complex = []
            for i in range(csi_len // 2):
                real = csi_raw[i * 2 + 1]
                imag = csi_raw[i * 2]
                csi_complex.append(complex(real, imag))

            return CSIPacket(
                sequence_id=int(parts[1]),
                mac=parts[2] if len(parts) > 2 else "",
                rssi=int(parts[3]) if len(parts) > 3 else 0,
                channel=int(parts[15]) if len(parts) > 15 else 0,
                timestamp=int(parts[17]) if len(parts) > 17 else 0,
                csi_data=csi_complex,
                raw_line=line,
            )

        except Exception as e:
            logger.debug(f"Parse error: {e}")
            return None

    def _notify_csi(self, packet: CSIPacket):
        """Notify CSI callbacks."""
        for callback in self._callbacks:
            try:
                callback(packet)
            except Exception as e:
                logger.error(f"CSI callback error: {e}")

    def _notify_error(self, error: Exception):
        """Notify error callbacks."""
        for callback in self._error_callbacks:
            try:
                callback(error)
            except Exception as e:
                logger.error(f"Error callback error: {e}")

    def read_csi_blocking(self, timeout: float = 5.0) -> Optional[CSIPacket]:
        """
        Read a single CSI packet (blocking but interruptible).

        Args:
            timeout: Read timeout in seconds

        Returns:
            CSIPacket or None if timeout
        """
        if not self._serial or not self._serial.is_open:
            return None

        start = time.time()
        buffer = b""
        poll_interval = 0.05  # 50ms polling for responsive interruption

        while time.time() - start < timeout:
            try:
                # Check if data is available (non-blocking)
                waiting = self._serial.in_waiting
                if waiting == 0:
                    time.sleep(poll_interval)
                    continue

                # Read available data
                data = self._serial.read(waiting)
                if data:
                    buffer += data

                # Process complete lines
                while b'\n' in buffer:
                    line, buffer = buffer.split(b'\n', 1)
                    decoded = line.decode('utf-8', errors='ignore').strip()
                    if "CSI_DATA" in decoded:
                        return self._parse_csi_line(decoded)

            except (serial.SerialException, OSError):
                return None
            except Exception:
                pass

        return None

    def flush_buffers(self):
        """Flush serial buffers."""
        if self._serial and self._serial.is_open:
            try:
                self._serial.reset_input_buffer()
                self._serial.reset_output_buffer()
            except Exception as e:
                logger.warning(f"Flush error: {e}")


class CSICollector:
    """
    High-level CSI data collector with automatic saving.
    """

    def __init__(
        self,
        esp32: ESP32Serial,
        output_dir: str = "data/csi",
        samples_per_file: int = 100,
    ):
        self.esp32 = esp32
        self.output_dir = output_dir
        self.samples_per_file = samples_per_file

        self._buffer: List[CSIPacket] = []
        self._file_count = 0
        self._total_packets = 0
        self._collecting = False
        self._metadata: Dict[str, Any] = {}

        # Register callback
        self.esp32.add_csi_callback(self._on_csi_packet)

    def start_collection(self, metadata: Optional[Dict[str, Any]] = None):
        """Start collecting CSI data."""
        import os
        os.makedirs(self.output_dir, exist_ok=True)

        self._buffer = []
        self._file_count = 0
        self._total_packets = 0
        self._metadata = metadata or {}
        self._collecting = True

        if not self.esp32.is_connected:
            self.esp32.connect()
        self.esp32.start()

    def stop_collection(self) -> Dict[str, Any]:
        """Stop collecting and return stats."""
        self._collecting = False
        self.esp32.stop()

        # Save remaining buffer
        if self._buffer:
            self._save_buffer()

        return {
            "files_saved": self._file_count,
            "total_packets": self._total_packets,
            "output_dir": self.output_dir,
        }

    def _on_csi_packet(self, packet: CSIPacket):
        """Handle incoming CSI packet."""
        if not self._collecting:
            return

        self._buffer.append(packet)
        self._total_packets += 1

        if len(self._buffer) >= self.samples_per_file:
            self._save_buffer()

    def _save_buffer(self):
        """Save current buffer to file."""
        import numpy as np
        from datetime import datetime

        if not self._buffer:
            return

        # Convert to numpy array
        data = np.array([p.csi_data for p in self._buffer])
        data = np.transpose(data, (1, 0))
        data = np.expand_dims(data, axis=0)

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"csi_{timestamp}_{self._file_count:04d}.npy"
        filepath = f"{self.output_dir}/{filename}"

        np.save(filepath, data)

        # Save metadata
        meta = {
            **self._metadata,
            "file_index": self._file_count,
            "samples": len(self._buffer),
            "timestamp": datetime.now().isoformat(),
        }
        meta_path = filepath.replace(".npy", ".json")
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Saved: {filename} ({len(self._buffer)} samples)")

        self._buffer = []
        self._file_count += 1


def test_connection(port: Optional[str] = None, timeout: float = 10.0) -> bool:
    """
    Test ESP32 connection and CSI data reception.

    Args:
        port: Serial port (auto-detect if None)
        timeout: Read timeout in seconds (default: 10.0)

    Returns:
        True if connection and data reception successful
    """
    import signal

    print("=" * 50)
    print("ESP32 Connection Test")
    print("=" * 50)
    print("Press Ctrl+C to cancel at any time.\n")

    esp32 = None
    interrupted = False

    def signal_handler(signum, frame):
        nonlocal interrupted
        interrupted = True
        print("\n\nTest interrupted by user.")

    # Setup signal handler
    old_handler = signal.signal(signal.SIGINT, signal_handler)

    try:
        # Find ports
        ports = ESP32Serial.find_esp32_ports()
        print(f"Available ports: {ports}")

        if not port:
            if ports:
                port = ports[0]
            else:
                print("ERROR: No ESP32 ports found")
                return False

        print(f"Testing port: {port}")

        # Create handler
        esp32 = ESP32Serial(port=port, timeout=2.0)

        # Connect
        if not esp32.connect():
            print("ERROR: Connection failed")
            return False

        print("Connected! Waiting for CSI data...")

        # Try to read CSI data with interrupt check
        start = time.time()
        packet = None
        while not interrupted and (time.time() - start) < timeout:
            packet = esp32.read_csi_blocking(timeout=1.0)
            if packet:
                break

        if interrupted:
            return False

        if packet:
            print(f"SUCCESS: Received CSI packet")
            print(f"  Sequence: {packet.sequence_id}")
            print(f"  RSSI: {packet.rssi} dBm")
            print(f"  CSI length: {len(packet.csi_data)}")
            return True
        else:
            print("WARNING: No CSI data received (timeout)")
            return False

    finally:
        # Restore signal handler
        signal.signal(signal.SIGINT, old_handler)
        # Cleanup
        if esp32:
            esp32.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_connection()

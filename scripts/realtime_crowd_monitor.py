#!/usr/bin/env python3
"""
Real-time Crowd Level Monitor

Continuously displays crowd level predictions using real ESP32 CSI data.

Usage:
    # Use ESP32 serial port directly
    python scripts/realtime_crowd_monitor.py --port /dev/cu.usbmodem21101

    # Use multiple ESP32 devices
    python scripts/realtime_crowd_monitor.py --port /dev/cu.usbmodem21101 --port /dev/cu.usbmodem21201

    # Simulation mode (no hardware)
    python scripts/realtime_crowd_monitor.py --simulate
"""

import sys
import os
import time
import argparse
import json
import threading
from collections import deque
from typing import Optional, List

import numpy as np
import torch
import serial

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wavira.models.crowd_estimator import CrowdEstimator, CrowdEstimatorConfig

CROWD_LABELS = {
    0: "Empty (0 people)",
    1: "Low (1-2 people)",
    2: "Medium (3-5 people)",
    3: "High (6+ people)",
}

CROWD_ICONS = {
    0: "    ",
    1: "👤  ",
    2: "👥  ",
    3: "👪👪",
}


class ESP32Reader:
    """Read CSI data from ESP32 serial port."""

    def __init__(self, port: str, baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self.serial: Optional[serial.Serial] = None
        self.running = False
        self.csi_buffer = deque(maxlen=200)
        self.rssi_buffer = deque(maxlen=200)
        self.packet_count = 0
        self.last_rssi = -100
        self.thread: Optional[threading.Thread] = None

    def start(self):
        """Start reading from serial port."""
        try:
            self.serial = serial.Serial(self.port, self.baudrate, timeout=0.1)
            self.serial.reset_input_buffer()
            self.running = True
            self.thread = threading.Thread(target=self._read_loop, daemon=True)
            self.thread.start()
            return True
        except Exception as e:
            print(f"Error opening {self.port}: {e}")
            return False

    def stop(self):
        """Stop reading."""
        self.running = False
        if self.serial:
            self.serial.close()

    def _parse_csi_line(self, line: str) -> Optional[tuple]:
        """Parse CSI_DATA line from ESP32."""
        if not line.startswith("CSI_DATA,"):
            return None

        try:
            # Find the JSON array at the end
            array_start = line.find('"[')
            if array_start == -1:
                return None

            # Extract JSON array
            data_str = line[array_start:].strip().strip('"')
            data = json.loads(data_str)

            # Parse header fields
            parts = line[:array_start].split(",")
            if len(parts) < 4:
                return None

            rssi = int(parts[3])

            # Convert I/Q pairs to amplitude
            csi_array = np.array(data, dtype=np.float32)
            if len(csi_array) >= 2 and len(csi_array) % 2 == 0:
                csi_complex = csi_array[0::2] + 1j * csi_array[1::2]
                amplitude = np.abs(csi_complex)
            else:
                amplitude = csi_array

            return amplitude, rssi

        except (ValueError, json.JSONDecodeError):
            return None

    def _read_loop(self):
        """Background thread to read serial data."""
        while self.running:
            try:
                if self.serial and self.serial.in_waiting:
                    line = self.serial.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        result = self._parse_csi_line(line)
                        if result:
                            amplitude, rssi = result
                            self.csi_buffer.append(amplitude)
                            self.rssi_buffer.append(rssi)
                            self.last_rssi = rssi
                            self.packet_count += 1
                else:
                    time.sleep(0.001)
            except Exception:
                time.sleep(0.01)

    def get_csi_window(self, window_size: int, n_subcarriers: int) -> Optional[np.ndarray]:
        """Get a window of CSI data for inference."""
        if len(self.csi_buffer) < window_size // 2:
            return None

        samples = list(self.csi_buffer)[-window_size:]

        # Resize each sample to match model's expected n_subcarriers
        resized = np.zeros((len(samples), n_subcarriers), dtype=np.float32)
        for i, s in enumerate(samples):
            if len(s) >= n_subcarriers:
                resized[i] = s[:n_subcarriers]
            else:
                resized[i, :len(s)] = s

        # Pad if we don't have enough samples
        if len(samples) < window_size:
            padded = np.zeros((window_size, n_subcarriers), dtype=np.float32)
            padded[-len(samples):] = resized
            resized = padded

        return resized


def load_model(model_path: str):
    """Load the crowd estimation model."""
    device = torch.device("cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        config = checkpoint.get("config", {})
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
        config = {"mode": "classification", "n_subcarriers": 52}

    model = CrowdEstimator(CrowdEstimatorConfig(**config))
    model.load_state_dict(state_dict)
    model.eval()

    return model, config


def generate_synthetic_csi(num_people: int, window_size: int = 100, n_subcarriers: int = 52):
    """Generate synthetic CSI data for simulation mode."""
    base = np.random.uniform(40, 60, (window_size, n_subcarriers))

    for _ in range(num_people):
        freq = np.random.uniform(0.1, 0.5)
        phase = np.random.uniform(0, 2 * np.pi)
        amplitude = np.random.uniform(2, 5)
        t = np.arange(window_size) / 10.0
        movement = amplitude * np.sin(2 * np.pi * freq * t + phase)
        affected = np.random.choice(n_subcarriers, size=10, replace=False)
        for sc in affected:
            base[:, sc] += movement

    noise = np.random.normal(0, 1, base.shape)
    return (base + noise).astype(np.float32)


def clear_screen():
    """Clear terminal screen."""
    print("\033[2J\033[H", end="")


def move_cursor(line: int):
    """Move cursor to specific line."""
    print(f"\033[{line};0H", end="")


def run_monitor(model, config, readers: List[ESP32Reader], simulate: bool = False):
    """Run the real-time monitor."""
    n_subcarriers = config.get("n_subcarriers", 52)
    window_size = config.get("seq_length", 100)

    clear_screen()
    print("=" * 70)
    print("  WAVIRA - REAL-TIME CROWD LEVEL MONITOR")
    print("  Wi-Fi CSI Based Crowd Estimation System")
    print("=" * 70)

    if simulate:
        print("\n  Mode: SIMULATION (no real data)")
    else:
        ports = [r.port for r in readers]
        print(f"\n  Mode: LIVE ESP32 DATA")
        print(f"  Devices: {', '.join(ports)}")

    print("  Press Ctrl+C to stop\n")

    start_time = time.time()
    iteration = 0

    try:
        while True:
            elapsed = time.time() - start_time

            # Get CSI data
            csi_data = None
            rssi = -100
            total_packets = 0
            buffer_size = 0

            if simulate:
                # Simulation mode - use synthetic data
                csi_data = generate_synthetic_csi(0, window_size, n_subcarriers)
                rssi = -45 + np.random.randint(-10, 10)
            else:
                # Real mode - get data from ESP32
                for reader in readers:
                    data = reader.get_csi_window(window_size, n_subcarriers)
                    if data is not None:
                        if csi_data is None:
                            csi_data = data
                        else:
                            # Average multiple devices
                            csi_data = (csi_data + data) / 2
                        rssi = max(rssi, reader.last_rssi)
                    total_packets += reader.packet_count
                    buffer_size += len(reader.csi_buffer)

            # Update display
            move_cursor(10)

            print(f"  Time: {elapsed:6.1f}s | Iteration: {iteration:5d}                    ")
            print("-" * 70)

            if csi_data is None:
                print("\n  ⏳ Waiting for CSI data...                                      ")
                print("     (Collecting samples from ESP32)                              ")
                print(f"\n  {'─' * 66}")
                print(f"  Packets received: {total_packets}                              ")
                print(f"  Buffer size: {buffer_size}                                     ")
                iteration += 1
                time.sleep(0.5)
                continue

            # Normalize to match training distribution
            # Training data: mean ~50, std ~6
            # Real data: mean ~10, std ~8
            # First normalize to zero mean, unit variance, then scale to training distribution
            csi_data = (csi_data - csi_data.mean()) / (csi_data.std() + 1e-8)
            # Scale to training distribution (mean=50, std=6)
            csi_data = csi_data * 6.0 + 50.0
            # Re-normalize as model expects
            csi_data = (csi_data - csi_data.mean()) / (csi_data.std() + 1e-8)

            # Run inference
            x = torch.from_numpy(csi_data).unsqueeze(0).float()

            with torch.no_grad():
                output = model(x)
                probs = torch.softmax(output, dim=1)
                pred_class = probs.argmax(dim=1).item()
                confidence = probs[0, pred_class].item()

            label = CROWD_LABELS[pred_class]
            icon = CROWD_ICONS[pred_class]
            bar_len = int(confidence * 40)
            bar = "█" * bar_len + "░" * (40 - bar_len)

            print(f"\n  {icon} CROWD LEVEL: {label:25s}                ")
            print(f"     Confidence: [{bar}] {confidence*100:5.1f}%")

            print(f"\n  {'─' * 66}")
            print("  Class Probabilities:")
            for i in range(4):
                p = probs[0, i].item() * 100
                mini_bar = "▓" * int(p / 2.5) + "░" * (40 - int(p / 2.5))
                marker = " ◀" if i == pred_class else "  "
                print(f"    {CROWD_LABELS[i]:25s} [{mini_bar}] {p:5.1f}%{marker}")

            print(f"\n  {'─' * 66}")
            if simulate:
                print(f"  Source: Synthetic data (simulation mode)                        ")
            else:
                print(f"  Source: ESP32 CSI (packets: {total_packets}, buffer: {buffer_size})    ")
            print(f"  Signal: RSSI={rssi}dBm                                           ")

            iteration += 1
            time.sleep(0.2)

    except KeyboardInterrupt:
        print(f"\n\n  Monitor stopped after {iteration} iterations.")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Real-time crowd level monitor using ESP32 CSI data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use single ESP32
  python scripts/realtime_crowd_monitor.py --port /dev/cu.usbmodem21101

  # Use multiple ESP32 devices
  python scripts/realtime_crowd_monitor.py -p /dev/cu.usbmodem21101 -p /dev/cu.usbmodem21201

  # Simulation mode (no hardware)
  python scripts/realtime_crowd_monitor.py --simulate
        """
    )
    parser.add_argument(
        "--model",
        default="checkpoints/crowd/classification_transformer_20260115_141816/best_model.pt",
        help="Path to trained model",
    )
    parser.add_argument(
        "-p", "--port",
        action="append",
        dest="ports",
        help="ESP32 serial port (can specify multiple times)",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run in simulation mode without hardware",
    )
    parser.add_argument(
        "--baudrate",
        type=int,
        default=115200,
        help="Serial baudrate",
    )

    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        sys.exit(1)

    # Load model
    print("Loading model...")
    model, config = load_model(args.model)
    print(f"Model loaded (n_subcarriers={config.get('n_subcarriers', 52)})")

    readers = []

    if args.simulate:
        print("Running in simulation mode")
    elif args.ports:
        # Start ESP32 readers
        for port in args.ports:
            print(f"Connecting to {port}...")
            reader = ESP32Reader(port, args.baudrate)
            if reader.start():
                readers.append(reader)
                print(f"  Connected to {port}")
            else:
                print(f"  Failed to connect to {port}")

        if not readers:
            print("Error: No ESP32 devices connected")
            sys.exit(1)

        # Wait for initial data
        print("Waiting for CSI data...")
        time.sleep(2)
    else:
        # Auto-detect ESP32 devices
        import glob
        ports = glob.glob("/dev/cu.usbmodem*")
        if ports:
            print(f"Auto-detected ESP32 devices: {ports}")
            for port in ports:
                reader = ESP32Reader(port, args.baudrate)
                if reader.start():
                    readers.append(reader)
                    print(f"  Connected to {port}")

            if readers:
                print("Waiting for CSI data...")
                time.sleep(2)
            else:
                print("No ESP32 devices responding, running in simulation mode")
                args.simulate = True
        else:
            print("No ESP32 devices found, running in simulation mode")
            args.simulate = True

    try:
        run_monitor(model, config, readers, simulate=args.simulate)
    finally:
        for reader in readers:
            reader.stop()


if __name__ == "__main__":
    main()

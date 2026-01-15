#!/usr/bin/env python3
"""
Gesture Data Collection Script

Collect CSI data from ESP32 devices for gesture recognition training.

Usage:
    # Single ESP32 collection
    python scripts/collect_gesture_data.py --port /dev/cu.usbserial-0001 \
        --output_dir data/gestures --participant user1

    # Dual ESP32 collection
    python scripts/collect_gesture_data.py --port /dev/cu.usbserial-0001 \
        --port2 /dev/cu.usbserial-0002 --output_dir data/dual_gesture
"""

import argparse
import logging
import sys
import time
import threading
import os
from pathlib import Path
from datetime import datetime
from collections import deque
from typing import Optional, List

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Default gestures from the paper
GESTURE_LABELS = [
    "zoom_out",      # 1) Fingers spread apart
    "zoom_in",       # 2) Fingers come together
    "circle_left",   # 3) Finger circles counterclockwise
    "circle_right",  # 4) Finger circles clockwise
    "swipe_left",    # 5) Hand swipes left
    "swipe_right",   # 6) Hand swipes right
    "flip_up",       # 7) Hand flips upward
    "flip_down",     # 8) Hand flips downward
]


class CSICollector:
    """
    Collect CSI data from ESP32 via serial port.
    """

    def __init__(
        self,
        port: str,
        baud_rate: int = 115200,
        n_subcarriers: int = 114,
    ):
        self.port = port
        self.baud_rate = baud_rate
        self.n_subcarriers = n_subcarriers

        self.serial = None
        self.running = False
        self.thread = None

        self.frames: List[np.ndarray] = []
        self.collecting = False
        self.lock = threading.Lock()

    def start(self):
        """Start serial reading."""
        import serial

        self.serial = serial.Serial(self.port, self.baud_rate, timeout=1)
        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        logger.info(f"CSI collector started on {self.port}")

    def stop(self):
        """Stop serial reading."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.serial:
            self.serial.close()

    def start_collecting(self):
        """Start collecting CSI frames."""
        with self.lock:
            self.frames = []
            self.collecting = True

    def stop_collecting(self) -> np.ndarray:
        """
        Stop collecting and return collected frames.

        Returns:
            CSI data of shape (n_routes, n_subcarriers, n_frames)
        """
        with self.lock:
            self.collecting = False
            if len(self.frames) == 0:
                return None

            # Stack frames along time axis
            csi_data = np.stack(self.frames, axis=-1)
            self.frames = []
            return csi_data

    def _read_loop(self):
        """Background serial reading loop."""
        while self.running:
            try:
                line = self.serial.readline().decode('utf-8', errors='ignore').strip()
                if line.startswith('CSI_DATA'):
                    frame = self._parse_csi_line(line)
                    if frame is not None and self.collecting:
                        with self.lock:
                            self.frames.append(frame)
            except Exception as e:
                logger.debug(f"Serial read error: {e}")

    def _parse_csi_line(self, line: str) -> Optional[np.ndarray]:
        """Parse CSI data line from ESP32."""
        import json

        try:
            # Try JSON array format first (ESP32 CSI tool output)
            json_start = line.find('[')
            json_end = line.rfind(']') + 1

            if json_start > 0 and json_end > json_start:
                json_str = line[json_start:json_end]
                csi_raw = json.loads(json_str)

                # CSI data is I/Q pairs, compute amplitude
                n_pairs = len(csi_raw) // 2
                amplitudes = []
                for i in range(n_pairs):
                    I = csi_raw[2*i]
                    Q = csi_raw[2*i + 1]
                    amp = np.sqrt(I**2 + Q**2)
                    amplitudes.append(amp)

                n_routes = 1
                if len(amplitudes) >= self.n_subcarriers:
                    frame = np.array(amplitudes[:self.n_subcarriers])
                    frame = frame.reshape(n_routes, self.n_subcarriers)
                    return frame

            else:
                # Try CSV format
                parts = line.split(',')
                if len(parts) < 6:
                    return None

                csi_values = [float(x) for x in parts[5:] if x.strip()]
                n_routes = 1

                if len(csi_values) >= self.n_subcarriers:
                    frame = np.array(csi_values[:self.n_subcarriers])
                    frame = frame.reshape(n_routes, self.n_subcarriers)
                    return frame

            return None
        except Exception:
            return None


def print_gesture_menu():
    """Print gesture selection menu."""
    print("\n" + "=" * 50)
    print("GESTURE SELECTION")
    print("=" * 50)
    for i, label in enumerate(GESTURE_LABELS, 1):
        print(f"  {i}. {label}")
    print("  q. Quit data collection")
    print("=" * 50)


def print_gesture_instructions(gesture: str):
    """Print instructions for performing a gesture."""
    instructions = {
        "zoom_out": "Spread your fingers apart (like zooming out on a touchscreen)",
        "zoom_in": "Bring your fingers together (like zooming in on a touchscreen)",
        "circle_left": "Draw a circle with your finger (counterclockwise)",
        "circle_right": "Draw a circle with your finger (clockwise)",
        "swipe_left": "Swipe your hand to the left",
        "swipe_right": "Swipe your hand to the right",
        "flip_up": "Flip your hand upward",
        "flip_down": "Flip your hand downward",
    }

    print(f"\n>>> GESTURE: {gesture.upper()}")
    print(f"    {instructions.get(gesture, 'Perform the gesture')}")
    print()


def collect_gesture_samples(
    collector: CSICollector,
    collector2: Optional[CSICollector],
    gesture: str,
    output_dir: Path,
    participant: str,
    samples_per_gesture: int = 50,
    dual_mode: bool = False,
):
    """
    Collect multiple samples for a single gesture.

    Args:
        collector: Primary CSI collector
        collector2: Secondary collector for dual mode
        gesture: Gesture label
        output_dir: Output directory
        participant: Participant identifier
        samples_per_gesture: Number of samples to collect
        dual_mode: Whether using dual ESP32 setup
    """
    gesture_dir = output_dir / gesture
    gesture_dir.mkdir(parents=True, exist_ok=True)

    # Find existing sample count
    existing = list(gesture_dir.glob(f"{participant}_*.npy"))
    start_idx = len(existing) // (2 if dual_mode else 1)

    print_gesture_instructions(gesture)

    sample_idx = start_idx
    while sample_idx < start_idx + samples_per_gesture:
        # Prompt for gesture
        input(f"  Sample {sample_idx + 1}/{start_idx + samples_per_gesture}: "
              f"Press ENTER when ready, then perform the gesture...")

        print("    Recording... ", end="", flush=True)

        # Start collecting
        collector.start_collecting()
        if collector2:
            collector2.start_collecting()

        # Wait for gesture (typical gesture is 1-2 seconds)
        time.sleep(2.0)

        # Stop collecting
        csi_data1 = collector.stop_collecting()
        csi_data2 = collector2.stop_collecting() if collector2 else None

        if csi_data1 is None or csi_data1.shape[-1] < 50:
            print("Failed! Not enough data collected. Try again.")
            continue

        # Save data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_base = f"{participant}_{sample_idx:03d}_{timestamp}"

        if dual_mode:
            np.save(gesture_dir / f"{filename_base}_device1.npy", csi_data1)
            if csi_data2 is not None:
                np.save(gesture_dir / f"{filename_base}_device2.npy", csi_data2)
            print(f"Saved! ({csi_data1.shape[-1]} frames, device1+device2)")
        else:
            np.save(gesture_dir / f"{filename_base}.npy", csi_data1)
            print(f"Saved! ({csi_data1.shape[-1]} frames)")

        sample_idx += 1

    print(f"\n  Completed {samples_per_gesture} samples for {gesture}!")


def collect_all_gestures(
    collector: CSICollector,
    collector2: Optional[CSICollector],
    output_dir: Path,
    participant: str,
    samples_per_gesture: int = 50,
):
    """Collect samples for all gestures."""
    dual_mode = collector2 is not None

    for gesture in GESTURE_LABELS:
        print(f"\n{'='*50}")
        print(f"COLLECTING: {gesture.upper()}")
        print(f"{'='*50}")

        collect_gesture_samples(
            collector=collector,
            collector2=collector2,
            gesture=gesture,
            output_dir=output_dir,
            participant=participant,
            samples_per_gesture=samples_per_gesture,
            dual_mode=dual_mode,
        )

        proceed = input("\nProceed to next gesture? (y/n): ")
        if proceed.lower() != 'y':
            break

    print("\nData collection complete!")


def interactive_collection(
    collector: CSICollector,
    collector2: Optional[CSICollector],
    output_dir: Path,
    participant: str,
    samples_per_gesture: int = 50,
):
    """Interactive gesture data collection."""
    dual_mode = collector2 is not None

    while True:
        print_gesture_menu()
        choice = input("Select gesture (1-8) or 'a' for all, 'q' to quit: ").strip()

        if choice.lower() == 'q':
            break
        elif choice.lower() == 'a':
            collect_all_gestures(
                collector, collector2, output_dir, participant, samples_per_gesture
            )
        elif choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(GESTURE_LABELS):
                gesture = GESTURE_LABELS[idx]
                collect_gesture_samples(
                    collector=collector,
                    collector2=collector2,
                    gesture=gesture,
                    output_dir=output_dir,
                    participant=participant,
                    samples_per_gesture=samples_per_gesture,
                    dual_mode=dual_mode,
                )
            else:
                print("Invalid selection!")
        else:
            print("Invalid input!")


def generate_synthetic_dataset(output_dir: Path, samples_per_gesture: int = 200):
    """Generate synthetic gesture dataset for testing."""
    from wavira.data.gesture_dataset import SyntheticGestureDataset

    logger.info(f"Generating synthetic dataset with {samples_per_gesture} samples per gesture...")

    # Generate dataset
    dataset = SyntheticGestureDataset(
        n_samples_per_gesture=samples_per_gesture,
        n_frames=100,  # Longer sequences for window extraction
        n_routes=3,
        n_subcarriers=114,
        gesture_labels=GESTURE_LABELS,
    )

    # Save samples to directory structure
    for i in range(len(dataset)):
        csi, label_idx = dataset[i]
        csi = csi.numpy()

        gesture = GESTURE_LABELS[label_idx]
        gesture_dir = output_dir / gesture
        gesture_dir.mkdir(parents=True, exist_ok=True)

        sample_idx = i % samples_per_gesture
        np.save(gesture_dir / f"synthetic_{sample_idx:03d}.npy", csi)

    logger.info(f"Synthetic dataset saved to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Collect gesture CSI data")

    parser.add_argument(
        "--port",
        type=str,
        default=None,
        help="Serial port for ESP32 device 1"
    )
    parser.add_argument(
        "--port2",
        type=str,
        default=None,
        help="Serial port for ESP32 device 2 (dual mode)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/gestures",
        help="Output directory for collected data"
    )
    parser.add_argument(
        "--participant",
        type=str,
        default="user1",
        help="Participant identifier"
    )
    parser.add_argument(
        "--samples_per_gesture",
        type=int,
        default=50,
        help="Number of samples to collect per gesture"
    )
    parser.add_argument(
        "--baud_rate",
        type=int,
        default=115200,
        help="Serial baud rate"
    )
    parser.add_argument(
        "--n_subcarriers",
        type=int,
        default=114,
        help="Number of CSI subcarriers"
    )
    parser.add_argument(
        "--generate_synthetic",
        action="store_true",
        help="Generate synthetic dataset instead of collecting"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.generate_synthetic:
        generate_synthetic_dataset(output_dir, args.samples_per_gesture)
        return

    if args.port is None:
        logger.error("Please specify --port or use --generate_synthetic")
        sys.exit(1)

    # Create collectors
    collector = CSICollector(
        port=args.port,
        baud_rate=args.baud_rate,
        n_subcarriers=args.n_subcarriers,
    )

    collector2 = None
    if args.port2:
        collector2 = CSICollector(
            port=args.port2,
            baud_rate=args.baud_rate,
            n_subcarriers=args.n_subcarriers,
        )

    try:
        # Start collectors
        collector.start()
        if collector2:
            collector2.start()

        # Wait for serial to initialize
        time.sleep(1)

        # Run interactive collection
        interactive_collection(
            collector=collector,
            collector2=collector2,
            output_dir=output_dir,
            participant=args.participant,
            samples_per_gesture=args.samples_per_gesture,
        )

    finally:
        collector.stop()
        if collector2:
            collector2.stop()


if __name__ == "__main__":
    main()

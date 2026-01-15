#!/usr/bin/env python3
"""
Real-time Gesture Recognition Inference

Run gesture recognition on live CSI data from ESP32 devices.

Usage:
    # Single ESP32 inference
    python scripts/gesture_inference.py --model best_model.pt --port /dev/cu.usbserial-0001

    # Dual ESP32 inference
    python scripts/gesture_inference.py --model best_model.pt \
        --port /dev/cu.usbserial-0001 --port2 /dev/cu.usbserial-0002

    # Test with synthetic data
    python scripts/gesture_inference.py --model best_model.pt --test_synthetic
"""

import argparse
import logging
import sys
import time
import threading
import queue
from pathlib import Path
from collections import deque
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass

import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from wavira.models.gesture_recognizer import (
    GestureRecognizer3DCNN,
    GestureRecognizerLite,
    DualESP32GestureRecognizer,
    DEFAULT_GESTURE_LABELS,
    create_gesture_model,
)
from wavira.data.gesture_preprocessing import (
    GesturePreprocessor,
    segment_gesture,
)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for real-time inference."""
    n_frames: int = 32  # Frames needed for one prediction
    frame_stride: int = 8  # Stride for sliding window
    confidence_threshold: float = 0.6  # Minimum confidence for prediction
    cooldown_frames: int = 16  # Frames to wait after prediction
    n_subcarriers: int = 114  # ESP32 subcarrier count
    n_routes: int = 3  # TX*RX routes
    sampling_rate: float = 100.0  # CSI sampling rate (Hz)


class CSIBuffer:
    """
    Circular buffer for storing incoming CSI frames.

    Maintains a sliding window of CSI data for continuous inference.
    """

    def __init__(self, max_frames: int = 200, n_routes: int = 3, n_subcarriers: int = 114):
        self.max_frames = max_frames
        self.n_routes = n_routes
        self.n_subcarriers = n_subcarriers

        # Initialize buffer
        self.buffer = np.zeros((n_routes, n_subcarriers, max_frames), dtype=np.float32)
        self.write_idx = 0
        self.frame_count = 0
        self.lock = threading.Lock()

    def add_frame(self, frame: np.ndarray):
        """
        Add a new CSI frame to the buffer.

        Args:
            frame: CSI frame of shape (n_routes, n_subcarriers)
        """
        with self.lock:
            self.buffer[:, :, self.write_idx] = frame
            self.write_idx = (self.write_idx + 1) % self.max_frames
            self.frame_count += 1

    def get_window(self, n_frames: int) -> Optional[np.ndarray]:
        """
        Get the most recent n_frames from the buffer.

        Args:
            n_frames: Number of frames to retrieve

        Returns:
            CSI window of shape (n_routes, n_subcarriers, n_frames) or None
        """
        if self.frame_count < n_frames:
            return None

        with self.lock:
            end_idx = self.write_idx
            start_idx = (end_idx - n_frames) % self.max_frames

            if start_idx < end_idx:
                return self.buffer[:, :, start_idx:end_idx].copy()
            else:
                # Wrap around
                part1 = self.buffer[:, :, start_idx:]
                part2 = self.buffer[:, :, :end_idx]
                return np.concatenate([part1, part2], axis=-1)

    def clear(self):
        """Clear the buffer."""
        with self.lock:
            self.buffer.fill(0)
            self.write_idx = 0
            self.frame_count = 0


class GestureInferenceEngine:
    """
    Real-time gesture recognition inference engine.

    Manages model loading, CSI buffering, and continuous prediction.
    """

    def __init__(
        self,
        model_path: str,
        config: Optional[InferenceConfig] = None,
        device: Optional[str] = None,
    ):
        self.config = config or InferenceConfig()

        # Setup device
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        logger.info(f"Using device: {self.device}")

        # Load model
        self.model, self.model_config, self.gesture_labels = self._load_model(model_path)
        self.model.eval()

        # Setup preprocessor
        self.preprocessor = GesturePreprocessor(
            butter_cutoff=20.0,
            sampling_rate=self.config.sampling_rate,
        )

        # CSI buffers
        self.buffer1 = CSIBuffer(
            max_frames=200,
            n_routes=self.config.n_routes,
            n_subcarriers=self.config.n_subcarriers,
        )
        self.buffer2 = None  # For dual-device mode

        # State tracking
        self.last_prediction_frame = 0
        self.prediction_history = deque(maxlen=10)

        self.is_dual = self.model_config.get('model_type', 'standard') == 'dual'
        if self.is_dual:
            self.buffer2 = CSIBuffer(
                max_frames=200,
                n_routes=self.config.n_routes,
                n_subcarriers=self.config.n_subcarriers,
            )

    def _load_model(self, model_path: str) -> Tuple[torch.nn.Module, Dict, List[str]]:
        """Load trained model from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device)

        model_config = checkpoint.get('config', {})
        gesture_labels = checkpoint.get('gesture_labels', DEFAULT_GESTURE_LABELS)

        # Create model
        model = create_gesture_model(
            model_type=model_config.get('model_type', 'standard'),
            n_gestures=model_config.get('n_gestures', 8),
            n_subcarriers=model_config.get('n_subcarriers', 114),
            n_routes=model_config.get('n_routes', 3),
            n_frames=model_config.get('n_frames', 32),
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)

        logger.info(f"Loaded model from {model_path}")
        logger.info(f"Model config: {model_config}")
        logger.info(f"Gesture labels: {gesture_labels}")

        return model, model_config, gesture_labels

    def add_csi_frame(self, frame: np.ndarray, device_id: int = 1):
        """
        Add a new CSI frame from ESP32.

        Args:
            frame: CSI frame of shape (n_routes, n_subcarriers)
            device_id: Device ID (1 or 2 for dual mode)
        """
        if device_id == 1:
            self.buffer1.add_frame(frame)
        elif device_id == 2 and self.buffer2 is not None:
            self.buffer2.add_frame(frame)

    def predict(self) -> Optional[Tuple[str, float, int]]:
        """
        Make a gesture prediction from current buffer state.

        Returns:
            Tuple of (gesture_label, confidence, gesture_idx) or None
        """
        # Check cooldown
        frames_since_last = self.buffer1.frame_count - self.last_prediction_frame
        if frames_since_last < self.config.cooldown_frames:
            return None

        # Get CSI windows
        window1 = self.buffer1.get_window(self.config.n_frames)
        if window1 is None:
            return None

        if self.is_dual:
            window2 = self.buffer2.get_window(self.config.n_frames)
            if window2 is None:
                return None

        # Preprocess
        window1 = self.preprocessor(window1)

        # Convert to tensor
        tensor1 = torch.from_numpy(window1).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            if self.is_dual:
                window2 = self.preprocessor(window2)
                tensor2 = torch.from_numpy(window2).unsqueeze(0).to(self.device)
                logits = self.model(tensor1, tensor2)
            else:
                logits = self.model(tensor1)

            probs = torch.softmax(logits, dim=-1)
            confidence, pred_idx = probs.max(dim=-1)

            confidence = confidence.item()
            pred_idx = pred_idx.item()

        # Check confidence threshold
        if confidence < self.config.confidence_threshold:
            return None

        # Update state
        self.last_prediction_frame = self.buffer1.frame_count
        gesture_label = self.gesture_labels[pred_idx]

        # Add to history
        self.prediction_history.append((gesture_label, confidence, time.time()))

        return gesture_label, confidence, pred_idx

    def get_prediction_history(self) -> List[Tuple[str, float, float]]:
        """Get recent prediction history."""
        return list(self.prediction_history)


class ESP32CSIReader:
    """
    Read CSI data from ESP32 via serial port.
    """

    def __init__(
        self,
        port: str,
        baud_rate: int = 115200,
        callback=None,
        n_subcarriers: int = 114,
    ):
        self.port = port
        self.baud_rate = baud_rate
        self.callback = callback
        self.n_subcarriers = n_subcarriers

        self.serial = None
        self.running = False
        self.thread = None

    def start(self):
        """Start reading CSI data in background thread."""
        import serial

        self.serial = serial.Serial(self.port, self.baud_rate, timeout=1)
        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        logger.info(f"Started CSI reader on {self.port}")

    def stop(self):
        """Stop reading."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.serial:
            self.serial.close()

    def _read_loop(self):
        """Background thread for reading serial data."""
        while self.running:
            try:
                line = self.serial.readline().decode('utf-8', errors='ignore').strip()
                if line.startswith('CSI_DATA'):
                    frame = self._parse_csi_line(line)
                    if frame is not None and self.callback:
                        self.callback(frame)
            except Exception as e:
                logger.warning(f"Error reading serial: {e}")

    def _parse_csi_line(self, line: str) -> Optional[np.ndarray]:
        """
        Parse CSI data line from ESP32.

        Handles multiple formats:
        1. JSON array format: CSI_DATA,...,"[I,Q,I,Q,...]"
        2. CSV format: CSI_DATA,<mac>,<rssi>,<rate>,<noise>,<data...>
        """
        import json

        try:
            # Try JSON array format first (ESP32 CSI tool output)
            json_start = line.find('[')
            json_end = line.rfind(']') + 1

            if json_start > 0 and json_end > json_start:
                # Extract JSON array
                json_str = line[json_start:json_end]
                csi_raw = json.loads(json_str)

                # CSI data is I/Q pairs, compute amplitude
                # Format: [I0, Q0, I1, Q1, ...]
                n_pairs = len(csi_raw) // 2
                amplitudes = []
                for i in range(n_pairs):
                    I = csi_raw[2*i]
                    Q = csi_raw[2*i + 1]
                    amp = np.sqrt(I**2 + Q**2)
                    amplitudes.append(amp)

                # ESP32 typically has 1 RX antenna for CSI
                # Reshape to (n_routes, n_subcarriers)
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

        except Exception as e:
            logger.debug(f"Failed to parse CSI line: {e}")
            return None


def run_realtime_inference(args):
    """Run real-time gesture recognition from ESP32."""
    config = InferenceConfig(
        n_frames=args.n_frames,
        confidence_threshold=args.confidence_threshold,
        n_subcarriers=args.n_subcarriers,
        n_routes=args.n_routes,
    )

    engine = GestureInferenceEngine(
        model_path=args.model,
        config=config,
        device=args.device,
    )

    # Callback for CSI frames
    def on_csi_frame(frame, device_id=1):
        engine.add_csi_frame(frame, device_id)

    # Start serial readers
    readers = []

    if args.port:
        reader1 = ESP32CSIReader(
            args.port,
            callback=lambda f: on_csi_frame(f, 1),
            n_subcarriers=args.n_subcarriers,
        )
        reader1.start()
        readers.append(reader1)

    if args.port2 and engine.is_dual:
        reader2 = ESP32CSIReader(
            args.port2,
            callback=lambda f: on_csi_frame(f, 2),
            n_subcarriers=args.n_subcarriers,
        )
        reader2.start()
        readers.append(reader2)

    logger.info("Starting real-time gesture recognition...")
    logger.info("Press Ctrl+C to stop")

    try:
        while True:
            result = engine.predict()
            if result:
                gesture, confidence, idx = result
                logger.info(f"Detected: {gesture} (confidence: {confidence:.3f})")

            time.sleep(0.05)  # 20 Hz prediction rate

    except KeyboardInterrupt:
        logger.info("Stopping...")

    finally:
        for reader in readers:
            reader.stop()


def run_synthetic_test(args):
    """Test inference with synthetic data."""
    from wavira.data.gesture_dataset import SyntheticGestureDataset

    config = InferenceConfig(
        n_frames=args.n_frames,
        confidence_threshold=0.3,  # Lower threshold for synthetic test
        n_subcarriers=args.n_subcarriers,
        n_routes=args.n_routes,
    )

    engine = GestureInferenceEngine(
        model_path=args.model,
        config=config,
        device=args.device,
    )

    # Generate synthetic data
    dataset = SyntheticGestureDataset(
        n_samples_per_gesture=10,
        n_frames=args.n_frames,
        n_routes=args.n_routes,
        n_subcarriers=args.n_subcarriers,
    )

    logger.info("Testing with synthetic data...")

    correct = 0
    total = 0

    for i in range(len(dataset)):
        csi, true_label = dataset[i]
        csi = csi.numpy()

        # Add frames to buffer
        engine.buffer1.clear()
        for t in range(args.n_frames):
            engine.add_csi_frame(csi[:, :, t])

        # Force prediction (bypass cooldown for testing)
        engine.last_prediction_frame = 0

        result = engine.predict()
        if result:
            gesture, confidence, pred_idx = result
            total += 1
            if pred_idx == true_label:
                correct += 1

            true_gesture = engine.gesture_labels[true_label]
            status = "OK" if pred_idx == true_label else "WRONG"
            logger.info(
                f"Sample {i}: True={true_gesture}, Pred={gesture}, "
                f"Conf={confidence:.3f} [{status}]"
            )

    if total > 0:
        accuracy = correct / total
        logger.info(f"\nTest accuracy: {accuracy:.4f} ({correct}/{total})")


def parse_args():
    parser = argparse.ArgumentParser(description="Gesture recognition inference")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
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
        "--test_synthetic",
        action="store_true",
        help="Test with synthetic data"
    )
    parser.add_argument(
        "--n_frames",
        type=int,
        default=32,
        help="Number of frames per prediction"
    )
    parser.add_argument(
        "--n_subcarriers",
        type=int,
        default=114,
        help="Number of CSI subcarriers"
    )
    parser.add_argument(
        "--n_routes",
        type=int,
        default=3,
        help="Number of TX*RX routes"
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.6,
        help="Minimum confidence for prediction"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for inference (cuda/mps/cpu)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.test_synthetic:
        run_synthetic_test(args)
    elif args.port:
        run_realtime_inference(args)
    else:
        logger.error("Please specify --port or --test_synthetic")
        sys.exit(1)


if __name__ == "__main__":
    main()

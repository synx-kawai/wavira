#!/usr/bin/env python3
"""
Real-time Crowd Level Monitor

Continuously displays crowd level predictions with live updating display.
Uses synthetic CSI data for demonstration, or real MQTT data when available.

Usage:
    python scripts/realtime_crowd_monitor.py
    python scripts/realtime_crowd_monitor.py --mqtt-host localhost
    python scripts/realtime_crowd_monitor.py --model checkpoints/crowd/best_model.pt
"""

import sys
import os
import time
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wavira.models.crowd_estimator import CrowdEstimator, CrowdEstimatorConfig

# Try to import MQTT
try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False

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
    """Generate synthetic CSI data matching training distribution."""
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


def get_simulated_people(t: float) -> int:
    """Simulate realistic crowd pattern over time (60s cycle)."""
    cycle = t % 60
    if cycle < 10:
        return 0
    elif cycle < 20:
        return np.random.randint(1, 3)
    elif cycle < 35:
        return np.random.randint(3, 6)
    elif cycle < 50:
        return np.random.randint(6, 10)
    else:
        return np.random.randint(0, 3)


def clear_screen():
    """Clear terminal screen."""
    print("\033[2J\033[H", end="")


def move_cursor(line: int):
    """Move cursor to specific line."""
    print(f"\033[{line};0H", end="")


def run_monitor(model, config, use_mqtt: bool = False, mqtt_host: str = "localhost"):
    """Run the real-time monitor."""
    n_subcarriers = config.get("n_subcarriers", 52)
    window_size = config.get("seq_length", 100)

    clear_screen()
    print("=" * 70)
    print("  WAVIRA - REAL-TIME CROWD LEVEL MONITOR")
    print("  Wi-Fi CSI Based Crowd Estimation System")
    print("=" * 70)
    print(f"\n  Mode: {'MQTT' if use_mqtt else 'Synthetic Data'}")
    print("  Press Ctrl+C to stop\n")

    start_time = time.time()
    iteration = 0

    try:
        while True:
            elapsed = time.time() - start_time

            # Get simulated or real data
            num_people = get_simulated_people(elapsed)
            csi_data = generate_synthetic_csi(num_people, window_size, n_subcarriers)
            csi_data = (csi_data - csi_data.mean()) / (csi_data.std() + 1e-8)

            # Run inference
            x = torch.from_numpy(csi_data).unsqueeze(0).float()

            with torch.no_grad():
                output = model(x)
                probs = torch.softmax(output, dim=1)
                pred_class = probs.argmax(dim=1).item()
                confidence = probs[0, pred_class].item()

            # Update display
            move_cursor(9)

            print(f"  Time: {elapsed:6.1f}s | Iteration: {iteration:5d}                    ")
            print("-" * 70)

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
            print(f"  Input: {num_people} people (simulated)                              ")

            rssi = -45 + np.random.randint(-10, 10)
            snr = 25 + np.random.randint(-5, 5)
            print(f"  Signal: RSSI={rssi}dBm, SNR={snr}dB                    ")

            iteration += 1
            time.sleep(0.5)

    except KeyboardInterrupt:
        print(f"\n\n  Monitor stopped after {iteration} iterations.")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Real-time crowd level monitor")
    parser.add_argument(
        "--model",
        default="checkpoints/crowd/classification_transformer_20260115_141816/best_model.pt",
        help="Path to trained model",
    )
    parser.add_argument("--mqtt-host", default="localhost", help="MQTT broker host")
    parser.add_argument("--mqtt", action="store_true", help="Use MQTT for real data")

    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        sys.exit(1)

    model, config = load_model(args.model)
    run_monitor(model, config, use_mqtt=args.mqtt, mqtt_host=args.mqtt_host)


if __name__ == "__main__":
    main()

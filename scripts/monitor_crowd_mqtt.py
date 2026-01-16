#!/usr/bin/env python3
"""
MQTT-based real-time crowd level monitoring.

Usage:
    python scripts/monitor_crowd_mqtt.py --model checkpoints/crowd/final_model.pt
    python scripts/monitor_crowd_mqtt.py --model checkpoints/crowd/final_model.pt --mqtt-host 13.115.255.111
"""

import sys
import os
import time
import json
import argparse
import logging
from collections import deque
from typing import Optional

import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import paho.mqtt.client as mqtt
except ImportError:
    print("Please install paho-mqtt: pip install paho-mqtt")
    sys.exit(1)

from wavira.models.crowd_estimator import CrowdEstimator, CrowdEstimatorConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


CROWD_LABELS = {
    0: "Empty (0 people)",
    1: "Low (1-2 people)",
    2: "Medium (3-5 people)",
    3: "High (6+ people)",
}


class MQTTCrowdMonitor:
    """Real-time crowd monitoring via MQTT."""

    def __init__(
        self,
        model_path: str,
        mqtt_host: str = "localhost",
        mqtt_port: int = 1883,
        device_id: Optional[str] = None,
        window_size: int = 50,
        mode: str = "classification",
    ):
        self.mqtt_host = mqtt_host
        self.mqtt_port = mqtt_port
        self.device_id = device_id
        self.window_size = window_size
        self.mode = mode

        # Load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # Handle both checkpoint formats (wrapped or direct state_dict)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            config = checkpoint.get("config", {})
            state_dict = checkpoint["model_state_dict"]
        else:
            # Direct state_dict, infer config from layer shapes
            state_dict = checkpoint
            # Check output layer shape to determine mode
            out_weight = state_dict.get("head.3.weight", None)
            in_weight = state_dict.get("temporal_conv.0.conv.weight", None)

            config = {}
            if out_weight is not None and out_weight.shape[0] == 4:
                config["mode"] = "classification"
            else:
                config["mode"] = "regression"

            # Infer n_subcarriers from input layer
            if in_weight is not None:
                config["n_subcarriers"] = in_weight.shape[1]

        self.n_subcarriers = config.get("n_subcarriers", 52)
        self.model = CrowdEstimator(CrowdEstimatorConfig(**config))
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Model loaded from {model_path} (n_subcarriers={self.n_subcarriers})")

        # CSI buffer
        self.csi_buffer = deque(maxlen=window_size)
        self.last_inference = 0
        self.inference_interval = 1.0

        # MQTT client
        self.client = mqtt.Client(client_id=f"wavira-monitor-{int(time.time())}")
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.running = False

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            topic = f"wavira/csi/{self.device_id}" if self.device_id else "wavira/csi/#"
            client.subscribe(topic)
            logger.info(f"Connected to MQTT, subscribed to {topic}")

    def _on_message(self, client, userdata, msg):
        try:
            data = json.loads(msg.payload.decode())
            csi_raw = data.get("data", [])

            if not csi_raw:
                return

            # Convert to amplitude
            csi_array = np.array(csi_raw, dtype=np.float32)
            if len(csi_array) % 2 == 0:
                csi_complex = csi_array[0::2] + 1j * csi_array[1::2]
                amplitude = np.abs(csi_complex)
            else:
                amplitude = csi_array

            self.csi_buffer.append(amplitude)

            # Run inference
            now = time.time()
            if now - self.last_inference > self.inference_interval:
                if len(self.csi_buffer) >= self.window_size // 2:
                    self._run_inference()
                self.last_inference = now

        except Exception as e:
            logger.warning(f"Error processing message: {e}")

    def _run_inference(self):
        if len(self.csi_buffer) < self.window_size // 2:
            return

        # Prepare input
        samples = list(self.csi_buffer)

        # Resize each sample to match model's expected n_subcarriers
        resized = np.zeros((len(samples), self.n_subcarriers), dtype=np.float32)
        for i, s in enumerate(samples):
            if len(s) >= self.n_subcarriers:
                resized[i] = s[:self.n_subcarriers]
            else:
                resized[i, :len(s)] = s

        # Convert to tensor
        x = torch.from_numpy(resized).unsqueeze(0).to(self.device)

        # Run inference
        with torch.no_grad():
            output = self.model(x)

            if self.mode == "classification":
                probs = torch.softmax(output, dim=1)
                pred_class = probs.argmax(dim=1).item()
                confidence = probs[0, pred_class].item()
                label = CROWD_LABELS.get(pred_class, f"Level {pred_class}")
                print(f"\rCrowd: {label:25s} (Conf: {confidence*100:5.1f}%)", end="", flush=True)
            else:
                count = output.item()
                print(f"\rEstimated people: {count:.1f}", end="", flush=True)

    def run(self):
        logger.info(f"Starting crowd monitor: {self.mqtt_host}:{self.mqtt_port}")
        self.running = True

        try:
            self.client.connect(self.mqtt_host, self.mqtt_port, 60)
            self.client.loop_forever()
        except KeyboardInterrupt:
            logger.info("Stopped by user")
        finally:
            self.client.disconnect()


def main():
    parser = argparse.ArgumentParser(description="MQTT-based real-time crowd monitoring")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--mqtt-host", default="localhost", help="MQTT broker host")
    parser.add_argument("--mqtt-port", type=int, default=1883, help="MQTT broker port")
    parser.add_argument("--device-id", default=None, help="Specific device ID")
    parser.add_argument("--window", type=int, default=50, help="Window size")
    parser.add_argument("--mode", choices=["classification", "regression"], default="classification")

    args = parser.parse_args()

    monitor = MQTTCrowdMonitor(
        model_path=args.model,
        mqtt_host=args.mqtt_host,
        mqtt_port=args.mqtt_port,
        device_id=args.device_id,
        window_size=args.window,
        mode=args.mode,
    )

    monitor.run()


if __name__ == "__main__":
    main()

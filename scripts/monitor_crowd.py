#!/usr/bin/env python3
"""
Real-time crowd level monitoring script.

Usage:
    python scripts/monitor_crowd.py --model checkpoints/crowd/final_model.pt
"""

import argparse
import os
import sys
import time
import torch
import numpy as np
import logging
from collections import deque
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from wavira.models.crowd_estimator import create_model
from wavira.utils.esp32_serial import ESP32Serial

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CROWD_LEVELS = {
    0: "Empty (0 people)",
    1: "Low (1-2 people)",
    2: "Medium (3-5 people)",
    3: "High (6+ people)"
}

def parse_args():
    parser = argparse.ArgumentParser(description="Monitor Crowd Level")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--port", type=str, default=None, help="Serial port")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate")
    parser.add_argument("--window", type=int, default=100, help="Sliding window size (packets)")
    parser.add_argument("--interval", type=float, default=1.0, help="Inference interval (seconds)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    parser.add_argument("--mode", type=str, default="classification", choices=["regression", "classification"], help="Model mode")
    parser.add_argument("--sim", action="store_true", help="Simulate data instead of connecting to ESP32")
    return parser.parse_args()

class CrowdMonitor:
    def __init__(self, args):
        self.args = args
        self.buffer = deque(maxlen=args.window)
        self.last_inference = 0
        self.device = torch.device(args.device)
        
        # Load model
        self.model = self._load_model(args.model)
        self.model.eval()
        
    def _load_model(self, path):
        if not os.path.exists(path):
            logger.error(f"Model file not found: {path}")
            sys.exit(1)
            
        checkpoint = torch.load(path, map_location=self.device)
        
        # Extract config if available, otherwise assume defaults
        config_dict = checkpoint.get('config', {})
        
        # Create model
        model = create_model(
            mode=config_dict.get('mode', self.args.mode),
            encoder_type=config_dict.get('encoder_type', 'transformer'),
            n_subcarriers=config_dict.get('n_subcarriers', 52),
            hidden_dim=config_dict.get('hidden_dim', 128),
            num_layers=config_dict.get('num_layers', 2)
        )
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model.to(self.device)
        logger.info(f"Model loaded from {path}")
        return model
        
    def _preprocess(self, buffer):
        # Buffer is list of (subcarriers,)
        data = np.array(buffer) # (seq, subcarriers)
        
        # Extract amplitude if complex
        if np.iscomplexobj(data):
            data = np.abs(data)
            
        # Standardize (per-sample normalization)
        mean = data.mean()
        std = data.std() + 1e-8
        data = (data - mean) / std
        
        # Add batch dim: (1, seq, subcarriers)
        # Note: CrowdEstimator expects (batch, seq, subcarriers)
        tensor = torch.from_numpy(data).float().unsqueeze(0)
        
        return tensor.to(self.device)

    def on_packet(self, packet):
        # packet.csi_data is typically (subcarriers,) or complex
        # We handle it in preprocess
        self.buffer.append(packet.csi_data)
        
        now = time.time()
        if len(self.buffer) == self.args.window and (now - self.last_inference > self.args.interval):
            self.infer()
            self.last_inference = now
            
    def infer(self):
        try:
            input_tensor = self._preprocess(list(self.buffer))
            
            with torch.no_grad():
                output = self.model(input_tensor)
                
                if self.args.mode == "regression":
                    count = output.item()
                    print(f"\rEstimated People: {count:.1f}      ", end="", flush=True)
                else:
                    probs = torch.softmax(output, dim=1)
                    pred_idx = output.argmax(dim=1).item()
                    confidence = probs[0][pred_idx].item()
                    level_str = CROWD_LEVELS.get(pred_idx, f"Unknown ({pred_idx})")
                    print(f"\rStatus: {level_str:<20} (Conf: {confidence:.1%})", end="", flush=True)
            
        except Exception as e:
            logger.error(f"Inference error: {e}")

    def simulate_data(self):
        """Generate fake data for testing."""
        logger.info("Starting SIMULATION mode (Ctrl+C to stop)...")
        import random
        n_subcarriers = 52
        
        try:
            while True:
                # Generate random CSI packet
                # Simulate a sine wave pattern that changes
                t = time.time()
                data = np.sin(np.linspace(0, 10, n_subcarriers) + t) + np.random.normal(0, 0.1, n_subcarriers)
                
                # Create a dummy packet object
                class DummyPacket:
                    pass
                packet = DummyPacket()
                packet.csi_data = data
                
                self.on_packet(packet)
                time.sleep(0.01) # 100Hz
        except KeyboardInterrupt:
            pass

    def start(self):
        if self.args.sim:
            self.simulate_data()
            return

        print(f"Connecting to ESP32 on {self.args.port or 'auto'}...")
        try:
            esp = ESP32Serial(port=self.args.port, baud_rate=self.args.baud)
            esp.add_csi_callback(self.on_packet)
            
            if esp.connect():
                print("Connected! Monitoring started (Press Ctrl+C to stop)...")
                esp.start()
                while True:
                    time.sleep(1)
            else:
                print("Failed to connect to ESP32.")
        except KeyboardInterrupt:
            print("\nStopping...")
            if 'esp' in locals():
                esp.stop()
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main_args = parse_args()
    monitor = CrowdMonitor(main_args)
    monitor.start()

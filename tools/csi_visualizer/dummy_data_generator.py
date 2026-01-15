#!/usr/bin/env python3
"""
Dummy CSI Data Generator for Performance Testing

Generates realistic CSI data and publishes to MQTT broker.
Used for testing dashboard performance without actual ESP32 devices.
"""

import argparse
import json
import math
import random
import time
from typing import Any

import paho.mqtt.client as mqtt


def generate_csi_amplitudes(num_subcarriers: int = 64, base_amp: float = 20.0) -> list[float]:
    """Generate realistic CSI amplitude values."""
    amplitudes = []
    for i in range(num_subcarriers):
        # Add some frequency-dependent variation
        freq_factor = 1.0 + 0.3 * math.sin(i * 0.2)
        # Add noise
        noise = random.gauss(0, 2)
        amp = base_amp * freq_factor + noise
        amplitudes.append(max(0, amp))
    return amplitudes


def generate_device_data(
    device_id: str,
    time_offset: float,
    presence_probability: float = 0.7
) -> dict[str, Any]:
    """Generate realistic CSI analysis data for a device."""
    # Simulate time-varying signal
    rssi = -50 + 10 * math.sin(time_offset * 0.5) + random.gauss(0, 3)

    # Generate amplitudes with some temporal variation
    base_amp = 20 + 5 * math.sin(time_offset * 0.3)
    amps = generate_csi_amplitudes(64, base_amp)

    # Simulate presence detection
    present = random.random() < presence_probability

    # Breath ratio varies when present
    if present:
        breath_ratio = 0.3 + 0.4 * abs(math.sin(time_offset * 0.1)) + random.gauss(0, 0.05)
        breath_ratio = max(0, min(1, breath_ratio))
        breathing = breath_ratio > 0.2
        breathing_rate = 12 + random.gauss(0, 2) if breathing else 0
    else:
        breath_ratio = random.gauss(0.05, 0.02)
        breath_ratio = max(0, min(0.15, breath_ratio))
        breathing = False
        breathing_rate = 0

    return {
        "device_id": device_id,
        "timestamp": time.time(),
        "rssi": round(rssi, 1),
        "amps": [round(a, 2) for a in amps],
        "present": present,
        "breath_ratio": round(breath_ratio, 3),
        "breathing": breathing,
        "breathing_rate": round(breathing_rate, 1) if breathing else None
    }


def main():
    parser = argparse.ArgumentParser(description="Generate dummy CSI data for testing")
    parser.add_argument("--mqtt-host", default="localhost", help="MQTT broker host")
    parser.add_argument("--mqtt-port", type=int, default=1883, help="MQTT broker port")
    parser.add_argument("--devices", type=int, default=3, help="Number of simulated devices")
    parser.add_argument("--rate", type=float, default=1.0, help="Data rate per device (Hz)")
    parser.add_argument("--duration", type=int, default=60, help="Duration in seconds (0=infinite)")
    args = parser.parse_args()

    # Connect to MQTT
    client = mqtt.Client(client_id=f"dummy-generator-{random.randint(1000, 9999)}")

    try:
        client.connect(args.mqtt_host, args.mqtt_port, 60)
        client.loop_start()
        print(f"Connected to MQTT broker at {args.mqtt_host}:{args.mqtt_port}")
    except Exception as e:
        print(f"Failed to connect to MQTT: {e}")
        return 1

    # Generate device IDs
    device_ids = [f"esp32-test-{i:02d}" for i in range(args.devices)]
    print(f"Simulating {len(device_ids)} devices: {device_ids}")
    print(f"Data rate: {args.rate} Hz per device")

    interval = 1.0 / args.rate
    start_time = time.time()
    message_count = 0

    try:
        while True:
            loop_start = time.time()
            elapsed = loop_start - start_time

            # Check duration
            if args.duration > 0 and elapsed >= args.duration:
                break

            # Generate and publish data for each device
            for device_id in device_ids:
                data = generate_device_data(
                    device_id,
                    elapsed,
                    presence_probability=0.6
                )

                topic = f"wavira/analysis/{device_id}"
                payload = json.dumps(data)
                client.publish(topic, payload, qos=0)
                message_count += 1

            # Status update every 10 seconds
            if message_count % (10 * args.devices) == 0:
                rate = message_count / elapsed if elapsed > 0 else 0
                print(f"[{elapsed:.0f}s] Published {message_count} messages ({rate:.1f} msg/s)")

            # Sleep to maintain rate
            sleep_time = interval - (time.time() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        client.loop_stop()
        client.disconnect()
        print(f"Total messages published: {message_count}")

    return 0


if __name__ == "__main__":
    exit(main())

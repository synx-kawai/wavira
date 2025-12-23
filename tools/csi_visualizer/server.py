#!/usr/bin/env python3
"""
CSI WebSocket Server - Improved Version
Issue #7: 呼吸・存在検知アルゴリズムの改善

Features:
- Improved breathing detection using BreathingDetector class
- Multi-subcarrier analysis
- Kalman filtering for noise reduction
- Presence detection
"""

import asyncio
import json
import re
import serial
import numpy as np
from collections import deque
import time
import threading
from breathing_detector import BreathingDetector

PORT = "/dev/cu.usbserial-110"
BAUD = 115200
WS_PORT = 8765

clients = set()
latest_data = None
data_lock = threading.Lock()

# Improved breathing detector
breathing_detector = BreathingDetector(sample_rate=10.0, history_seconds=15.0)
breathing_status = {
    "breathing": True,
    "present": False,
    "breath_ratio": 0.0,
    "breath_rate": 0,
    "hold_duration": 0.0,
    "confidence": 0.0
}


def parse_csi(line):
    if not line.startswith("CSI_DATA"):
        return None
    try:
        match = re.search(r'\[([^\]]+)\]', line)
        if not match:
            return None
        vals = [int(x) for x in match.group(1).split(',')]
        parts = line.split(',')
        amps = []
        for i in range(4, len(vals) - 1, 2):
            amps.append((vals[i]**2 + vals[i+1]**2) ** 0.5)
        return {
            "type": "csi",
            "pkt": int(parts[1]) if len(parts) > 1 else 0,
            "rssi": int(parts[3]) if len(parts) > 3 else 0,
            "ch": int(parts[4]) if len(parts) > 4 else 0,
            "amps": amps
        }
    except:
        return None


def update_breathing(amps, timestamp):
    """Update breathing detection with new CSI data."""
    global breathing_status

    if not amps:
        return

    amplitudes = np.array(amps, dtype=np.float32)
    state = breathing_detector.update(amplitudes, timestamp)

    breathing_status = {
        "breathing": bool(state.is_breathing),
        "present": bool(state.is_present),
        "breath_ratio": float(round(state.breath_ratio, 3)),
        "breath_rate": float(round(state.breath_rate, 1)),
        "hold_duration": float(round(state.hold_duration, 1)),
        "confidence": float(round(state.confidence, 2))
    }


def serial_thread():
    """Background thread for serial reading"""
    global latest_data

    print(f"[Serial] Connecting to {PORT}...")
    try:
        ser = serial.Serial(PORT, BAUD, timeout=1)
        time.sleep(1)
        ser.reset_input_buffer()
    except Exception as e:
        print(f"[Serial] Error: {e}")
        return

    print("[Serial] Connected - reading data...")
    count = 0

    while True:
        try:
            raw = ser.read(4096)
            if raw:
                text = raw.decode('utf-8', errors='ignore')
                for line in text.split('\n'):
                    line = line.strip()
                    if line.startswith("CSI_DATA"):
                        data = parse_csi(line)
                        if data and data['amps']:
                            now = time.time()
                            with data_lock:
                                latest_data = data
                                # Update breathing detection
                                update_breathing(data['amps'], now)
                            count += 1
                            if count % 50 == 0:
                                print(f"[Serial] {count} packets, "
                                      f"breathing={breathing_status['breathing']}, "
                                      f"rate={breathing_status['breath_rate']:.0f}/min")
        except Exception as e:
            print(f"[Serial] Error: {e}")
            time.sleep(1)


async def broadcaster():
    last_pkt = -1
    last_send = 0

    while True:
        now = time.time()

        # 100ms間隔で送信（10Hz）
        if now - last_send < 0.1:
            await asyncio.sleep(0.05)
            continue

        with data_lock:
            data = latest_data

        if data and clients:
            if data['pkt'] != last_pkt:
                last_pkt = data['pkt']
                last_send = now

                send_data = data.copy()
                send_data["breath"] = breathing_status.copy()

                msg = json.dumps(send_data)

                dead = set()
                for ws in list(clients):
                    try:
                        await asyncio.wait_for(ws.send(msg), timeout=0.2)
                    except:
                        dead.add(ws)

                for ws in dead:
                    clients.discard(ws)

        await asyncio.sleep(0.05)


async def handle_client(ws):
    clients.add(ws)
    print(f"[WS] Client connected ({len(clients)})")
    try:
        async for _ in ws:
            pass
    except:
        pass
    finally:
        clients.discard(ws)
        print(f"[WS] Client disconnected ({len(clients)})")


async def main():
    import websockets

    print("=" * 50)
    print("CSI Breathing Monitor Server (Improved)")
    print("=" * 50)

    # Start serial thread
    thread = threading.Thread(target=serial_thread, daemon=True)
    thread.start()

    # Start broadcaster
    asyncio.create_task(broadcaster())

    print(f"[WS] Starting server on port {WS_PORT}...")

    async with websockets.serve(handle_client, "0.0.0.0", WS_PORT):
        print(f"[WS] Server ready - open http://localhost:8080/index.html")
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown")

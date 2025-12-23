#!/usr/bin/env python3
"""
CSI WebSocket Server - Simple Debug Version
"""

import asyncio
import json
import re
import serial
import numpy as np
from collections import deque
import time as time_module

PORT = "/dev/cu.usbserial-2120"
BAUD = 115200
WS_PORT = 8765

clients = set()
latest_data = None
amp_history = deque(maxlen=30)
time_history = deque(maxlen=30)
breathing_status = {"breathing": True, "breath_ratio": 0.0, "breath_rate": 0, "hold_duration": 0.0}
breath_hold_start = None
data_count = 0


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
            amps.append(round((vals[i]**2 + vals[i+1]**2) ** 0.5, 1))
        return {
            "type": "csi",
            "pkt": int(parts[1]) if len(parts) > 1 else 0,
            "rssi": int(parts[3]) if len(parts) > 3 else 0,
            "ch": int(parts[4]) if len(parts) > 4 else 0,
            "amps": amps
        }
    except:
        return None


def analyze_breathing():
    global breathing_status, breath_hold_start

    if len(amp_history) < 15:
        return

    samples = np.array(amp_history)
    n = len(samples)

    if len(time_history) > 1:
        duration = time_history[-1] - time_history[0]
        sample_rate = n / duration if duration > 0 else 10
    else:
        sample_rate = 10

    if sample_rate < 1:
        sample_rate = 10

    try:
        fft_vals = np.abs(np.fft.rfft(samples * np.hanning(n)))
        freqs = np.fft.rfftfreq(n, 1/sample_rate)
        fft_vals[0] = 0
        total_power = np.sum(fft_vals[1:]) + 1e-10

        breath_mask = (freqs >= 0.15) & (freqs <= 0.5)
        breath_power = np.sum(fft_vals[breath_mask])
        breath_ratio = breath_power / total_power

        if np.any(breath_mask) and breath_power > 0:
            breath_freqs = freqs[breath_mask]
            breath_vals = fft_vals[breath_mask]
            peak_freq = breath_freqs[np.argmax(breath_vals)]
            breath_rate = int(peak_freq * 60)
        else:
            breath_rate = 0

        now = time_module.time()
        if breath_ratio < 0.10:
            if breathing_status["breathing"]:
                breath_hold_start = now
            breathing_status["breathing"] = False
            if breath_hold_start:
                breathing_status["hold_duration"] = now - breath_hold_start
        else:
            breathing_status["breathing"] = True
            breathing_status["hold_duration"] = 0.0
            breath_hold_start = None

        breathing_status["breath_ratio"] = round(breath_ratio, 3)
        breathing_status["breath_rate"] = breath_rate
    except Exception as e:
        print(f"[Analyze] Error: {e}")


async def serial_reader():
    global latest_data, data_count

    print(f"[Serial] Connecting to {PORT}...")
    try:
        ser = serial.Serial(PORT, BAUD, timeout=1)
        time_module.sleep(1)
        ser.reset_input_buffer()
    except Exception as e:
        print(f"[Serial] Error: {e}")
        return

    print("[Serial] Connected")

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
                            latest_data = data
                            data_count += 1
                            avg_amp = sum(data['amps']) / len(data['amps'])
                            amp_history.append(avg_amp)
                            time_history.append(time_module.time())
                            if data_count % 10 == 0:
                                print(f"[Serial] Received {data_count} packets")
            await asyncio.sleep(0.05)
        except Exception as e:
            print(f"[Serial] Error: {e}")
            await asyncio.sleep(1)


async def breathing_analyzer():
    while True:
        analyze_breathing()
        await asyncio.sleep(0.5)


async def broadcaster():
    last_pkt = -1
    send_count = 0

    while True:
        if latest_data and clients:
            if latest_data['pkt'] != last_pkt:
                last_pkt = latest_data['pkt']

                send_data = latest_data.copy()
                send_data["breath"] = breathing_status.copy()

                msg = json.dumps(send_data)

                for ws in list(clients):
                    try:
                        await asyncio.wait_for(ws.send(msg), timeout=0.1)
                        send_count += 1
                        if send_count % 20 == 0:
                            print(f"[WS] Sent {send_count} messages")
                    except Exception as e:
                        clients.discard(ws)
                        print(f"[WS] Client error: {e}")

        await asyncio.sleep(0.1)


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
    print("CSI WebSocket Server - Debug Version")
    print(f"Open browser: file:///Users/msduser/dev/wavira/tools/csi_visualizer/index.html")
    print("=" * 50)

    asyncio.create_task(serial_reader())
    asyncio.create_task(breathing_analyzer())
    asyncio.create_task(broadcaster())

    async with websockets.serve(handle_client, "0.0.0.0", WS_PORT):
        print(f"[WS] Server listening on port {WS_PORT}")
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown")

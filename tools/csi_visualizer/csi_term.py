#!/usr/bin/env python3
"""
CSI Terminal Visualizer - Buffer-based reading
"""

import sys
import re
import time
import serial

PORT = "/dev/cu.usbserial-2120"
BAUD = 115200
WIDTH = 50
HEIGHT = 12

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
            'pkt': int(parts[1]) if len(parts) > 1 else 0,
            'rssi': int(parts[3]) if len(parts) > 3 else 0,
            'ch': int(parts[4]) if len(parts) > 4 else 0,
            'amps': amps
        }
    except:
        return None

def draw(data):
    amps = data['amps']
    max_amp = max(amps) or 1

    step = max(1, len(amps) // WIDTH)
    sampled = [amps[i] for i in range(0, len(amps), step)][:WIDTH]

    sys.stdout.write('\033[H')  # Home
    print(f"\033[32mPKT:{data['pkt']:6d} RSSI:{data['rssi']:3d}dBm CH:{data['ch']:2d}\033[K")

    for row in range(HEIGHT, 0, -1):
        threshold = (row / HEIGHT) * max_amp
        line = "".join("█" if amp >= threshold else " " for amp in sampled)
        print(f"\033[32m{line}\033[0m\033[K")

    sys.stdout.flush()

def main():
    print(f"Connecting to {PORT}...")
    try:
        ser = serial.Serial(PORT, BAUD, timeout=0)  # Non-blocking
    except Exception as e:
        print(f"Error: {e}")
        return

    print("Connected. Ctrl+C to exit.")
    sys.stdout.write('\033[2J\033[?25l')  # Clear, hide cursor
    sys.stdout.flush()

    buffer = ""
    last_draw = 0

    try:
        while True:
            # Read available bytes (non-blocking)
            if ser.in_waiting:
                chunk = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
                buffer += chunk

                # Process complete lines
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip()

                    if line.startswith('CSI_DATA'):
                        data = parse_csi(line)
                        if data and data['amps']:
                            # Throttle display to ~20 FPS
                            now = time.time()
                            if now - last_draw >= 0.05:
                                draw(data)
                                last_draw = now

                # Prevent buffer overflow
                if len(buffer) > 10000:
                    buffer = buffer[-5000:]
            else:
                time.sleep(0.001)  # 1ms sleep when no data

    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write('\033[?25h\033[0m\n')
        ser.close()
        print("Closed.")

if __name__ == "__main__":
    main()

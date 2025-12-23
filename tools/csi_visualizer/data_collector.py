#!/usr/bin/env python3
"""
CSI Data Collector for Crowd Level Estimation
Issue #4: データ収集パイプライン構築

Usage:
    python data_collector.py --num_people 3 --duration 60 --output data/session_001.h5
"""

import argparse
import serial
import time
import re
import numpy as np
import h5py
from datetime import datetime
from pathlib import Path


def parse_csi(line: str) -> dict | None:
    """Parse CSI_DATA line and extract features."""
    if not line.startswith("CSI_DATA"):
        return None

    try:
        match = re.search(r'\[([^\]]+)\]', line)
        if not match:
            return None

        vals = [int(x) for x in match.group(1).split(',')]
        parts = line.split(',')

        # Calculate amplitude and phase for each subcarrier
        amplitudes = []
        phases = []
        for i in range(4, len(vals) - 1, 2):
            real, imag = vals[i], vals[i + 1]
            amp = np.sqrt(real**2 + imag**2)
            phase = np.arctan2(imag, real)
            amplitudes.append(amp)
            phases.append(phase)

        return {
            "packet_id": int(parts[1]) if len(parts) > 1 else 0,
            "mac": parts[2] if len(parts) > 2 else "",
            "rssi": int(parts[3]) if len(parts) > 3 else 0,
            "channel": int(parts[4]) if len(parts) > 4 else 0,
            "amplitudes": np.array(amplitudes, dtype=np.float32),
            "phases": np.array(phases, dtype=np.float32),
        }
    except Exception as e:
        return None


def collect_data(
    port: str,
    baud: int,
    duration: int,
    num_people: int,
    activity: str,
    output_path: Path
) -> None:
    """Collect CSI data with labels."""

    print(f"=" * 60)
    print(f"CSI Data Collector")
    print(f"=" * 60)
    print(f"Port: {port}")
    print(f"Duration: {duration} seconds")
    print(f"Label: {num_people} people, activity={activity}")
    print(f"Output: {output_path}")
    print(f"=" * 60)

    # Prepare data storage
    timestamps = []
    amplitudes_list = []
    phases_list = []
    rssi_list = []
    mac_list = []

    # Connect to serial
    ser = serial.Serial(port, baud, timeout=1)
    time.sleep(1)
    ser.reset_input_buffer()

    print(f"\nCollecting data... Press Ctrl+C to stop early.\n")

    start_time = time.time()
    packet_count = 0

    try:
        while time.time() - start_time < duration:
            data = ser.read(2000)
            if not data:
                continue

            text = data.decode('utf-8', errors='ignore')
            for line in text.split('\n'):
                parsed = parse_csi(line.strip())
                if parsed and len(parsed["amplitudes"]) > 0:
                    timestamps.append(time.time())
                    amplitudes_list.append(parsed["amplitudes"])
                    phases_list.append(parsed["phases"])
                    rssi_list.append(parsed["rssi"])
                    mac_list.append(parsed["mac"])
                    packet_count += 1

            # Progress update
            elapsed = time.time() - start_time
            if int(elapsed) % 10 == 0 and elapsed > 0:
                rate = packet_count / elapsed
                print(f"  [{int(elapsed):3d}s] Collected {packet_count} packets ({rate:.1f} Hz)")

    except KeyboardInterrupt:
        print("\n\nCollection interrupted by user.")

    finally:
        ser.close()

    # Save to HDF5
    if packet_count > 0:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Pad arrays to same length
        max_len = max(len(a) for a in amplitudes_list)
        amplitudes_padded = np.zeros((len(amplitudes_list), max_len), dtype=np.float32)
        phases_padded = np.zeros((len(phases_list), max_len), dtype=np.float32)

        for i, (amp, phase) in enumerate(zip(amplitudes_list, phases_list)):
            amplitudes_padded[i, :len(amp)] = amp
            phases_padded[i, :len(phase)] = phase

        with h5py.File(output_path, 'w') as f:
            # Data
            f.create_dataset('timestamps', data=np.array(timestamps))
            f.create_dataset('amplitudes', data=amplitudes_padded)
            f.create_dataset('phases', data=phases_padded)
            f.create_dataset('rssi', data=np.array(rssi_list))

            # Metadata
            f.attrs['num_people'] = num_people
            f.attrs['activity'] = activity
            f.attrs['duration'] = duration
            f.attrs['collection_date'] = datetime.now().isoformat()
            f.attrs['port'] = port
            f.attrs['packet_count'] = packet_count
            f.attrs['sample_rate'] = packet_count / (timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0

        print(f"\n{'=' * 60}")
        print(f"Collection Complete!")
        print(f"{'=' * 60}")
        print(f"Packets collected: {packet_count}")
        print(f"Sample rate: {packet_count / duration:.1f} Hz")
        print(f"Saved to: {output_path}")
    else:
        print("No data collected!")


def main():
    parser = argparse.ArgumentParser(description='CSI Data Collector for Crowd Level Estimation')
    parser.add_argument('--port', type=str, default='/dev/cu.usbserial-110',
                        help='Serial port')
    parser.add_argument('--baud', type=int, default=115200,
                        help='Baud rate')
    parser.add_argument('--duration', type=int, default=60,
                        help='Collection duration in seconds')
    parser.add_argument('--num_people', type=int, required=True,
                        help='Number of people in the area (label)')
    parser.add_argument('--activity', type=str, default='stationary',
                        choices=['stationary', 'moving', 'mixed'],
                        help='Activity type')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path (HDF5)')

    args = parser.parse_args()

    # Generate output filename if not specified
    if args.output is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = f'data/csi_{args.num_people}people_{args.activity}_{timestamp}.h5'

    output_path = Path(args.output)

    collect_data(
        port=args.port,
        baud=args.baud,
        duration=args.duration,
        num_people=args.num_people,
        activity=args.activity,
        output_path=output_path
    )


if __name__ == '__main__':
    main()

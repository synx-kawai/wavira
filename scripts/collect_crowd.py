#!/usr/bin/env python3
"""
Crowd level data collection script for CSI-based crowd estimation.

Usage:
    python scripts/collect_crowd.py --level 0 --location office --samples 100
    python scripts/collect_crowd.py --level 1 --location office --samples 100
    python scripts/collect_crowd.py --level 2 --location office --samples 100

Crowd levels:
    0: Empty (0-1 people)
    1: Moderate (2-5 people)
    2: Crowded (6+ people)
"""

import serial
import time
import json
import numpy as np
import os
import argparse
from datetime import datetime


CROWD_LEVELS = {
    0: "empty",      # 0-1 people
    1: "moderate",   # 2-5 people
    2: "crowded",    # 6+ people
}


def collect_crowd_csi(
    port: str,
    baud: int,
    output_dir: str,
    level: int,
    location: str,
    samples_per_file: int = 100,
    num_files: int = 10,
    timeout_sec: int = 300,
):
    """
    Collect CSI data for crowd level estimation.

    Args:
        port: Serial port
        baud: Baud rate
        output_dir: Output directory
        level: Crowd level (0, 1, 2)
        location: Location identifier
        samples_per_file: Number of CSI packets per file
        num_files: Number of files to collect
        timeout_sec: Global timeout in seconds
    """
    # Create output directory structure
    level_name = CROWD_LEVELS.get(level, f"level_{level}")
    save_dir = os.path.join(output_dir, location, level_name)
    os.makedirs(save_dir, exist_ok=True)

    # Generate session ID
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print(f"Crowd Level Data Collection")
    print("=" * 60)
    print(f"Location:    {location}")
    print(f"Level:       {level} ({level_name})")
    print(f"Output:      {save_dir}")
    print(f"Target:      {num_files} files x {samples_per_file} samples")
    print(f"Session:     {session_id}")
    print("=" * 60)

    # Confirm with user
    input(f"\nPress ENTER to start collecting '{level_name}' data...")

    print(f"\nConnecting to {port} ({baud} baud)...")

    try:
        s = serial.Serial(port, baud, timeout=0.1)
        # Soft reset
        s.setDTR(False)
        s.setRTS(True)
        time.sleep(0.2)
        s.setRTS(False)
        time.sleep(0.2)
        s.reset_input_buffer()
        time.sleep(1)
        print("Connected. Waiting for CSI data...")
    except Exception as e:
        print(f"Error opening serial port: {e}")
        return

    packet_buffer = []
    file_count = 0
    total_packets = 0
    start_time = time.time()

    try:
        while file_count < num_files:
            # Check global timeout
            elapsed = time.time() - start_time
            if elapsed > timeout_sec:
                print(f"\nTimeout reached ({timeout_sec}s). Stopping.")
                break

            # Read line
            try:
                line = s.readline()
            except Exception as e:
                continue

            if not line:
                continue

            try:
                decoded = line.decode('utf-8', errors='ignore').strip()
            except:
                continue

            if not decoded or "CSI_DATA" not in decoded:
                continue

            # Process CSI Data
            try:
                idx = decoded.find("CSI_DATA")
                csv_part = decoded[idx:]
                parts = csv_part.split(',')

                if len(parts) < 25:
                    continue

                json_str = parts[-1].strip()
                if json_str.startswith('"'):
                    json_str = json_str[1:]
                if json_str.endswith('"'):
                    json_str = json_str[:-1]

                csi_raw_data = json.loads(json_str)
                csi_len = int(parts[-3])

                if len(csi_raw_data) != csi_len:
                    continue

                # Convert to complex
                csi_complex = []
                for i in range(csi_len // 2):
                    real = csi_raw_data[i * 2 + 1]
                    imag = csi_raw_data[i * 2]
                    csi_complex.append(complex(real, imag))

                packet_buffer.append(csi_complex)
                total_packets += 1

                # Progress display
                progress = len(packet_buffer) / samples_per_file * 100
                overall = (file_count * samples_per_file + len(packet_buffer)) / (num_files * samples_per_file) * 100
                print(f"\rFile {file_count + 1}/{num_files} | "
                      f"Packets: {len(packet_buffer)}/{samples_per_file} | "
                      f"Overall: {overall:.1f}%", end="", flush=True)

                # Save file when buffer is full
                if len(packet_buffer) >= samples_per_file:
                    data_array = np.array(packet_buffer)
                    data_array = np.transpose(data_array, (1, 0))
                    data_array = np.expand_dims(data_array, axis=0)

                    filename = f"{session_id}_{file_count:04d}.npy"
                    filepath = os.path.join(save_dir, filename)
                    np.save(filepath, data_array)

                    # Save metadata
                    meta = {
                        "level": level,
                        "level_name": level_name,
                        "location": location,
                        "session_id": session_id,
                        "file_index": file_count,
                        "samples": samples_per_file,
                        "timestamp": datetime.now().isoformat(),
                    }
                    meta_path = filepath.replace(".npy", ".json")
                    with open(meta_path, 'w') as f:
                        json.dump(meta, f, indent=2)

                    print(f"\n  Saved: {filename}")
                    packet_buffer = []
                    file_count += 1

            except Exception as e:
                continue

    except KeyboardInterrupt:
        print("\n\nCollection interrupted by user.")
    finally:
        if 's' in locals() and s.is_open:
            s.close()

    print("\n" + "=" * 60)
    print(f"Collection Complete")
    print(f"  Files saved: {file_count}")
    print(f"  Total packets: {total_packets}")
    print(f"  Location: {save_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Collect CSI data for crowd level estimation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Crowd Levels:
  0: Empty (0-1 people)
  1: Moderate (2-5 people)
  2: Crowded (6+ people)

Example:
  python scripts/collect_crowd.py --level 0 --location office
  python scripts/collect_crowd.py --level 1 --location office
  python scripts/collect_crowd.py --level 2 --location office
        """
    )

    parser.add_argument(
        "-l", "--level",
        type=int,
        required=True,
        choices=[0, 1, 2],
        help="Crowd level (0=empty, 1=moderate, 2=crowded)"
    )
    parser.add_argument(
        "--location",
        type=str,
        default="room1",
        help="Location identifier (default: room1)"
    )
    parser.add_argument(
        "-p", "--port",
        default="/dev/cu.usbserial-5AE90127161",
        help="Serial port"
    )
    parser.add_argument(
        "-b", "--baud",
        type=int,
        default=115200,
        help="Baud rate"
    )
    parser.add_argument(
        "-o", "--output",
        default="data/crowd",
        help="Output directory (default: data/crowd)"
    )
    parser.add_argument(
        "-n", "--num-files",
        type=int,
        default=10,
        help="Number of files to collect (default: 10)"
    )
    parser.add_argument(
        "-s", "--samples",
        type=int,
        default=100,
        help="Samples per file (default: 100)"
    )
    parser.add_argument(
        "-t", "--timeout",
        type=int,
        default=300,
        help="Timeout in seconds (default: 300)"
    )

    args = parser.parse_args()

    collect_crowd_csi(
        port=args.port,
        baud=args.baud,
        output_dir=args.output,
        level=args.level,
        location=args.location,
        samples_per_file=args.samples,
        num_files=args.num_files,
        timeout_sec=args.timeout,
    )


if __name__ == "__main__":
    main()

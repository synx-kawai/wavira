import serial
import time
import sys
import csv
import json
import numpy as np
import os

def collect_csi(port, baud, output_dir, samples_per_file=100, timeout_sec=60):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Starting CSI collection on {port} ({baud} baud) with timeout {timeout_sec}s...")
    start_time = time.time()
    
    try:
        s = serial.Serial(port, baud, timeout=0.1) # Short timeout for responsive reading
        # Soft reset
        s.setDTR(False)
        s.setRTS(True)
        time.sleep(0.2)
        s.setRTS(False)
        time.sleep(0.2)
        print("Reset triggered. Flushing buffer...", flush=True)
        s.reset_input_buffer()
        time.sleep(1) # Wait for boot logs to start
    except Exception as e:
        print(f"Error opening serial port: {e}", flush=True)
        return

    packet_buffer = []
    file_count = 0
    total_packets = 0
    raw_buffer = ""
    last_activity = time.time()
    
    print("Waiting for CSI data...", flush=True)
    
    try:
        while True:
            # Check global timeout
            if time.time() - start_time > timeout_sec:
                print(f"\nTimeout reached ({timeout_sec}s). Stopping collection.", flush=True)
                break

            # Read line with timeout
            try:
                line = s.readline()
            except Exception as e:
                print(f"Read error: {e}", flush=True)
                continue

            if not line:
                # No data yet
                if time.time() - last_activity > 2.0:
                    print(".", end="", flush=True)
                    last_activity = time.time()
                continue
            
            try:
                decoded = line.decode('utf-8', errors='ignore').strip()
            except:
                continue

            if not decoded:
                continue

            # Show progress logs
            if "CSI_DATA" not in decoded:
                if decoded.startswith("I (") or "IP" in decoded:
                    print(f"\n[{decoded}]", flush=True)
                continue

            # Process CSI Data
            try:
                # Find start of CSI_DATA
                idx = decoded.find("CSI_DATA")
                csv_part = decoded[idx:]
                parts = csv_part.split(',')

                if len(parts) < 25:
                    continue

                json_str = parts[-1]
                # Handle potential quoting
                json_str = json_str.strip()
                if json_str.startswith('"'): json_str = json_str[1:]
                if json_str.endswith('"'): json_str = json_str[:-1]

                csi_raw_data = json.loads(json_str)
                csi_len = int(parts[-3])
                
                if len(csi_raw_data) != csi_len:
                    continue

                csi_complex = []
                for i in range(csi_len // 2):
                    real = csi_raw_data[i * 2 + 1]
                    imag = csi_raw_data[i * 2]
                    csi_complex.append(complex(real, imag))
                
                packet_buffer.append(csi_complex)
                total_packets += 1
                if total_packets % 10 == 0 or len(packet_buffer) == 1:
                        print(f"\rCollected: {len(packet_buffer)}/{samples_per_file} (Total: {total_packets})", end="", flush=True)

                if len(packet_buffer) >= samples_per_file:
                    data_array = np.array(packet_buffer)
                    data_array = np.transpose(data_array, (1, 0))
                    data_array = np.expand_dims(data_array, axis=0)
                    
                    filename = os.path.join(output_dir, f"csi_sample_{file_count}.npy")
                    np.save(filename, data_array)
                    print(f"\nSaved {filename}", flush=True)
                    
                    packet_buffer = []
                    file_count += 1
                    if file_count >= 1: 
                        print("\nCollection complete.", flush=True)
                        return
            except Exception as e:
                # print(f"\nParse Error: {e}", flush=True) # Optional debug
                continue
    except KeyboardInterrupt:
        print("\nCollection stopped.", flush=True)
    finally:
        if 's' in locals() and s.isOpen():
            s.close()
            print("\nSerial port closed.", flush=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", default="/dev/cu.usbserial-5AE90127161")
    parser.add_argument("-b", "--baud", type=int, default=115200)
    parser.add_argument("-o", "--out", default="data/raw_csi")
    parser.add_argument("-n", "--count", type=int, default=100)
    parser.add_argument("-t", "--timeout", type=int, default=60, help="Global timeout in seconds")
    
    args = parser.parse_args()
    collect_csi(args.port, args.baud, args.out, args.count, args.timeout)

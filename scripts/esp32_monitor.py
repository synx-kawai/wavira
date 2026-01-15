#!/usr/bin/env python3
"""
ESP32 Serial Monitor - Safe, timeout-enabled serial port reader.

This script is designed to be used by Claude Code without causing freezes.
It uses pyserial with proper timeout handling.

Usage:
    python scripts/esp32_monitor.py                    # Auto-detect all devices
    python scripts/esp32_monitor.py --port /dev/cu.usbserial-2110
    python scripts/esp32_monitor.py --duration 5      # Read for 5 seconds
    python scripts/esp32_monitor.py --list            # List available ports
"""

import argparse
import glob
import sys
import time
from datetime import datetime

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    print("Error: pyserial not installed. Run: pip install pyserial", file=sys.stderr)
    sys.exit(1)


def list_esp32_ports() -> list[str]:
    """List all USB serial ports that might be ESP32 devices."""
    patterns = [
        "/dev/cu.usbserial-*",
        "/dev/cu.SLAB_USBtoUART*",
        "/dev/ttyUSB*",
    ]

    ports = []
    for pattern in patterns:
        ports.extend(glob.glob(pattern))

    return sorted(ports)


def read_serial_safe(port: str, baud: int = 115200, duration: float = 5.0) -> dict:
    """
    Safely read from serial port with timeout.

    Returns:
        dict with keys: success, lines, error, stats
    """
    result = {
        "port": port,
        "success": False,
        "lines": [],
        "error": None,
        "stats": {
            "total_bytes": 0,
            "line_count": 0,
            "csi_count": 0,
            "mqtt_status": None,
            "wifi_status": None,
        }
    }

    try:
        ser = serial.Serial(
            port=port,
            baudrate=baud,
            timeout=0.5,  # Short read timeout
            write_timeout=1.0,
        )
    except serial.SerialException as e:
        result["error"] = f"Cannot open port: {e}"
        return result

    try:
        end_time = time.time() + duration

        while time.time() < end_time:
            if ser.in_waiting:
                try:
                    line = ser.readline()
                    result["stats"]["total_bytes"] += len(line)

                    decoded = line.decode('utf-8', errors='ignore').strip()
                    if decoded:
                        result["lines"].append(decoded)
                        result["stats"]["line_count"] += 1

                        # Parse status from log lines
                        if "CSI_DATA" in decoded or '"data":' in decoded:
                            result["stats"]["csi_count"] += 1
                        if "MQTT" in decoded:
                            if "connected" in decoded.lower():
                                result["stats"]["mqtt_status"] = "connected"
                            elif "disconnect" in decoded.lower():
                                result["stats"]["mqtt_status"] = "disconnected"
                        if "Wi-Fi" in decoded or "WIFI" in decoded:
                            if "connected" in decoded.lower():
                                result["stats"]["wifi_status"] = "connected"
                            elif "disconnect" in decoded.lower():
                                result["stats"]["wifi_status"] = "disconnected"
                except Exception:
                    pass
            else:
                time.sleep(0.05)

        result["success"] = True

    except Exception as e:
        result["error"] = str(e)
    finally:
        ser.close()

    return result


def print_device_status(result: dict) -> None:
    """Print formatted device status."""
    port_name = result["port"].split("/")[-1]
    stats = result["stats"]

    print(f"\n{'='*60}")
    print(f"Device: {port_name}")
    print(f"{'='*60}")

    if not result["success"]:
        print(f"  ERROR: {result['error']}")
        return

    # Status summary
    wifi = stats.get("wifi_status", "unknown")
    mqtt = stats.get("mqtt_status", "unknown")

    print(f"  Wi-Fi:      {wifi}")
    print(f"  MQTT:       {mqtt}")
    print(f"  CSI Data:   {stats['csi_count']} packets")
    print(f"  Total:      {stats['line_count']} lines, {stats['total_bytes']} bytes")

    # Last few log lines
    if result["lines"]:
        show_all = len(result["lines"]) <= 50
        if show_all:
            print(f"\n  All output ({len(result['lines'])} lines):")
            for line in result["lines"]:
                # Truncate long lines
                display = line[:100] + "..." if len(line) > 100 else line
                print(f"    {display}")
        else:
            print(f"\n  Recent output (last 30 of {len(result['lines'])} lines):")
            for line in result["lines"][-30:]:
                # Truncate long lines
                display = line[:100] + "..." if len(line) > 100 else line
                print(f"    {display}")


def main():
    parser = argparse.ArgumentParser(
        description="Safe ESP32 serial monitor with timeout"
    )
    parser.add_argument(
        "--port", "-p",
        help="Serial port (default: auto-detect all)"
    )
    parser.add_argument(
        "--baud", "-b",
        type=int,
        default=115200,
        help="Baud rate (default: 115200)"
    )
    parser.add_argument(
        "--duration", "-d",
        type=float,
        default=5.0,
        help="Read duration in seconds (default: 5)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available ports and exit"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format"
    )

    args = parser.parse_args()

    # List ports
    if args.list:
        ports = list_esp32_ports()
        if ports:
            print("Available ESP32 serial ports:")
            for p in ports:
                print(f"  {p}")
        else:
            print("No ESP32 serial ports found")
        return

    # Determine ports to scan
    if args.port:
        ports = [args.port]
    else:
        ports = list_esp32_ports()
        if not ports:
            print("No ESP32 serial ports detected", file=sys.stderr)
            sys.exit(1)

    print(f"ESP32 Status Check - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Scanning {len(ports)} device(s) for {args.duration}s each...")

    results = []
    for port in ports:
        result = read_serial_safe(port, args.baud, args.duration)
        results.append(result)
        print_device_status(result)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    active = sum(1 for r in results if r["success"] and r["stats"]["line_count"] > 0)
    csi_active = sum(1 for r in results if r["stats"]["csi_count"] > 0)

    print(f"  Devices responding:     {active}/{len(results)}")
    print(f"  Devices sending CSI:    {csi_active}/{len(results)}")

    if args.json:
        import json
        print("\nJSON Output:")
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

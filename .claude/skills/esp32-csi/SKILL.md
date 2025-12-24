---
name: esp32-csi
description: Use this skill when working with ESP32 CSI hardware, including serial port monitoring, firmware flashing, CSI data collection, or troubleshooting ESP32 connectivity issues.
---

# ESP32 CSI Hardware Operations

This skill provides expertise for ESP32 Channel State Information (CSI) operations.

## Capabilities

- Serial port detection and monitoring
- Firmware flashing with esptool
- CSI data collection and validation
- Troubleshooting connectivity issues

## Serial Port Detection

```bash
# macOS
ls /dev/cu.usbserial-*

# Linux
ls /dev/ttyUSB*
```

## Firmware Flashing

Use the pre-built firmware for consistent operation:

```bash
esptool --chip esp32 --port /dev/cu.usbserial-XX --baud 460800 \
  write_flash 0x0 tools/csi_visualizer/firmware/esp32_csi_firmware.bin
```

## CSI Data Format

ESP32 outputs CSI data in CSV format:
```
CSI_DATA,<mac>,<rssi>,<rate>,<sig_mode>,<mcs>,<bandwidth>,<smoothing>,<not_sounding>,<aggregation>,<stbc>,<fec_coding>,<sgi>,<noise_floor>,<ampdu_cnt>,<channel>,<secondary_channel>,<local_timestamp>,<ant>,<sig_len>,<rx_state>,<len>,<data>
```

## Common Issues

1. **No serial port detected**: Check USB cable, try different port
2. **Permission denied**: Add user to dialout group (Linux) or use sudo
3. **No CSI data**: Verify WiFi credentials in firmware config
4. **Garbled output**: Check baud rate (default: 115200)

## Data Collection

```bash
# Collect crowd level data
python scripts/collect_crowd.py --level 0 --location office --num-files 10
```

## Visualizer Dashboard

```bash
# Start multi-device dashboard
python tools/csi_visualizer/dashboard.py
```

Configuration in `tools/csi_visualizer/config.yaml`.

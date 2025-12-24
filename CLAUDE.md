# Wavira Project Guidelines

## Project Overview

Wavira is a Wi-Fi CSI (Channel State Information) based person re-identification and crowd estimation system using ESP32 devices and deep learning (WhoFi architecture).

## Technology Stack

- **Language**: Python 3.9+
- **ML Framework**: PyTorch 2.0+
- **Hardware**: ESP32 with CSI support (ESP32-WROOM, ESP32-S3)
- **Build System**: ESP-IDF for firmware
- **Testing**: pytest

## Project Structure

```
wavira/
├── wavira/           # Core library
│   ├── models/       # WhoFi model architecture
│   ├── losses/       # In-batch negative loss
│   ├── training/     # Training utilities
│   └── utils/        # Metrics, ESP32 serial
├── scripts/          # CLI tools
├── tools/            # CSI visualizer dashboard
├── esp-csi/          # ESP-IDF CSI firmware
├── tests/            # pytest tests
└── data/             # CSI data (gitignored)
```

## Development Commands

```bash
# Install dependencies
pip install -e .

# Run tests
pytest tests/ -v

# Train with synthetic data
python scripts/train.py --use_synthetic --epochs 50

# Collect CSI data
python scripts/collect_crowd.py --level 0 --location office

# Start CSI visualizer dashboard
python tools/csi_visualizer/dashboard.py
```

## Code Style

- Follow PEP 8
- Use type hints for function signatures
- Docstrings in Google style
- Max line length: 100 characters

## ESP32 Hardware Notes

- Serial ports on macOS: `/dev/cu.usbserial-*`
- Default baud rate: 115200
- CSI output format: `CSI_DATA,<mac>,<rssi>,<rate>,<noise>,<data>`
- Use pre-built firmware from `tools/csi_visualizer/firmware/esp32_csi_firmware.bin`

## Testing Guidelines

- All new features must have tests
- Run `pytest tests/ -v` before committing
- Test files should match `test_*.py` pattern
- Use fixtures for common setup

## Git Workflow

- Branch naming: `feature/<description>`, `fix/<description>`
- Commit format: Conventional Commits (feat:, fix:, docs:, test:, chore:)
- Binary files use Git LFS (*.bin)

## Security Considerations

- Never commit WiFi credentials or API keys
- Use environment variables for sensitive data
- Serial port paths are machine-specific; use placeholders in docs

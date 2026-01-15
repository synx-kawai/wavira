# Development Guide

This guide covers setting up a local development environment for Wavira.

## Prerequisites

- Python 3.9+
- pip or uv package manager
- Git
- Docker and Docker Compose (for services)
- ESP-IDF (for firmware development, optional)

## Quick Setup

```bash
# Clone the repository
git clone https://github.com/your-org/wavira.git
cd wavira

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install with development dependencies
pip install -e ".[dev]"

# Verify installation
pytest tests/ -v
```

## Project Structure

```
wavira/
├── wavira/                 # Core library
│   ├── models/             # Neural network architectures
│   │   ├── encoder.py      # WhoFi encoder
│   │   ├── crowd_estimator.py
│   │   └── gesture_recognizer.py
│   ├── losses/             # Loss functions
│   ├── training/           # Training utilities
│   ├── data/               # Dataset classes
│   └── utils/              # Helper utilities
├── scripts/                # CLI tools
├── tools/                  # Auxiliary tools
│   └── csi_visualizer/     # Real-time dashboard
│       └── services/       # Backend services
├── esp-csi/                # ESP32 firmware
├── tests/                  # Test suite
├── docs/                   # Documentation
└── data/                   # Data directory (gitignored)
```

## Running Services Locally

### MQTT Broker (Mosquitto)

```bash
cd tools/csi_visualizer
docker-compose up mosquitto -d
```

### Full Stack

```bash
cd tools/csi_visualizer
docker-compose up -d
```

Services:
- MQTT Broker: `localhost:1883` (MQTT), `localhost:9001` (WebSocket)
- History Collector: `localhost:8080`
- CSI Processor: Background service

### Dashboard

Open `tools/csi_visualizer/dashboard_multi.html` in a browser after starting services.

## Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=wavira --cov-report=html

# Specific test file
pytest tests/test_encoder.py -v

# Run only fast tests
pytest tests/ -v -m "not slow"
```

## Code Quality

```bash
# Format code
black wavira/ scripts/ tests/
isort wavira/ scripts/ tests/

# Lint
flake8 wavira/ scripts/ tests/
pylint wavira/

# Type check
mypy wavira/
```

## ESP32 Firmware Development

### Prerequisites

1. Install ESP-IDF v5.0+
2. Set up IDF environment: `. $IDF_PATH/export.sh`

### Build and Flash

```bash
cd esp-csi/examples/get-started/csi_recv

# Configure
idf.py menuconfig

# Build
idf.py build

# Flash
idf.py -p /dev/cu.usbserial-0001 flash monitor
```

### Using Pre-built Firmware

```bash
cd esp-csi/examples/get-started/csi_recv

# Flash pre-built binary
esptool.py --chip esp32 -p /dev/cu.usbserial-0001 \
    write_flash 0x0 firmware/esp32_csi_firmware.bin
```

## Training Models

### Crowd Estimation

```bash
# With synthetic data (development)
python scripts/train_crowd.py --synthetic --epochs 50

# With real data
python scripts/train_crowd.py --data_dir data/crowd_csi/ --epochs 100
```

### Gesture Recognition

```bash
# With synthetic data
python scripts/train_gesture.py --use_synthetic --epochs 25

# With real data
python scripts/train_gesture.py --data_dir data/gestures/ --epochs 50
```

## Debugging

### CSI Data Collection

```bash
# Monitor serial output
python scripts/esp32_monitor.py --port /dev/cu.usbserial-0001

# Save to file
python scripts/collect_csi.py --output data/session_001/
```

### MQTT Debugging

```bash
# Subscribe to all topics
mosquitto_sub -h localhost -t "wavira/#" -v

# Publish test data
mosquitto_pub -h localhost -t "wavira/csi/device1" -m '{"rssi": -50}'
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MQTT_HOST` | MQTT broker hostname | `localhost` |
| `MQTT_PORT` | MQTT broker port | `1883` |
| `API_KEYS` | Comma-separated API keys | None |
| `REQUIRE_API_KEY` | Enable API authentication | `false` |
| `RATE_LIMIT_ENABLED` | Enable rate limiting | `true` |
| `LOG_LEVEL` | Logging level | `INFO` |

## Common Issues

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common problems and solutions.

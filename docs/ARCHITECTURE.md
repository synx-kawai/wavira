# System Architecture

This document describes the high-level architecture of the Wavira system.

## Overview

Wavira is a Wi-Fi CSI (Channel State Information) based sensing system for:
- Person re-identification (WhoFi architecture)
- Crowd level estimation (4-class classification)
- Gesture recognition (3D CNN)

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ESP32 Devices                                 │
│  ┌──────────────┐  ┌──────────────┐                                 │
│  │  ESP32-S3    │  │  ESP32-S3    │     (Up to N devices)           │
│  │  CSI TX/RX   │  │  CSI TX/RX   │                                 │
│  └──────┬───────┘  └──────┬───────┘                                 │
└─────────┼─────────────────┼─────────────────────────────────────────┘
          │                 │
          │  Serial/ESP-NOW │
          ▼                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      MQTT Broker (Mosquitto)                         │
│  Topics:                                                             │
│    wavira/csi/<device_id>    → Raw CSI amplitude data               │
│    wavira/analysis/<type>    → Inference results                    │
│    wavira/device/<id>/status → Device heartbeat                     │
└────────────┬───────────────────────────┬────────────────────────────┘
             │                           │
             ▼                           ▼
┌────────────────────────┐    ┌────────────────────────────┐
│   CSI Processor        │    │   History Collector        │
│   ─────────────────    │    │   ──────────────────       │
│   • Preprocessing      │    │   • SQLite storage         │
│   • Model inference    │    │   • REST API (:8080)       │
│   • Real-time analysis │    │   • Data aggregation       │
└────────────┬───────────┘    └────────────────────────────┘
             │
             ▼
┌────────────────────────┐    ┌────────────────────────────┐
│   Trained Models       │    │   Web Dashboard            │
│   ─────────────────    │    │   ──────────────────       │
│   • CrowdEstimator     │    │   • Real-time plots        │
│   • GestureRecognizer  │    │   • Multi-device view      │
│   • WhoFi (Re-ID)      │    │   • Alerts & history       │
└────────────────────────┘    └────────────────────────────┘
```

## Project Structure

```
wavira/
├── wavira/                 # Core Python library
│   ├── models/             # Neural network architectures
│   │   ├── whofi.py        # Person re-identification (WhoFi)
│   │   ├── encoder.py      # Transformer/LSTM encoders
│   │   ├── crowd_estimator.py  # Crowd level estimation
│   │   └── gesture_recognizer.py  # Gesture recognition
│   ├── losses/             # Loss functions
│   │   └── inbatch_loss.py # In-batch negative sampling
│   ├── training/           # Training infrastructure
│   │   └── trainer.py      # Training loop & config
│   ├── data/               # Dataset classes
│   │   ├── dataset.py      # CSIDataset
│   │   ├── crowd_dataset.py
│   │   └── gesture_dataset.py
│   └── utils/              # Utilities
│       ├── metrics.py      # CMC, mAP evaluation
│       └── esp32_serial.py # Serial communication
├── scripts/                # CLI tools
│   ├── train.py            # Model training
│   ├── collect_crowd_mqtt.py   # MQTT data collection
│   ├── monitor_crowd_mqtt.py   # Real-time inference
│   └── gesture_inference.py    # Gesture recognition
├── tools/csi_visualizer/   # Dashboard & services
│   ├── services/           # Python services
│   ├── *.html              # Web UI
│   └── docker-compose.yml  # Container orchestration
├── esp-csi/                # ESP-IDF firmware
├── tests/                  # pytest test suite
├── checkpoints/            # Trained models
└── data/                   # Training data (gitignored)
```

## Components

### ESP32 Firmware

**Location**: `esp-csi/examples/get-started/csi_recv/`

Responsibilities:
- Capture CSI data from Wi-Fi signals
- Parse and format CSI packets
- Publish to MQTT broker
- Control LED indicators

CSI Data Format:
```
CSI_DATA,<mac>,<rssi>,<rate>,<noise>,<csi_data_hex>
```

### MQTT Broker (Mosquitto)

**Configuration**: `tools/csi_visualizer/mosquitto/`

Topics:
- `wavira/csi/<device_id>` - Raw CSI data
- `wavira/analysis/<type>` - Analysis results
- `wavira/device/<device_id>/status` - Device status
- `wavira/command/<device_id>` - Commands to devices

### CSI Processor

**Location**: `tools/csi_visualizer/services/csi_processor.py`

Responsibilities:
- Subscribe to CSI data topics
- Preprocess CSI samples
- Run inference (crowd estimation, gesture recognition)
- Publish analysis results

### History Collector

**Location**: `tools/csi_visualizer/services/history_collector.py`

Responsibilities:
- Store historical CSI data
- Provide REST API for queries
- Handle data aggregation
- Security (authentication, rate limiting)

API Endpoints:
- `GET /api/v1/devices` - List devices
- `GET /api/v1/history/{device_id}` - Get history
- `GET /api/v1/summary/{device_id}` - Hourly summary

### Dashboard

**Location**: `tools/csi_visualizer/dashboard_multi.html`

Features:
- Real-time CSI visualization
- Multi-device monitoring
- Amplitude/phase plots
- Device status indicators

## Core Library

### Models (`wavira/models/`)

#### WhoFi Person Re-identification (`whofi.py`)

The WhoFi model extracts L2-normalized signature vectors from CSI data for person re-identification.

```python
# Architecture
CSI Input (batch, 3, 114, seq_len)
    ↓
Reshape → (batch, seq_len, 3*114)
    ↓
TransformerEncoder (self-attention)
    ↓
Global Average Pooling
    ↓
Signature Module → Linear(d_model, 256)
    ↓
L2 Normalization → (batch, 256)
```

**Key Parameters:**
- Input: 3 antenna channels × 114 subcarriers × 200 time steps
- Output: 256-dimensional normalized embedding
- Encoder: Transformer (8 heads, 6 layers) or BiLSTM

#### Crowd Estimator (`crowd_estimator.py`)

Estimates crowd density from CSI amplitude patterns.

```python
# Architecture
CSI Input (batch, seq_len, n_subcarriers)
    ↓
TemporalBlock (1D CNN stack)
    ↓
TransformerEncoder
    ↓
Classification Head → (batch, 4)  # 4 crowd levels
```

**Crowd Levels:**
| Level | Label    | Description    |
|-------|----------|----------------|
| 0     | Empty    | 0 people       |
| 1     | Low      | 1-2 people     |
| 2     | Medium   | 3-5 people     |
| 3     | High     | 6+ people      |

**Key Parameters:**
- Input: 50 time steps × 52 subcarriers (adjustable)
- Mode: "classification" (4 classes) or "regression" (person count)
- Inference rate: ~10 Hz

#### Gesture Recognizer (`gesture_recognizer.py`)

3D CNN for spatiotemporal gesture patterns in CSI data.

```python
# Architecture
CSI Input (batch, 1, n_frames, n_routes, n_subcarriers)
    ↓
Upsample to 120×120
    ↓
3D CNN Blocks (5 layers: 64→128→256→512→512)
    ↓
Global Average Pooling
    ↓
FC → Dropout → FC → (batch, n_gestures)
```

**Supported Gestures (8 classes):**
- Zoom in/out, Circle CW/CCW
- Swipe left/right/up/down

**Variants:**
- `GestureRecognizer3DCNN`: Standard version
- `GestureRecognizerLite`: Reduced complexity
- `DualESP32GestureRecognizer`: Multi-device fusion

### Encoders (`encoder.py`)

Interchangeable sequence encoders:

| Encoder     | Description                  | Use Case          |
|-------------|------------------------------|-------------------|
| Transformer | Multi-head self-attention    | Best accuracy     |
| LSTM        | Unidirectional recurrent     | Lower latency     |
| BiLSTM      | Bidirectional recurrent      | Better context    |

```python
encoder = get_encoder(encoder_type="transformer", config=config)
```

### Loss Functions (`wavira/losses/`)

#### In-Batch Negative Loss (`inbatch_loss.py`)

Contrastive learning for re-identification:
- Uses other samples in batch as negatives
- Temperature-scaled cross-entropy
- Supports symmetric loss variant

```python
loss_fn = InBatchNegativeLoss(temperature=0.07, symmetric=True)
loss = loss_fn(embeddings, labels)
```

#### Triplet Loss

Alternative for metric learning:
- Mining strategies: hard, semi-hard, all
- Margin parameter for separation

### Data Processing (`wavira/data/`)

| Module                    | Purpose                          |
|---------------------------|----------------------------------|
| `dataset.py`              | Base CSIDataset (numpy/directory)|
| `crowd_dataset.py`        | HDF5 crowd datasets              |
| `gesture_dataset.py`      | Gesture sequence loading         |
| `preprocessing.py`        | Amplitude, phase extraction      |
| `gesture_preprocessing.py`| Gesture-specific transforms      |

**Preprocessing Pipeline:**
```python
Raw I/Q → Complex → Amplitude → Hampel Filter → Normalize
```

## Data Flow

### Real-time Inference Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Data Collection                              │
├─────────────────────────────────────────────────────────────────────┤
│  ESP32 Device                                                        │
│     │                                                                │
│     │  WiFi Beacon (1 Mbps, 40MHz)                                  │
│     ▼                                                                │
│  CSI Callback → Extract I/Q Pairs (52-114 subcarriers)              │
│     │                                                                │
│     │  ESP-NOW / Serial                                             │
│     ▼                                                                │
│  MQTT Publish → wavira/csi/<device_id>                              │
│     │           { "data": [I0,Q0,I1,Q1,...], "rssi": -45 }          │
└─────┼───────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         CSI Processor                                │
├─────────────────────────────────────────────────────────────────────┤
│  MQTT Subscribe → wavira/csi/#                                       │
│     │                                                                │
│     ▼                                                                │
│  Parse JSON → Extract I/Q → Complex Numbers                         │
│     │                                                                │
│     ▼                                                                │
│  Amplitude = |I + jQ| → Buffer (window_size=50)                     │
│     │                                                                │
│     ▼                                                                │
│  Model Inference (every 1 second)                                   │
│     │  ┌─────────────────────────────────────────────┐              │
│     │  │ CrowdEstimator: [batch, 50, 52] → [batch, 4]│              │
│     │  │ GestureRecognizer: [batch, 1, 32, 3, 52]    │              │
│     │  └─────────────────────────────────────────────┘              │
│     ▼                                                                │
│  MQTT Publish → wavira/analysis/crowd                               │
│                 { "level": 0, "confidence": 0.95, "label": "Empty" }│
└─────────────────────────────────────────────────────────────────────┘
```

### Training Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Data Collection                              │
├─────────────────────────────────────────────────────────────────────┤
│  scripts/collect_crowd_mqtt.py                                       │
│     │                                                                │
│     │  --level 0 --location office --samples 100                    │
│     ▼                                                                │
│  MQTT Subscribe → Buffer 100 samples → Save HDF5                    │
│                                                                      │
│  Output: data/crowd/empty_office_20260115_120000_0000.h5            │
│     ├── amplitudes: (100, 52) float32                               │
│     ├── rssi: (100,) int16                                          │
│     └── attrs: {level, level_name, num_people, location, ...}       │
└─────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Training                                     │
├─────────────────────────────────────────────────────────────────────┤
│  scripts/train_crowd.py                                              │
│     │                                                                │
│     ▼                                                                │
│  CrowdDataset (HDF5 loader + sliding window)                        │
│     │                                                                │
│     ▼                                                                │
│  DataLoader (batch_size=32, shuffle=True)                           │
│     │                                                                │
│     ▼                                                                │
│  CrowdEstimator.forward() → CrossEntropyLoss                        │
│     │                                                                │
│     ▼                                                                │
│  Adam Optimizer + StepLR Scheduler                                  │
│     │                                                                │
│     ▼                                                                │
│  TensorBoard Logging (loss, accuracy per epoch)                     │
│     │                                                                │
│     ▼                                                                │
│  Checkpoint: checkpoints/crowd/<timestamp>/final_model.pt           │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Formats

**MQTT CSI Message (JSON):**
```json
{
  "device_id": "esp32_01",
  "timestamp": 1736928000.123,
  "rssi": -45,
  "data": [12, 34, -5, 67, ...]  // I/Q pairs
}
```

**HDF5 Training File:**
```
/amplitudes    Dataset (N, n_subcarriers) float32
/rssi          Dataset (N,) int16
Attributes:
  level        int64  (0-3)
  level_name   string ("empty", "low", "medium", "high")
  num_people   int64
  location     string
  timestamp    string
  samples      int64
  n_subcarriers int64
```

**Model Checkpoint:**
```python
{
  "model_state_dict": {...},
  "optimizer_state_dict": {...},
  "epoch": 50,
  "config": {"mode": "classification", "n_subcarriers": 52, ...}
}
```

## Deployment

### Docker Compose

```yaml
# tools/csi_visualizer/docker-compose.yml
services:
  mosquitto:          # MQTT broker (:1883, :9001 WebSocket)
  csi_processor:      # Model inference service
  history_collector:  # REST API + SQLite (:8080)
  nginx:              # Reverse proxy (:80)
```

**Quick Start:**
```bash
cd tools/csi_visualizer
docker-compose up -d
```

### Standalone Scripts

For development or single-device deployments:

```bash
# Data Collection
python scripts/collect_crowd_mqtt.py --level 0 --mqtt-host localhost

# Real-time Monitoring
python scripts/monitor_crowd_mqtt.py \
  --model checkpoints/crowd/final_model.pt \
  --mqtt-host localhost
```

### Production Considerations

**Security:**
- Enable MQTT authentication (see `mosquitto.prod.conf`)
- Configure ACLs for topic access control
- Enable API key authentication (`X-API-Key` header)
- Set appropriate rate limits (default: 100 req/min)
- Use TLS for encrypted connections

**Scaling:**
- Multiple CSI Processors for load distribution
- Redis for cross-instance coordination
- PostgreSQL for production history storage

**Monitoring:**
- TensorBoard for training metrics
- Prometheus-compatible health endpoints
- Structured JSON logging

## Configuration Reference

### Model Hyperparameters

| Parameter       | WhoFi    | Crowd     | Gesture   |
|-----------------|----------|-----------|-----------|
| n_subcarriers   | 114      | 52        | 52        |
| hidden_dim      | 256      | 128       | 256       |
| n_heads         | 8        | 4         | -         |
| n_layers        | 6        | 4         | 5 (CNN)   |
| dropout         | 0.1      | 0.2       | 0.5       |
| output_dim      | 256      | 4         | 8         |

### Training Hyperparameters

| Parameter        | Default  | Description                  |
|------------------|----------|------------------------------|
| batch_size       | 32       | Samples per batch            |
| learning_rate    | 0.0001   | Adam optimizer LR            |
| epochs           | 300      | Maximum training epochs      |
| lr_decay         | 0.95     | LR multiplier per step       |
| lr_step_size     | 50       | Epochs between LR decay      |
| early_stopping   | 50       | Patience for early stop      |
| temperature      | 0.07     | Contrastive loss temp        |

### Environment Variables

```bash
# MQTT Configuration
MQTT_HOST=localhost
MQTT_PORT=1883
MQTT_USERNAME=wavira
MQTT_PASSWORD=secret

# API Security
API_KEY_ENABLED=true
API_KEYS=key1,key2,key3
RATE_LIMIT_PER_MINUTE=100

# Database
HISTORY_DB_PATH=/data/history.db
```

## Design Decisions

### Why MQTT?

- **Lightweight protocol** for IoT devices
- **Native WebSocket support** for browsers
- **Publish/subscribe pattern** fits data flow
- **Low latency** for real-time applications
- **QoS levels** for reliability guarantees

### Why Transformer Encoder?

- **Long-range dependencies** in CSI time series
- **Better performance** than LSTM on longer sequences (200+ packets)
- **Parallelizable** training (faster than RNNs)
- **Self-attention** visualizes feature importance

### Why 3D CNN for Gestures?

- **Spatiotemporal patterns** in CSI amplitude matrices
- **Translation invariance** for gesture variations
- **Efficient computation** compared to 3D Transformers
- **Proven architecture** from video classification

### Why HDF5 for Training Data?

- **Efficient storage** with compression (gzip)
- **Partial loading** for large datasets
- **Metadata support** via attributes
- **Cross-platform** compatibility

### Why In-Batch Negative Loss?

- **Memory efficient** compared to contrastive pairs
- **Scales with batch size** (more negatives = better)
- **Simple implementation** (just matrix operations)
- **Proven effectiveness** in metric learning

## Future Considerations

- Kubernetes deployment (Issue #45)
- OTA firmware updates (Issue #44)
- Breathing/presence detection (Issue #46)
- Model quantization for edge (Issue #43)
- ONNX export for cross-platform inference
- Federated learning for privacy-preserving training

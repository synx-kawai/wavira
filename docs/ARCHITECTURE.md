# System Architecture

This document describes the high-level architecture of the Wavira system.

## Overview

Wavira is a Wi-Fi CSI (Channel State Information) based sensing system for:
- Person re-identification
- Crowd level estimation
- Gesture recognition

```
┌─────────────┐     ┌─────────────┐     ┌─────────────────┐
│   ESP32     │────▶│    MQTT     │────▶│  CSI Processor  │
│  Devices    │     │   Broker    │     │                 │
└─────────────┘     └─────────────┘     └────────┬────────┘
                           │                     │
                           │                     ▼
                    ┌──────┴──────┐     ┌─────────────────┐
                    │  Dashboard  │     │    History      │
                    │  (Browser)  │     │   Collector     │
                    └─────────────┘     └─────────────────┘
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

#### WhoFi Encoder (`encoder.py`)
- Transformer-based CSI encoder
- Produces fixed-size embeddings for re-identification

```python
encoder = WhoFiEncoder(config)
embedding = encoder(csi_input)  # [batch, 128]
```

#### Crowd Estimator (`crowd_estimator.py`)
- Classification (4 levels) or regression mode
- Transformer or LSTM encoder options

```python
model = CrowdEstimator(config)
prediction = model(csi_window)  # [batch, 4] or [batch, 1]
```

#### Gesture Recognizer (`gesture_recognizer.py`)
- 3D CNN architecture
- Supports single and dual ESP32 configurations

```python
model = GestureRecognizer3DCNN(config)
prediction = model(csi_sequence)  # [batch, num_gestures]
```

### Data Processing (`wavira/data/`)

- `crowd_dataset.py` - Crowd estimation datasets
- `gesture_dataset.py` - Gesture recognition datasets
- `gesture_preprocessing.py` - Preprocessing pipeline

### Loss Functions (`wavira/losses/`)

- `in_batch_negative.py` - Contrastive loss for re-identification

## Data Flow

### Real-time Inference

```
ESP32 → MQTT → CSI Processor → Inference → MQTT → Dashboard
                    ↓
              History Collector → Database
```

### Training Pipeline

```
Raw CSI Data → Preprocessing → DataLoader → Model → Training Loop
                                                        ↓
                                               Checkpoint → Inference
```

## Deployment

### Docker Compose

```yaml
services:
  mosquitto:    # MQTT broker
  csi_processor: # Inference service
  history_collector: # API + storage
```

### Production Considerations

- Enable MQTT authentication (see `mosquitto.prod.conf`)
- Configure ACLs for topic access control
- Enable API key authentication
- Set appropriate rate limits
- Use TLS for encrypted connections

## Design Decisions

### Why MQTT?

- Lightweight protocol for IoT devices
- Native WebSocket support for browsers
- Publish/subscribe pattern fits data flow
- Low latency for real-time applications

### Why Transformer Encoder?

- Captures long-range dependencies in CSI data
- Better performance than LSTM on longer sequences
- Self-attention visualizes feature importance

### Why 3D CNN for Gestures?

- Captures spatiotemporal patterns
- Effective on CSI amplitude matrices
- Lightweight for edge deployment

## Future Considerations

- Kubernetes deployment (Issue #45)
- OTA firmware updates (Issue #44)
- Breathing/presence detection (Issue #46)
- Model quantization for edge (Issue #43)

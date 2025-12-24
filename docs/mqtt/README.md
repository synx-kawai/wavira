# MQTT Architecture for Wavira

This document describes the MQTT-based communication architecture for the Wavira CSI data collection system.

## Overview

Wavira uses MQTT as the primary protocol for transmitting CSI (Channel State Information) data from ESP32 devices to the central server. MQTT provides:

- **Low overhead**: Minimal protocol overhead compared to HTTP
- **Reliability**: QoS levels ensure message delivery
- **Real-time**: Publish/subscribe pattern for instant updates
- **Scalability**: Efficient handling of thousands of concurrent connections

## Components

```
┌─────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   ESP32     │────▶│  MQTT Broker    │────▶│   API Server    │
│  Devices    │     │  (Mosquitto)    │     │  (Python)       │
└─────────────┘     └─────────────────┘     └─────────────────┘
                            │                        │
                            │                        ▼
                            │               ┌─────────────────┐
                            │               │    Dashboard    │
                            │               │   (WebSocket)   │
                            └──────────────▶└─────────────────┘
```

## Quick Start

### 1. Start the MQTT Broker

```bash
cd docker
docker-compose up -d mosquitto
```

### 2. Configure ESP32

Set the following in `menuconfig`:
- `WAVIRA_MQTT_BROKER_URL`: Your broker URL
- `WAVIRA_MQTT_USERNAME`: Device username
- `WAVIRA_MQTT_PASSWORD`: Device password

### 3. Start the API Server

```bash
cd tools/csi_visualizer
python api_server.py
```

## Documentation

- [Topic Design](./topics.md) - MQTT topic structure and payload formats
- [Authentication](./authentication.md) - Device authentication and security
- [ESP32 Setup](./esp32-setup.md) - Firmware configuration guide
- [Troubleshooting](./troubleshooting.md) - Common issues and solutions

## Configuration

See [docker/mosquitto/](../../docker/mosquitto/) for broker configuration.

Environment variables:
- `MQTT_BROKER_HOST`: Broker hostname (default: localhost)
- `MQTT_BROKER_PORT`: Broker port (default: 1883)
- `MQTT_USERNAME`: Client username
- `MQTT_PASSWORD`: Client password

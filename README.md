# Wavira

Wi-Fi-based Person Re-Identification using Deep Learning

## Overview

Wavira is a deep learning framework for person re-identification using Wi-Fi Channel State Information (CSI). Based on the [WhoFi architecture](https://arxiv.org/abs/2507.12869), this system identifies individuals through their unique electromagnetic signatures captured from Wi-Fi signals, offering a privacy-preserving alternative to camera-based surveillance.

### Key Features

- **Non-visual identification**: Uses Wi-Fi signals instead of cameras, working through walls and in darkness
- **Privacy-preserving**: No identifiable visual data is captured or stored
- **Transformer-based architecture**: Leverages attention mechanisms for robust feature extraction
- **In-batch negative loss**: Efficient contrastive learning without explicit pair mining
- **Multiple encoder options**: Supports Transformer, LSTM, and Bi-LSTM architectures

### Performance

Based on the original WhoFi paper using the NTU-Fi dataset:

| Metric | Transformer | Bi-LSTM | LSTM |
|--------|-------------|---------|------|
| Rank-1 | 95.5% | 94.2% | 93.1% |
| mAP | 88.4% | 85.7% | 82.3% |

## Installation

### Requirements

- Python >= 3.9
- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- SciPy >= 1.10.0

### Install from source

```bash
git clone https://github.com/your-org/wavira.git
cd wavira
pip install -e .
```

### Install dependencies only

```bash
pip install -r requirements.txt
```

## Quick Start

### Training with synthetic data

For quick testing without real CSI data:

```bash
python scripts/train.py --use_synthetic --epochs 50
```

### Training with real data

Organize your CSI data in the following structure:

```
data/
├── person_001/
│   ├── sample_001.npy
│   ├── sample_002.npy
│   └── ...
├── person_002/
│   └── ...
└── ...
```

Each `.npy` file should contain CSI data in shape `(n_rx_antennas, n_subcarriers, n_packets)`.

```bash
python scripts/train.py --data_dir /path/to/data --epochs 300
```

### Evaluation

```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --data_dir /path/to/test_data
```

## Architecture

### Overview

```
CSI Input (3 x 114 x 200)
         │
         ▼
┌─────────────────┐
│  Reshape/Flat   │  → (200, 342)
└─────────────────┘
         │
         ▼
┌─────────────────┐
│    Encoder      │  Transformer / LSTM / Bi-LSTM
│  (+ Pos. Enc.)  │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│  Mean Pooling   │  → (342,)
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ Signature Module│  Linear → ReLU → Linear
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ L2 Normalize    │  → (256,)
└─────────────────┘
```

### CSI Preprocessing

1. **Amplitude Extraction**: Extract magnitude from complex CSI values
2. **Hampel Filtering**: Remove outliers using median absolute deviation
3. **Phase Sanitization**: Remove linear phase shifts caused by hardware offsets

### Training

- **Loss**: In-batch negative loss with cosine similarity
- **Optimizer**: Adam (lr=0.0001)
- **Scheduler**: StepLR (gamma=0.95 every 50 epochs)
- **Batch size**: 8
- **Epochs**: 300

## API Usage

### Basic inference

```python
import torch
from wavira import WhoFi

# Load model
model = WhoFi(
    n_channels=3,
    n_subcarriers=114,
    encoder_type="transformer",
    signature_dim=256,
)
model.load_state_dict(torch.load("model.pt")["model_state_dict"])
model.eval()

# Extract signature from CSI data
csi = torch.randn(1, 3, 114, 200)  # (batch, channels, subcarriers, packets)
signature = model(csi)  # (1, 256)

# Compare two signatures
similarity = model.compute_similarity(signature1, signature2)
```

### Custom training

```python
from wavira import WhoFi, CSIDataset, InBatchNegativeLoss
from wavira.training import Trainer, TrainingConfig

# Create dataset
dataset = CSIDataset(
    data_dir="/path/to/data",
    sequence_length=200,
    preprocess=True,
)

# Configure training
config = TrainingConfig(
    encoder_type="transformer",
    hidden_dim=256,
    signature_dim=256,
    batch_size=8,
    epochs=300,
    learning_rate=0.0001,
)

# Train
trainer = Trainer(config)
results = trainer.train(train_loader, val_loader)
```

### Preprocessing raw CSI

```python
from wavira.data import preprocess_csi, hampel_filter, phase_sanitization
import numpy as np

# Load complex CSI data
csi_complex = np.load("raw_csi.npy")  # shape: (3, 114, 2000)

# Full preprocessing
processed = preprocess_csi(
    csi_complex,
    hampel_window=5,
    hampel_threshold=3,
    use_phase=False,  # Use amplitude only
)
```

## Project Structure

```
wavira/
├── wavira/
│   ├── __init__.py
│   ├── data/
│   │   ├── preprocessing.py   # CSI preprocessing (Hampel filter, phase sanitization)
│   │   └── dataset.py         # PyTorch Dataset classes
│   ├── models/
│   │   ├── encoder.py         # Transformer, LSTM, Bi-LSTM encoders
│   │   └── whofi.py           # Main WhoFi model
│   ├── losses/
│   │   └── inbatch_loss.py    # In-batch negative loss, triplet loss
│   ├── training/
│   │   └── trainer.py         # Training loop and utilities
│   └── utils/
│       └── metrics.py         # CMC, mAP evaluation metrics
├── scripts/
│   ├── train.py               # Training script
│   └── evaluate.py            # Evaluation script
├── tests/
│   └── test_model.py          # Unit tests
├── requirements.txt
├── setup.py
└── README.md
```

## Testing

Run unit tests:

```bash
pytest tests/ -v
```

## Crowd Level Estimation

Wavira also supports crowd level estimation using CSI signals. This feature enables non-visual monitoring of room occupancy.

### Crowd Levels

| Level | Status | People |
|-------|--------|--------|
| 0 | Empty | 0-1 |
| 1 | Moderate | 2-5 |
| 2 | Crowded | 6+ |

### Hardware Setup

#### Requirements

- ESP32 with CSI support (ESP32-WROOM, ESP32-S3, etc.)
- USB cable for serial connection
- Wi-Fi access point (router)

#### ESP32 Setup

1. Clone the ESP-CSI repository:
```bash
cd esp-csi/examples/get-started/csi_recv
idf.py set-target esp32  # or esp32s3
idf.py build
idf.py flash monitor
```

2. Configure Wi-Fi credentials in `menuconfig`:
```bash
idf.py menuconfig
# Navigate to: Example Configuration → WiFi SSID/Password
```

3. Place ESP32 on desk with clear line of sight to the monitored area.

### Data Collection

#### Quick Start

```bash
# Collect "empty" level data (0-1 people in room)
python scripts/collect_crowd.py --level 0 --location office

# Collect "moderate" level data (2-5 people)
python scripts/collect_crowd.py --level 1 --location office

# Collect "crowded" level data (6+ people)
python scripts/collect_crowd.py --level 2 --location office
```

#### Collection Parameters

| Option | Default | Description |
|--------|---------|-------------|
| `-l, --level` | (required) | Crowd level (0, 1, 2) |
| `--location` | room1 | Location identifier |
| `-n, --num-files` | 10 | Number of files to collect |
| `-s, --samples` | 100 | CSI packets per file |
| `-p, --port` | /dev/cu.usbserial-* | Serial port |
| `-t, --timeout` | 300 | Timeout in seconds |

#### Data Collection Procedure

1. **Prepare the environment**
   - Set up ESP32 in the target location
   - Ensure stable Wi-Fi connection
   - Verify serial port: `ls /dev/cu.usb*`

2. **Collect Level 0 (Empty)**
   ```bash
   # Clear the room (0-1 people)
   python scripts/collect_crowd.py --level 0 --location office --num-files 20
   ```

3. **Collect Level 1 (Moderate)**
   ```bash
   # Have 2-5 people in the room, moving naturally
   python scripts/collect_crowd.py --level 1 --location office --num-files 20
   ```

4. **Collect Level 2 (Crowded)**
   ```bash
   # Have 6+ people in the room
   python scripts/collect_crowd.py --level 2 --location office --num-files 20
   ```

5. **Tips for quality data**
   - People should move naturally during collection
   - Collect multiple sessions at different times of day
   - Keep ESP32 position consistent
   - Avoid major furniture changes during collection

#### Output Structure

```
data/crowd/
└── office/
    ├── empty/
    │   ├── 20241222_143000_0000.npy   # CSI data
    │   ├── 20241222_143000_0000.json  # Metadata
    │   └── ...
    ├── moderate/
    │   └── ...
    └── crowded/
        └── ...
```

#### Metadata Format

Each `.json` file contains:
```json
{
  "level": 0,
  "level_name": "empty",
  "location": "office",
  "session_id": "20241222_143000",
  "file_index": 0,
  "samples": 100,
  "timestamp": "2024-12-22T14:30:05"
}
```

### Training (Coming Soon)

```bash
# Train crowd level classifier
python scripts/train_crowd.py --data_dir data/crowd --epochs 100
```

### Real-time Inference (Coming Soon)

```bash
# Start real-time crowd monitoring
python scripts/monitor_crowd.py --model checkpoints/crowd_model.pt
```

## NTU-Fi Dataset

The [NTU-Fi dataset](https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark) contains:

- 14 subjects
- 60 samples per subject (3 scenarios)
- CSI format: 3 RX antennas × 114 subcarriers × 2000 packets

## Citation

If you use this work, please cite the original WhoFi paper:

```bibtex
@article{avola2025whofi,
  title={WhoFi: Deep Person Re-Identification via Wi-Fi Channel Signal Encoding},
  author={Avola, Danilo and Pannone, Daniele and Montagnini, Dario and Emam, Emad},
  journal={arXiv preprint arXiv:2507.12869},
  year={2025}
}
```

## License

This project is for research and educational purposes.

## References

- [WhoFi Paper (arXiv)](https://arxiv.org/abs/2507.12869)
- [NTU-Fi Dataset](https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark)
- [Wi-Fi Sensing Survey](https://arxiv.org/abs/1901.00555)

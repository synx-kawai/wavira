---
name: train-model
description: Use this skill when training WhoFi models, adjusting hyperparameters, evaluating model performance, or debugging training issues.
---

# Model Training Skill

This skill provides expertise for training WhoFi models for Wi-Fi CSI person re-identification and crowd estimation.

## Quick Start

```bash
# Train with synthetic data (for testing)
python scripts/train.py --use_synthetic --epochs 50

# Train with real data
python scripts/train.py --data_dir data/person_reid --epochs 300

# Evaluate model
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt
```

## Training Configuration

### Default Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--encoder_type` | transformer | transformer, lstm, bi-lstm |
| `--hidden_dim` | 256 | Encoder hidden dimension |
| `--signature_dim` | 256 | Output embedding dimension |
| `--batch_size` | 8 | Training batch size |
| `--epochs` | 300 | Training epochs |
| `--lr` | 0.0001 | Learning rate |
| `--margin` | 0.3 | Contrastive loss margin |

### Encoder Types

1. **Transformer** (recommended): Best accuracy, slower training
2. **Bi-LSTM**: Good balance of speed and accuracy
3. **LSTM**: Fastest training, lower accuracy

## Data Format

### Person Re-ID Data
```
data/
├── person_001/
│   ├── sample_001.npy  # Shape: (3, 114, 200)
│   ├── sample_002.npy
│   └── ...
├── person_002/
│   └── ...
```

### Crowd Level Data
```
data/crowd/
├── location_name/
│   ├── empty/      # Level 0
│   ├── moderate/   # Level 1
│   └── crowded/    # Level 2
```

## Training Monitoring

### TensorBoard
```bash
tensorboard --logdir runs/
```

### Key Metrics
- **Loss**: In-batch negative loss (should decrease)
- **Rank-1**: Top-1 retrieval accuracy (should increase)
- **mAP**: Mean Average Precision (should increase)

## Common Issues

### Out of Memory
```python
# Reduce batch size
python scripts/train.py --batch_size 4

# Use gradient accumulation
python scripts/train.py --accumulation_steps 2
```

### Slow Convergence
```python
# Increase learning rate
python scripts/train.py --lr 0.0005

# Use learning rate warmup
python scripts/train.py --warmup_epochs 10
```

### Overfitting
```python
# Add dropout
python scripts/train.py --dropout 0.3

# Use data augmentation
python scripts/train.py --augment
```

## Model Architecture

```
CSI Input (3 x 114 x 200)
         │
    Reshape (200, 342)
         │
    Encoder (Transformer/LSTM)
         │
    Mean Pooling (342,)
         │
    Signature Module (256,)
         │
    L2 Normalize
```

## Checkpoints

Checkpoints saved to `checkpoints/`:
- `best_model.pt`: Best validation performance
- `last_model.pt`: Latest checkpoint
- `epoch_N.pt`: Periodic checkpoints

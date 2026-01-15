---
name: csi-data-analyzer
description: Use this skill to analyze CSI data quality, compute statistics, detect anomalies, and validate data before training. Helpful for debugging data collection issues.
---

# CSI Data Analyzer Skill

This skill provides tools for analyzing Wi-Fi CSI data quality and statistics.

## Quick Commands

```bash
# Analyze data directory
python -c "from wavira.utils import analyze_csi_data; analyze_csi_data('data/person_reid')"

# Check single file
python -c "import numpy as np; d=np.load('sample.npy'); print(f'Shape: {d.shape}, Range: [{d.min():.2f}, {d.max():.2f}]')"
```

## Data Quality Checks

### 1. Shape Validation

Expected CSI data shape: `(3, 114, 200)` or `(n_channels, n_subcarriers, n_timestamps)`

```python
import numpy as np
import os

def validate_shape(data_dir, expected_shape=(3, 114, 200)):
    """Validate all .npy files have correct shape."""
    issues = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith('.npy'):
                path = os.path.join(root, f)
                data = np.load(path)
                if data.shape != expected_shape:
                    issues.append(f"{path}: {data.shape}")
    return issues
```

### 2. Value Range Check

CSI amplitude should be in reasonable range (typically 0-100 for normalized data).

```python
def check_value_range(data_dir, min_val=-100, max_val=100):
    """Check for outliers or corrupted data."""
    issues = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith('.npy'):
                path = os.path.join(root, f)
                data = np.load(path)
                if data.min() < min_val or data.max() > max_val:
                    issues.append(f"{path}: range [{data.min():.2f}, {data.max():.2f}]")
    return issues
```

### 3. NaN/Inf Detection

```python
def check_nan_inf(data_dir):
    """Detect NaN or Inf values."""
    issues = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith('.npy'):
                path = os.path.join(root, f)
                data = np.load(path)
                if np.isnan(data).any():
                    issues.append(f"{path}: contains NaN")
                if np.isinf(data).any():
                    issues.append(f"{path}: contains Inf")
    return issues
```

## Statistical Analysis

### Dataset Statistics

```python
def dataset_stats(data_dir):
    """Compute overall dataset statistics."""
    all_data = []
    file_count = 0
    class_counts = {}

    for root, _, files in os.walk(data_dir):
        class_name = os.path.basename(root)
        npy_files = [f for f in files if f.endswith('.npy')]
        if npy_files:
            class_counts[class_name] = len(npy_files)
            file_count += len(npy_files)

    return {
        'total_files': file_count,
        'num_classes': len(class_counts),
        'class_distribution': class_counts,
        'min_samples_per_class': min(class_counts.values()) if class_counts else 0,
        'max_samples_per_class': max(class_counts.values()) if class_counts else 0,
    }
```

### Per-Channel Statistics

```python
def channel_stats(data_path):
    """Analyze per-channel statistics."""
    data = np.load(data_path)
    stats = {}
    for ch in range(data.shape[0]):
        ch_data = data[ch]
        stats[f'channel_{ch}'] = {
            'mean': float(ch_data.mean()),
            'std': float(ch_data.std()),
            'min': float(ch_data.min()),
            'max': float(ch_data.max()),
        }
    return stats
```

## Anomaly Detection

### Subcarrier Null Detection

Some subcarriers may be nulled (guard bands, DC subcarrier).

```python
def detect_null_subcarriers(data_dir, threshold=0.01):
    """Find subcarriers with near-zero variance."""
    sample_files = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith('.npy'):
                sample_files.append(os.path.join(root, f))
                if len(sample_files) >= 10:
                    break

    if not sample_files:
        return []

    # Stack samples
    samples = np.stack([np.load(f) for f in sample_files])

    # Check variance per subcarrier
    null_subcarriers = []
    for sc in range(samples.shape[2]):
        sc_var = samples[:, :, sc, :].var()
        if sc_var < threshold:
            null_subcarriers.append(sc)

    return null_subcarriers
```

## Data Collection Quality

### Temporal Consistency

```python
def check_temporal_consistency(data_path):
    """Check for sudden jumps or discontinuities."""
    data = np.load(data_path)

    # Compute temporal difference
    diff = np.abs(np.diff(data, axis=-1))

    # Find large jumps
    threshold = diff.mean() + 3 * diff.std()
    jumps = np.where(diff > threshold)

    return {
        'mean_diff': float(diff.mean()),
        'max_diff': float(diff.max()),
        'num_large_jumps': len(jumps[0]),
    }
```

## Recommended Workflow

1. **Before Training**:
   - Run shape validation
   - Check for NaN/Inf
   - Verify class balance
   - Compute dataset statistics

2. **Debugging Poor Performance**:
   - Analyze channel statistics
   - Detect null subcarriers
   - Check temporal consistency
   - Compare train/test distributions

3. **After Data Collection**:
   - Validate all new files
   - Check value ranges
   - Ensure sufficient samples per class

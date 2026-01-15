"""
CSI Data Validation Framework

Provides comprehensive validation for WiFi Channel State Information data:
- Input data format validation
- Anomaly detection
- Data quality metrics
- Duplicate detection
- Missing data handling
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Union, Dict, Any
from enum import Enum
import hashlib
import json


class ValidationLevel(Enum):
    """Validation severity levels."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """Represents a single validation issue."""
    level: ValidationLevel
    code: str
    message: str
    location: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class ValidationResult:
    """Result of validation operation."""
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def add_issue(self, level: ValidationLevel, code: str, message: str,
                  location: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """Add a validation issue."""
        self.issues.append(ValidationIssue(level, code, message, location, details))
        if level == ValidationLevel.ERROR:
            self.is_valid = False

    def has_errors(self) -> bool:
        """Check if there are any error-level issues."""
        return any(issue.level == ValidationLevel.ERROR for issue in self.issues)

    def has_warnings(self) -> bool:
        """Check if there are any warning-level issues."""
        return any(issue.level == ValidationLevel.WARNING for issue in self.issues)

    def error_count(self) -> int:
        """Count error-level issues."""
        return sum(1 for issue in self.issues if issue.level == ValidationLevel.ERROR)

    def warning_count(self) -> int:
        """Count warning-level issues."""
        return sum(1 for issue in self.issues if issue.level == ValidationLevel.WARNING)


@dataclass
class CSIDataConfig:
    """Configuration for CSI data validation."""
    expected_subcarriers: int = 52
    expected_rx_antennas: int = 1
    expected_tx_antennas: int = 1
    min_packets: int = 10
    max_packets: int = 10000
    amplitude_min: float = 0.0
    amplitude_max: float = 1000.0
    rssi_min: float = -100.0
    rssi_max: float = 0.0
    nan_threshold: float = 0.01  # Max 1% NaN values
    inf_threshold: float = 0.001  # Max 0.1% Inf values
    zero_threshold: float = 0.1  # Max 10% zero values
    outlier_std_threshold: float = 5.0  # Values beyond 5 std are outliers


class CSIFormatValidator:
    """Validates CSI data format and structure."""

    def __init__(self, config: Optional[CSIDataConfig] = None):
        self.config = config or CSIDataConfig()

    def validate_shape(self, data: np.ndarray, expected_dims: int = 3) -> ValidationResult:
        """
        Validate array shape and dimensions.

        Args:
            data: CSI data array
            expected_dims: Expected number of dimensions

        Returns:
            ValidationResult with shape validation details
        """
        result = ValidationResult(is_valid=True)
        result.metrics["shape"] = data.shape
        result.metrics["ndim"] = data.ndim
        result.metrics["size"] = data.size

        if data.ndim != expected_dims:
            result.add_issue(
                ValidationLevel.ERROR,
                "INVALID_DIMENSIONS",
                f"Expected {expected_dims} dimensions, got {data.ndim}",
                details={"expected": expected_dims, "actual": data.ndim}
            )

        if data.size == 0:
            result.add_issue(
                ValidationLevel.ERROR,
                "EMPTY_DATA",
                "Data array is empty"
            )

        return result

    def validate_subcarriers(self, data: np.ndarray, subcarrier_axis: int = 1) -> ValidationResult:
        """
        Validate subcarrier count.

        Args:
            data: CSI data array
            subcarrier_axis: Axis containing subcarrier dimension

        Returns:
            ValidationResult with subcarrier validation details
        """
        result = ValidationResult(is_valid=True)

        if data.ndim <= subcarrier_axis:
            result.add_issue(
                ValidationLevel.ERROR,
                "INVALID_AXIS",
                f"Data has {data.ndim} dimensions, cannot access axis {subcarrier_axis}"
            )
            return result

        n_subcarriers = data.shape[subcarrier_axis]
        result.metrics["n_subcarriers"] = n_subcarriers

        if n_subcarriers != self.config.expected_subcarriers:
            result.add_issue(
                ValidationLevel.WARNING,
                "UNEXPECTED_SUBCARRIER_COUNT",
                f"Expected {self.config.expected_subcarriers} subcarriers, got {n_subcarriers}",
                details={"expected": self.config.expected_subcarriers, "actual": n_subcarriers}
            )

        return result

    def validate_packets(self, data: np.ndarray, packet_axis: int = -1) -> ValidationResult:
        """
        Validate packet count.

        Args:
            data: CSI data array
            packet_axis: Axis containing packet dimension

        Returns:
            ValidationResult with packet validation details
        """
        result = ValidationResult(is_valid=True)
        n_packets = data.shape[packet_axis]
        result.metrics["n_packets"] = n_packets

        if n_packets < self.config.min_packets:
            result.add_issue(
                ValidationLevel.ERROR,
                "TOO_FEW_PACKETS",
                f"Minimum {self.config.min_packets} packets required, got {n_packets}",
                details={"min": self.config.min_packets, "actual": n_packets}
            )

        if n_packets > self.config.max_packets:
            result.add_issue(
                ValidationLevel.WARNING,
                "TOO_MANY_PACKETS",
                f"Data has {n_packets} packets, exceeds recommended max {self.config.max_packets}",
                details={"max": self.config.max_packets, "actual": n_packets}
            )

        return result

    def validate_format(self, data: np.ndarray) -> ValidationResult:
        """
        Perform full format validation.

        Args:
            data: CSI data array

        Returns:
            Combined ValidationResult
        """
        result = ValidationResult(is_valid=True)

        # Check data type
        result.metrics["dtype"] = str(data.dtype)
        if not np.issubdtype(data.dtype, np.number):
            result.add_issue(
                ValidationLevel.ERROR,
                "INVALID_DTYPE",
                f"Expected numeric dtype, got {data.dtype}"
            )
            return result

        # Combine individual validations
        shape_result = self.validate_shape(data)
        result.issues.extend(shape_result.issues)
        result.metrics.update(shape_result.metrics)

        if shape_result.is_valid and data.ndim >= 2:
            sub_result = self.validate_subcarriers(data)
            result.issues.extend(sub_result.issues)
            result.metrics.update(sub_result.metrics)

            pkt_result = self.validate_packets(data)
            result.issues.extend(pkt_result.issues)
            result.metrics.update(pkt_result.metrics)

        result.is_valid = not result.has_errors()
        return result


class CSIAnomalyDetector:
    """Detects anomalies in CSI data."""

    def __init__(self, config: Optional[CSIDataConfig] = None):
        self.config = config or CSIDataConfig()

    def detect_nan_inf(self, data: np.ndarray) -> ValidationResult:
        """
        Detect NaN and Inf values.

        Args:
            data: CSI data array

        Returns:
            ValidationResult with NaN/Inf detection details
        """
        result = ValidationResult(is_valid=True)
        total_elements = data.size

        nan_count = np.sum(np.isnan(data))
        inf_count = np.sum(np.isinf(data))
        nan_ratio = nan_count / total_elements if total_elements > 0 else 0
        inf_ratio = inf_count / total_elements if total_elements > 0 else 0

        result.metrics["nan_count"] = int(nan_count)
        result.metrics["nan_ratio"] = float(nan_ratio)
        result.metrics["inf_count"] = int(inf_count)
        result.metrics["inf_ratio"] = float(inf_ratio)

        if nan_ratio > self.config.nan_threshold:
            result.add_issue(
                ValidationLevel.ERROR,
                "EXCESSIVE_NAN",
                f"NaN ratio {nan_ratio:.2%} exceeds threshold {self.config.nan_threshold:.2%}",
                details={"count": nan_count, "ratio": nan_ratio}
            )

        if inf_ratio > self.config.inf_threshold:
            result.add_issue(
                ValidationLevel.ERROR,
                "EXCESSIVE_INF",
                f"Inf ratio {inf_ratio:.2%} exceeds threshold {self.config.inf_threshold:.2%}",
                details={"count": inf_count, "ratio": inf_ratio}
            )

        return result

    def detect_zero_values(self, data: np.ndarray) -> ValidationResult:
        """
        Detect excessive zero values.

        Args:
            data: CSI data array

        Returns:
            ValidationResult with zero value detection details
        """
        result = ValidationResult(is_valid=True)
        total_elements = data.size

        zero_count = np.sum(data == 0)
        zero_ratio = zero_count / total_elements if total_elements > 0 else 0

        result.metrics["zero_count"] = int(zero_count)
        result.metrics["zero_ratio"] = float(zero_ratio)

        if zero_ratio > self.config.zero_threshold:
            result.add_issue(
                ValidationLevel.WARNING,
                "EXCESSIVE_ZEROS",
                f"Zero ratio {zero_ratio:.2%} exceeds threshold {self.config.zero_threshold:.2%}",
                details={"count": zero_count, "ratio": zero_ratio}
            )

        return result

    def detect_outliers(self, data: np.ndarray) -> ValidationResult:
        """
        Detect statistical outliers using z-score method.

        Args:
            data: CSI data array

        Returns:
            ValidationResult with outlier detection details
        """
        result = ValidationResult(is_valid=True)

        # Filter out NaN/Inf for statistics
        valid_data = data[np.isfinite(data)]
        if len(valid_data) == 0:
            result.add_issue(
                ValidationLevel.ERROR,
                "NO_VALID_DATA",
                "No valid (finite) data for outlier detection"
            )
            return result

        mean = np.mean(valid_data)
        std = np.std(valid_data)

        result.metrics["mean"] = float(mean)
        result.metrics["std"] = float(std)
        result.metrics["min"] = float(np.min(valid_data))
        result.metrics["max"] = float(np.max(valid_data))

        if std > 0:
            z_scores = np.abs((valid_data - mean) / std)
            outlier_count = np.sum(z_scores > self.config.outlier_std_threshold)
            outlier_ratio = outlier_count / len(valid_data)

            result.metrics["outlier_count"] = int(outlier_count)
            result.metrics["outlier_ratio"] = float(outlier_ratio)

            if outlier_ratio > 0.05:  # More than 5% outliers
                result.add_issue(
                    ValidationLevel.WARNING,
                    "HIGH_OUTLIER_RATIO",
                    f"Outlier ratio {outlier_ratio:.2%} is high (> 5%)",
                    details={"count": outlier_count, "ratio": outlier_ratio}
                )
        else:
            result.metrics["outlier_count"] = 0
            result.metrics["outlier_ratio"] = 0.0

        return result

    def detect_amplitude_range(self, amplitudes: np.ndarray) -> ValidationResult:
        """
        Validate amplitude values are within expected range.

        Args:
            amplitudes: Amplitude data array

        Returns:
            ValidationResult with range validation details
        """
        result = ValidationResult(is_valid=True)

        # Filter out NaN/Inf
        valid_amps = amplitudes[np.isfinite(amplitudes)]
        if len(valid_amps) == 0:
            return result

        min_amp = np.min(valid_amps)
        max_amp = np.max(valid_amps)

        result.metrics["amplitude_min"] = float(min_amp)
        result.metrics["amplitude_max"] = float(max_amp)

        if min_amp < self.config.amplitude_min:
            below_count = np.sum(valid_amps < self.config.amplitude_min)
            result.add_issue(
                ValidationLevel.WARNING,
                "AMPLITUDE_BELOW_RANGE",
                f"Found {below_count} amplitude values below {self.config.amplitude_min}",
                details={"count": below_count, "min_value": min_amp}
            )

        if max_amp > self.config.amplitude_max:
            above_count = np.sum(valid_amps > self.config.amplitude_max)
            result.add_issue(
                ValidationLevel.WARNING,
                "AMPLITUDE_ABOVE_RANGE",
                f"Found {above_count} amplitude values above {self.config.amplitude_max}",
                details={"count": above_count, "max_value": max_amp}
            )

        return result

    def detect_all_anomalies(self, data: np.ndarray) -> ValidationResult:
        """
        Run all anomaly detection methods.

        Args:
            data: CSI data array

        Returns:
            Combined ValidationResult
        """
        result = ValidationResult(is_valid=True)

        nan_inf = self.detect_nan_inf(data)
        result.issues.extend(nan_inf.issues)
        result.metrics.update(nan_inf.metrics)

        zeros = self.detect_zero_values(data)
        result.issues.extend(zeros.issues)
        result.metrics.update(zeros.metrics)

        outliers = self.detect_outliers(data)
        result.issues.extend(outliers.issues)
        result.metrics.update(outliers.metrics)

        amplitudes = self.detect_amplitude_range(data)
        result.issues.extend(amplitudes.issues)
        result.metrics.update(amplitudes.metrics)

        result.is_valid = not result.has_errors()
        return result


class CSIQualityMetrics:
    """Calculate quality metrics for CSI data."""

    @staticmethod
    def signal_to_noise_ratio(data: np.ndarray, axis: int = -1) -> float:
        """
        Estimate signal-to-noise ratio.

        Uses variance-based estimation: SNR = var(signal) / var(noise)
        Noise is estimated from high-frequency components.

        Args:
            data: CSI data array
            axis: Axis along which to compute SNR

        Returns:
            Estimated SNR in dB
        """
        if data.size == 0:
            return 0.0

        # Filter out invalid values
        valid_data = np.where(np.isfinite(data), data, np.nan)

        # Signal power (total variance)
        signal_var = np.nanvar(valid_data)

        # Noise estimation using high-frequency differences
        diff = np.diff(valid_data, axis=axis)
        noise_var = np.nanvar(diff) / 2  # Divide by 2 for differentiation

        if noise_var < 1e-10:
            return float('inf')

        snr = signal_var / noise_var
        snr_db = 10 * np.log10(snr) if snr > 0 else 0.0

        return float(snr_db)

    @staticmethod
    def temporal_consistency(data: np.ndarray, axis: int = -1) -> float:
        """
        Measure temporal consistency (smoothness) of the signal.

        Lower values indicate smoother, more consistent signals.

        Args:
            data: CSI data array
            axis: Time axis

        Returns:
            Temporal consistency score (0-1, lower is better)
        """
        if data.size == 0:
            return 1.0

        valid_data = np.where(np.isfinite(data), data, np.nan)

        # First difference
        diff1 = np.diff(valid_data, axis=axis)
        # Second difference (acceleration)
        diff2 = np.diff(diff1, axis=axis)

        # Normalized second derivative variance
        data_std = np.nanstd(valid_data)
        if data_std < 1e-10:
            return 0.0

        consistency = np.nanstd(diff2) / data_std

        # Normalize to 0-1 range
        return float(min(1.0, consistency / 2))

    @staticmethod
    def subcarrier_correlation(data: np.ndarray, subcarrier_axis: int = 1) -> float:
        """
        Measure average correlation between subcarriers.

        High correlation indicates coherent signal across subcarriers.

        Args:
            data: CSI data array (antennas, subcarriers, packets)
            subcarrier_axis: Axis containing subcarriers

        Returns:
            Average correlation coefficient (0-1)
        """
        if data.ndim < 2:
            return 0.0

        # Reshape to (subcarriers, -1)
        data = np.moveaxis(data, subcarrier_axis, 0)
        n_subcarriers = data.shape[0]
        flat_data = data.reshape(n_subcarriers, -1)

        # Filter out rows with all NaN
        valid_rows = ~np.all(np.isnan(flat_data), axis=1)
        flat_data = flat_data[valid_rows]

        if flat_data.shape[0] < 2:
            return 0.0

        # Compute correlation matrix
        try:
            # Replace NaN with 0 for correlation computation
            clean_data = np.nan_to_num(flat_data, nan=0.0)
            corr_matrix = np.corrcoef(clean_data)

            # Average off-diagonal correlations
            mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
            avg_corr = np.nanmean(np.abs(corr_matrix[mask]))

            return float(avg_corr) if np.isfinite(avg_corr) else 0.0
        except Exception:
            return 0.0

    def compute_all_metrics(self, data: np.ndarray) -> Dict[str, float]:
        """
        Compute all quality metrics.

        Args:
            data: CSI data array

        Returns:
            Dictionary of metric names to values
        """
        return {
            "snr_db": self.signal_to_noise_ratio(data),
            "temporal_consistency": self.temporal_consistency(data),
            "subcarrier_correlation": self.subcarrier_correlation(data),
        }


class CSIDuplicateDetector:
    """Detect duplicate samples in CSI datasets."""

    def __init__(self, tolerance: float = 1e-6):
        """
        Initialize detector.

        Args:
            tolerance: Numerical tolerance for considering values equal
        """
        self.tolerance = tolerance

    def compute_hash(self, data: np.ndarray) -> str:
        """
        Compute a hash for a data sample.

        Args:
            data: Sample data array

        Returns:
            Hash string
        """
        # Round to tolerance and convert to bytes
        rounded = np.round(data / self.tolerance) * self.tolerance
        data_bytes = rounded.tobytes()
        return hashlib.md5(data_bytes).hexdigest()

    def find_duplicates(self, samples: List[np.ndarray]) -> List[Tuple[int, int]]:
        """
        Find duplicate pairs in a list of samples.

        Args:
            samples: List of sample arrays

        Returns:
            List of (index1, index2) tuples for duplicate pairs
        """
        hashes: Dict[str, int] = {}
        duplicates: List[Tuple[int, int]] = []

        for i, sample in enumerate(samples):
            h = self.compute_hash(sample)
            if h in hashes:
                duplicates.append((hashes[h], i))
            else:
                hashes[h] = i

        return duplicates

    def find_near_duplicates(self, samples: List[np.ndarray],
                             similarity_threshold: float = 0.99) -> List[Tuple[int, int, float]]:
        """
        Find near-duplicate pairs based on correlation.

        Args:
            samples: List of sample arrays
            similarity_threshold: Minimum correlation to consider near-duplicate

        Returns:
            List of (index1, index2, similarity) tuples
        """
        n = len(samples)
        if n < 2:
            return []

        near_duplicates: List[Tuple[int, int, float]] = []

        # Flatten samples for correlation
        flat_samples = [s.flatten() for s in samples]

        for i in range(n):
            for j in range(i + 1, n):
                try:
                    corr = np.corrcoef(flat_samples[i], flat_samples[j])[0, 1]
                    if np.isfinite(corr) and corr >= similarity_threshold:
                        near_duplicates.append((i, j, float(corr)))
                except Exception:
                    continue

        return near_duplicates

    def detect_duplicates(self, samples: List[np.ndarray]) -> ValidationResult:
        """
        Detect both exact and near duplicates.

        Args:
            samples: List of sample arrays

        Returns:
            ValidationResult with duplicate detection details
        """
        result = ValidationResult(is_valid=True)
        result.metrics["total_samples"] = len(samples)

        if len(samples) < 2:
            return result

        # Exact duplicates
        exact_dups = self.find_duplicates(samples)
        result.metrics["exact_duplicates"] = len(exact_dups)

        if exact_dups:
            result.add_issue(
                ValidationLevel.WARNING,
                "EXACT_DUPLICATES_FOUND",
                f"Found {len(exact_dups)} exact duplicate pairs",
                details={"pairs": exact_dups[:10]}  # Limit to first 10
            )

        # Near duplicates (only for small datasets to avoid O(n^2) cost)
        if len(samples) <= 1000:
            near_dups = self.find_near_duplicates(samples)
            result.metrics["near_duplicates"] = len(near_dups)

            if near_dups:
                result.add_issue(
                    ValidationLevel.INFO,
                    "NEAR_DUPLICATES_FOUND",
                    f"Found {len(near_dups)} near-duplicate pairs (similarity >= 0.99)",
                    details={"pairs": near_dups[:10]}
                )
        else:
            result.metrics["near_duplicates"] = "skipped (dataset too large)"

        return result


class MissingDataHandler:
    """Handle missing data in CSI samples."""

    @staticmethod
    def detect_missing(data: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Detect missing values (NaN) in data.

        Args:
            data: CSI data array

        Returns:
            Tuple of (boolean mask of missing values, missing ratio)
        """
        missing_mask = np.isnan(data)
        missing_ratio = np.sum(missing_mask) / data.size if data.size > 0 else 0
        return missing_mask, float(missing_ratio)

    @staticmethod
    def interpolate_missing(data: np.ndarray, axis: int = -1,
                           method: str = "linear") -> np.ndarray:
        """
        Interpolate missing values along an axis.

        Args:
            data: CSI data array with NaN values
            axis: Axis along which to interpolate
            method: Interpolation method ("linear", "nearest", "zero")

        Returns:
            Data with missing values interpolated
        """
        data = data.copy()

        # Move target axis to last position
        data = np.moveaxis(data, axis, -1)
        original_shape = data.shape
        flat_shape = (-1, data.shape[-1])
        data = data.reshape(flat_shape)

        for i in range(data.shape[0]):
            row = data[i]
            mask = np.isnan(row)

            if not np.any(mask):
                continue

            if np.all(mask):
                # All missing, fill with zeros
                data[i] = 0
                continue

            # Get valid indices and values
            valid_indices = np.where(~mask)[0]
            valid_values = row[~mask]

            # Interpolate
            missing_indices = np.where(mask)[0]

            if method == "nearest":
                # Nearest neighbor
                for idx in missing_indices:
                    closest = valid_indices[np.argmin(np.abs(valid_indices - idx))]
                    data[i, idx] = row[closest]
            elif method == "zero":
                data[i, mask] = 0
            else:
                # Linear interpolation
                data[i, mask] = np.interp(missing_indices, valid_indices, valid_values)

        # Restore shape
        data = data.reshape(original_shape)
        data = np.moveaxis(data, -1, axis)

        return data

    @staticmethod
    def drop_missing_samples(samples: List[np.ndarray],
                            threshold: float = 0.1) -> Tuple[List[np.ndarray], List[int]]:
        """
        Drop samples with too many missing values.

        Args:
            samples: List of sample arrays
            threshold: Maximum allowed missing ratio per sample

        Returns:
            Tuple of (filtered samples, indices of kept samples)
        """
        kept_samples = []
        kept_indices = []

        for i, sample in enumerate(samples):
            _, missing_ratio = MissingDataHandler.detect_missing(sample)
            if missing_ratio <= threshold:
                kept_samples.append(sample)
                kept_indices.append(i)

        return kept_samples, kept_indices


class CSIDataValidator:
    """
    Main validator class combining all validation components.

    Usage:
        validator = CSIDataValidator()
        result = validator.validate(csi_data)
        print(result.is_valid)
        print(result.metrics)
    """

    def __init__(self, config: Optional[CSIDataConfig] = None):
        self.config = config or CSIDataConfig()
        self.format_validator = CSIFormatValidator(self.config)
        self.anomaly_detector = CSIAnomalyDetector(self.config)
        self.quality_metrics = CSIQualityMetrics()
        self.duplicate_detector = CSIDuplicateDetector()
        self.missing_handler = MissingDataHandler()

    def validate(self, data: np.ndarray,
                check_format: bool = True,
                check_anomalies: bool = True,
                check_quality: bool = True) -> ValidationResult:
        """
        Perform comprehensive validation on CSI data.

        Args:
            data: CSI data array
            check_format: Whether to validate format
            check_anomalies: Whether to detect anomalies
            check_quality: Whether to compute quality metrics

        Returns:
            Combined ValidationResult
        """
        result = ValidationResult(is_valid=True)

        if check_format:
            fmt_result = self.format_validator.validate_format(data)
            result.issues.extend(fmt_result.issues)
            result.metrics["format"] = fmt_result.metrics

            if not fmt_result.is_valid:
                result.is_valid = False
                return result

        if check_anomalies:
            anomaly_result = self.anomaly_detector.detect_all_anomalies(data)
            result.issues.extend(anomaly_result.issues)
            result.metrics["anomalies"] = anomaly_result.metrics

        if check_quality:
            quality = self.quality_metrics.compute_all_metrics(data)
            result.metrics["quality"] = quality

        # Check missing data
        missing_mask, missing_ratio = self.missing_handler.detect_missing(data)
        result.metrics["missing_ratio"] = missing_ratio

        if missing_ratio > self.config.nan_threshold:
            result.add_issue(
                ValidationLevel.WARNING,
                "HIGH_MISSING_RATIO",
                f"Missing data ratio {missing_ratio:.2%} is high"
            )

        result.is_valid = not result.has_errors()
        return result

    def validate_dataset(self, samples: List[np.ndarray],
                        check_duplicates: bool = True) -> ValidationResult:
        """
        Validate a dataset of multiple samples.

        Args:
            samples: List of sample arrays
            check_duplicates: Whether to check for duplicates

        Returns:
            ValidationResult for the dataset
        """
        result = ValidationResult(is_valid=True)
        result.metrics["n_samples"] = len(samples)

        if not samples:
            result.add_issue(
                ValidationLevel.ERROR,
                "EMPTY_DATASET",
                "Dataset contains no samples"
            )
            return result

        # Validate each sample
        sample_results = []
        for i, sample in enumerate(samples):
            sample_result = self.validate(sample)
            if not sample_result.is_valid:
                sample_results.append((i, sample_result))

        result.metrics["invalid_samples"] = len(sample_results)

        if sample_results:
            result.add_issue(
                ValidationLevel.WARNING,
                "INVALID_SAMPLES",
                f"{len(sample_results)} samples failed validation",
                details={"indices": [i for i, _ in sample_results[:10]]}
            )

        # Check for duplicates
        if check_duplicates:
            dup_result = self.duplicate_detector.detect_duplicates(samples)
            result.issues.extend(dup_result.issues)
            result.metrics["duplicates"] = {
                "exact": dup_result.metrics.get("exact_duplicates", 0),
                "near": dup_result.metrics.get("near_duplicates", 0),
            }

        result.is_valid = not result.has_errors()
        return result

    def generate_report(self, result: ValidationResult) -> str:
        """
        Generate a human-readable report from validation result.

        Args:
            result: ValidationResult to report

        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 60)
        lines.append("CSI Data Validation Report")
        lines.append("=" * 60)
        lines.append("")

        # Summary
        status = "PASSED" if result.is_valid else "FAILED"
        lines.append(f"Status: {status}")
        lines.append(f"Errors: {result.error_count()}")
        lines.append(f"Warnings: {result.warning_count()}")
        lines.append("")

        # Issues
        if result.issues:
            lines.append("-" * 40)
            lines.append("Issues:")
            lines.append("-" * 40)
            for issue in result.issues:
                prefix = f"[{issue.level.value.upper()}]"
                lines.append(f"  {prefix} {issue.code}: {issue.message}")
            lines.append("")

        # Metrics
        if result.metrics:
            lines.append("-" * 40)
            lines.append("Metrics:")
            lines.append("-" * 40)
            lines.append(json.dumps(result.metrics, indent=2, default=str))

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

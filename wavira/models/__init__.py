"""Neural network models for Wi-Fi CSI-based analysis."""

from wavira.models.encoder import (
    TransformerEncoder,
    LSTMEncoder,
    BiLSTMEncoder,
    get_encoder,
)
from wavira.models.whofi import WhoFi
from wavira.models.crowd_estimator import (
    CrowdEstimator,
    CrowdEstimatorConfig,
    CrowdEstimatorLight,
    create_model,
)
from wavira.models.gesture_recognizer import (
    GestureRecognizer3DCNN,
    GestureRecognizerLite,
    DualESP32GestureRecognizer,
    GestureRecognizerConfig,
    create_gesture_model,
    DEFAULT_GESTURE_LABELS,
)

__all__ = [
    "TransformerEncoder",
    "LSTMEncoder",
    "BiLSTMEncoder",
    "get_encoder",
    "WhoFi",
    "CrowdEstimator",
    "CrowdEstimatorConfig",
    "CrowdEstimatorLight",
    "create_model",
    # Gesture recognition
    "GestureRecognizer3DCNN",
    "GestureRecognizerLite",
    "DualESP32GestureRecognizer",
    "GestureRecognizerConfig",
    "create_gesture_model",
    "DEFAULT_GESTURE_LABELS",
]

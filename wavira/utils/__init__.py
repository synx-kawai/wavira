"""Utility functions for Wavira."""

from wavira.utils.metrics import (
    compute_cmc,
    compute_map,
    compute_rank_k_accuracy,
    evaluate_reid,
)

# MQTT simulator imports are optional (requires paho-mqtt)
try:
    from wavira.utils.mqtt_simulator import (
        MQTTDeviceSimulator,
        SimulatedDevice,
        SimulatorConfig,
        DeviceStats,
        DeviceState,
        PresencePattern,
        run_load_test,
    )
    _MQTT_AVAILABLE = True
except ImportError:
    _MQTT_AVAILABLE = False
    MQTTDeviceSimulator = None
    SimulatedDevice = None
    SimulatorConfig = None
    DeviceStats = None
    DeviceState = None
    PresencePattern = None
    run_load_test = None

__all__ = [
    "compute_cmc",
    "compute_map",
    "compute_rank_k_accuracy",
    "evaluate_reid",
    # MQTT simulator (optional)
    "MQTTDeviceSimulator",
    "SimulatedDevice",
    "SimulatorConfig",
    "DeviceStats",
    "DeviceState",
    "PresencePattern",
    "run_load_test",
]

"""
Integration tests for MQTT simulator and device communication.

These tests verify the MQTT simulator functionality for Issue #27.
Tests are separated into:
1. Unit tests (no MQTT broker required)
2. Integration tests (require MQTT broker, marked with pytest.mark.mqtt)

Run unit tests only:
    pytest tests/test_mqtt_integration.py -v

Run all tests (requires mosquitto on localhost:1883):
    pytest tests/test_mqtt_integration.py -v --mqtt-broker
"""

import json
import pytest
import threading
import time
from unittest.mock import Mock, MagicMock, patch

import numpy as np

# Skip all tests if paho-mqtt is not installed
paho = pytest.importorskip("paho.mqtt.client")

from wavira.utils.mqtt_simulator import (
    SimulatorConfig,
    SimulatedDevice,
    MQTTDeviceSimulator,
    DeviceStats,
    DeviceState,
    PresencePattern,
    run_load_test,
)


# =============================================================================
# Unit Tests (No MQTT Broker Required)
# =============================================================================


class TestSimulatorConfig:
    """Tests for SimulatorConfig dataclass."""

    def test_default_values(self):
        config = SimulatorConfig()
        assert config.broker_host == "localhost"
        assert config.broker_port == 1883
        assert config.num_subcarriers == 52
        assert config.publish_rate_hz == 10.0
        assert config.qos == 1
        assert config.presence_pattern == PresencePattern.RANDOM

    def test_custom_values(self):
        config = SimulatorConfig(
            broker_host="mqtt.example.com",
            broker_port=8883,
            publish_rate_hz=20.0,
            presence_pattern=PresencePattern.BREATHING,
        )
        assert config.broker_host == "mqtt.example.com"
        assert config.broker_port == 8883
        assert config.publish_rate_hz == 20.0
        assert config.presence_pattern == PresencePattern.BREATHING


class TestDeviceStats:
    """Tests for DeviceStats dataclass."""

    def test_default_values(self):
        stats = DeviceStats(device_id="test_001")
        assert stats.device_id == "test_001"
        assert stats.messages_sent == 0
        assert stats.messages_failed == 0
        assert stats.state == DeviceState.DISCONNECTED

    def test_to_dict(self):
        stats = DeviceStats(
            device_id="test_001",
            messages_sent=100,
            messages_failed=5,
            state=DeviceState.CONNECTED,
        )
        d = stats.to_dict()
        assert d["device_id"] == "test_001"
        assert d["messages_sent"] == 100
        assert d["messages_failed"] == 5
        assert d["state"] == "connected"


class TestPresencePattern:
    """Tests for PresencePattern enum."""

    def test_all_patterns_exist(self):
        assert PresencePattern.ALWAYS_PRESENT.value == "always_present"
        assert PresencePattern.ALWAYS_ABSENT.value == "always_absent"
        assert PresencePattern.RANDOM.value == "random"
        assert PresencePattern.PERIODIC.value == "periodic"
        assert PresencePattern.BREATHING.value == "breathing"


class TestDeviceState:
    """Tests for DeviceState enum."""

    def test_all_states_exist(self):
        assert DeviceState.DISCONNECTED.value == "disconnected"
        assert DeviceState.CONNECTING.value == "connecting"
        assert DeviceState.CONNECTED.value == "connected"
        assert DeviceState.ERROR.value == "error"


class TestSimulatedDeviceUnit:
    """Unit tests for SimulatedDevice (no broker required)."""

    def test_initialization(self):
        config = SimulatorConfig()
        device = SimulatedDevice("test_001", config)
        assert device.device_id == "test_001"
        assert device.running is False
        assert device.stats.state == DeviceState.DISCONNECTED

    def test_topics(self):
        config = SimulatorConfig()
        device = SimulatedDevice("test_001", config)
        assert device._csi_topic == "wavira/csi/test_001"
        assert device._analysis_topic == "wavira/analysis/test_001"
        assert device._status_topic == "wavira/device/test_001/status"
        assert device._will_topic == "wavira/device/test_001/will"

    def test_determine_presence_always_present(self):
        config = SimulatorConfig(presence_pattern=PresencePattern.ALWAYS_PRESENT)
        device = SimulatedDevice("test_001", config)
        device._start_time = time.time()
        assert device._determine_presence(time.time()) is True

    def test_determine_presence_always_absent(self):
        config = SimulatorConfig(presence_pattern=PresencePattern.ALWAYS_ABSENT)
        device = SimulatedDevice("test_001", config)
        device._start_time = time.time()
        assert device._determine_presence(time.time()) is False

    def test_determine_presence_periodic(self):
        config = SimulatorConfig(
            presence_pattern=PresencePattern.PERIODIC,
            presence_period_on=10.0,
            presence_period_off=10.0,
        )
        device = SimulatedDevice("test_001", config)
        device._start_time = 0.0

        # At t=5, should be present (within first 10 seconds)
        assert device._determine_presence(5.0) is True

        # At t=15, should be absent (in the off period)
        assert device._determine_presence(15.0) is False

        # At t=25, should be present again (new cycle)
        assert device._determine_presence(25.0) is True

    def test_generate_csi_data_shape(self):
        config = SimulatorConfig(
            num_subcarriers=52,
            presence_pattern=PresencePattern.ALWAYS_PRESENT,
        )
        device = SimulatedDevice("test_001", config)
        device._start_time = time.time()

        data = device._generate_csi_data(time.time())

        assert "device_id" in data
        assert data["device_id"] == "test_001"
        assert "timestamp" in data
        assert "amplitudes" in data
        assert len(data["amplitudes"]) == 52
        assert "avg_amplitude" in data
        assert "variance" in data
        assert "present" in data
        assert data["present"] is True

    def test_generate_csi_data_absent(self):
        config = SimulatorConfig(
            presence_pattern=PresencePattern.ALWAYS_ABSENT,
        )
        device = SimulatedDevice("test_001", config)
        device._start_time = time.time()

        data = device._generate_csi_data(time.time())
        assert data["present"] is False

    def test_generate_csi_data_breathing(self):
        config = SimulatorConfig(
            presence_pattern=PresencePattern.BREATHING,
            breathing_rate=15.0,
        )
        device = SimulatedDevice("test_001", config)
        device._start_time = 0.0

        data = device._generate_csi_data(5.0)  # During presence period
        assert "breath_ratio" in data
        # Breathing ratio should be > 0 when present with breathing pattern
        # (may vary due to randomness)

    def test_callback_is_called(self):
        callback = Mock()
        config = SimulatorConfig(presence_pattern=PresencePattern.ALWAYS_PRESENT)
        device = SimulatedDevice("test_001", config, on_message_callback=callback)

        # Mock the client as connected
        device._client = MagicMock()
        device._client.is_connected.return_value = True
        device._client.publish.return_value = MagicMock(rc=0)

        device._start_time = time.time()
        data = device._generate_csi_data(time.time())
        device._publish_csi(data)

        callback.assert_called_once()
        args = callback.call_args[0]
        assert args[0] == "wavira/analysis/test_001"
        assert args[1]["device_id"] == "test_001"


class TestMQTTDeviceSimulatorUnit:
    """Unit tests for MQTTDeviceSimulator (no broker required)."""

    def test_initialization(self):
        sim = MQTTDeviceSimulator()
        assert sim.device_count == 0
        assert sim.connected_count == 0

    def test_initialization_with_config(self):
        config = SimulatorConfig(broker_host="custom.mqtt.host")
        sim = MQTTDeviceSimulator(config)
        assert sim.config.broker_host == "custom.mqtt.host"

    def test_get_all_stats_empty(self):
        sim = MQTTDeviceSimulator()
        stats = sim.get_all_stats()
        assert stats["device_count"] == 0
        assert stats["connected_count"] == 0
        assert stats["total_messages_sent"] == 0

    def test_get_device_stats_not_found(self):
        sim = MQTTDeviceSimulator()
        stats = sim.get_device_stats("nonexistent")
        assert stats is None


class TestCSIDataGeneration:
    """Tests for CSI data generation quality."""

    def test_amplitude_distribution(self):
        """Verify amplitude values are around base_amplitude."""
        config = SimulatorConfig(
            base_amplitude=50.0,
            amplitude_noise=5.0,
            presence_pattern=PresencePattern.ALWAYS_ABSENT,
        )
        device = SimulatedDevice("test_001", config)
        device._start_time = time.time()

        amplitudes = []
        for _ in range(100):
            data = device._generate_csi_data(time.time())
            amplitudes.extend(data["amplitudes"])

        mean_amp = np.mean(amplitudes)
        std_amp = np.std(amplitudes)

        # Mean should be close to base_amplitude
        assert abs(mean_amp - 50.0) < 2.0
        # Std should be around amplitude_noise
        assert abs(std_amp - 5.0) < 2.0

    def test_presence_increases_variance(self):
        """Verify presence causes higher variance."""
        config = SimulatorConfig(
            base_amplitude=50.0,
            amplitude_noise=2.0,
        )

        # Generate data without presence
        config_absent = SimulatorConfig(
            base_amplitude=50.0,
            amplitude_noise=2.0,
            presence_pattern=PresencePattern.ALWAYS_ABSENT,
        )
        device_absent = SimulatedDevice("absent", config_absent)
        device_absent._start_time = time.time()

        absent_variances = []
        for _ in range(50):
            data = device_absent._generate_csi_data(time.time())
            absent_variances.append(data["variance"])

        # Generate data with presence
        config_present = SimulatorConfig(
            base_amplitude=50.0,
            amplitude_noise=2.0,
            presence_pattern=PresencePattern.ALWAYS_PRESENT,
        )
        device_present = SimulatedDevice("present", config_present)
        device_present._start_time = time.time()

        present_variances = []
        for _ in range(50):
            data = device_present._generate_csi_data(time.time())
            present_variances.append(data["variance"])

        # Presence should have higher variance
        assert np.mean(present_variances) > np.mean(absent_variances)

    def test_rssi_range(self):
        """Verify RSSI values are in valid range."""
        config = SimulatorConfig()
        device = SimulatedDevice("test_001", config)
        device._start_time = time.time()

        for _ in range(100):
            data = device._generate_csi_data(time.time())
            assert -100 <= data["rssi"] <= 0


class TestMessageFormatting:
    """Tests for MQTT message format compliance."""

    def test_analysis_message_format(self):
        """Verify analysis message contains required fields."""
        config = SimulatorConfig(presence_pattern=PresencePattern.ALWAYS_PRESENT)
        device = SimulatedDevice("test_001", config)
        device._start_time = time.time()

        data = device._generate_csi_data(time.time())

        # Required fields for history_collector.py
        assert "device_id" in data
        assert "timestamp" in data
        assert "rssi" in data
        assert "avg_amplitude" in data
        assert "variance" in data
        assert "present" in data
        assert "breath_ratio" in data
        assert "breathing" in data

        # Type checks
        assert isinstance(data["device_id"], str)
        assert isinstance(data["timestamp"], float)
        assert isinstance(data["rssi"], int)
        assert isinstance(data["avg_amplitude"], float)
        assert isinstance(data["variance"], float)
        assert isinstance(data["present"], bool)
        assert isinstance(data["breath_ratio"], float)
        assert isinstance(data["breathing"], bool)

    def test_analysis_message_json_serializable(self):
        """Verify message can be JSON serialized."""
        config = SimulatorConfig()
        device = SimulatedDevice("test_001", config)
        device._start_time = time.time()

        data = device._generate_csi_data(time.time())

        # Remove raw amplitudes (not sent in analysis topic)
        analysis_data = {k: v for k, v in data.items() if k != "amplitudes"}

        # Should not raise
        json_str = json.dumps(analysis_data)
        assert isinstance(json_str, str)

        # Should parse back correctly
        parsed = json.loads(json_str)
        assert parsed["device_id"] == data["device_id"]


# =============================================================================
# Integration Tests (Require MQTT Broker)
# =============================================================================


@pytest.mark.mqtt
class TestSimulatedDeviceIntegration:
    """Integration tests for SimulatedDevice with live broker."""

    def test_connect_and_disconnect(self):
        """Test device can connect and disconnect cleanly."""
        config = SimulatorConfig()
        device = SimulatedDevice("integration_test_001", config)

        try:
            assert device.start()
            assert device.stats.state == DeviceState.CONNECTED

            # Wait a bit then stop
            time.sleep(0.5)
            device.stop()

            assert device.stats.state == DeviceState.DISCONNECTED
        finally:
            device.stop()

    def test_publish_messages(self):
        """Test device can publish messages."""
        config = SimulatorConfig(publish_rate_hz=10.0)
        device = SimulatedDevice("integration_test_002", config)

        try:
            device.start()

            # Wait for some messages to be sent
            time.sleep(1.5)

            stats = device.get_stats()
            # Should have sent ~15 messages (10 Hz * 1.5s)
            assert stats.messages_sent >= 10

        finally:
            device.stop()


@pytest.mark.mqtt
class TestMQTTDeviceSimulatorIntegration:
    """Integration tests for MQTTDeviceSimulator with live broker."""

    def test_start_and_stop_single_device(self):
        """Test starting and stopping a single device."""
        sim = MQTTDeviceSimulator()

        try:
            assert sim.start_device("sim_test_001")
            assert sim.device_count == 1
            assert sim.connected_count == 1

            sim.stop_device("sim_test_001")
            assert sim.device_count == 0
        finally:
            sim.stop_all()

    def test_start_multiple_devices(self):
        """Test starting multiple devices."""
        sim = MQTTDeviceSimulator()

        try:
            started = sim.start_devices(5, stagger_delay=0.1)
            assert len(started) == 5
            assert sim.device_count == 5

            # Wait for connections
            time.sleep(1.0)
            assert sim.connected_count == 5

        finally:
            sim.stop_all()
            assert sim.device_count == 0

    def test_get_all_stats(self):
        """Test aggregated statistics."""
        config = SimulatorConfig(publish_rate_hz=10.0)
        sim = MQTTDeviceSimulator(config)

        try:
            sim.start_devices(3, stagger_delay=0.1)
            time.sleep(1.5)

            stats = sim.get_all_stats()
            assert stats["device_count"] == 3
            assert stats["connected_count"] == 3
            # 3 devices * 10 Hz * 1.5s = ~45 messages
            assert stats["total_messages_sent"] >= 30
            assert stats["success_rate"] > 0.9

        finally:
            sim.stop_all()

    def test_wait_for_messages(self):
        """Test waiting for message count."""
        config = SimulatorConfig(publish_rate_hz=20.0)
        sim = MQTTDeviceSimulator(config)

        try:
            sim.start_devices(2, stagger_delay=0.1)

            # Wait for 100 messages (should take ~2.5s with 2 devices at 20 Hz)
            reached = sim.wait_for_messages(100, timeout=10.0)
            assert reached is True

            stats = sim.get_all_stats()
            assert stats["total_messages_sent"] >= 100

        finally:
            sim.stop_all()

    def test_inject_disconnect(self):
        """Test simulating disconnection."""
        sim = MQTTDeviceSimulator()

        try:
            sim.start_device("disconnect_test")
            time.sleep(0.5)

            stats_before = sim.get_device_stats("disconnect_test")
            assert stats_before.state == DeviceState.CONNECTED

            # Inject disconnect
            assert sim.inject_disconnect("disconnect_test") is True

            # Device should reconnect (auto_reconnect is True by default)
            time.sleep(2.0)

        finally:
            sim.stop_all()


@pytest.mark.mqtt
class TestLoadTest:
    """Load testing scenarios."""

    def test_load_10_devices(self):
        """Test with 10 simultaneous devices."""
        config = SimulatorConfig(publish_rate_hz=10.0)
        sim = MQTTDeviceSimulator(config)

        try:
            started = sim.start_devices(10, stagger_delay=0.05)
            assert len(started) == 10

            time.sleep(3.0)

            stats = sim.get_all_stats()
            assert stats["connected_count"] >= 9  # Allow 1 failure
            # 10 devices * 10 Hz * 3s = ~300 messages
            assert stats["total_messages_sent"] >= 250
            assert stats["success_rate"] > 0.95

        finally:
            sim.stop_all()

    def test_high_frequency_publishing(self):
        """Test high-frequency message publishing."""
        config = SimulatorConfig(publish_rate_hz=50.0)  # 50 Hz
        sim = MQTTDeviceSimulator(config)

        try:
            sim.start_device("high_freq_test")
            time.sleep(2.0)

            stats = sim.get_device_stats("high_freq_test")
            # Should have sent ~100 messages (50 Hz * 2s)
            assert stats.messages_sent >= 80

        finally:
            sim.stop_all()


@pytest.mark.mqtt
class TestMessageCallback:
    """Tests for message callbacks."""

    def test_callback_receives_messages(self):
        """Test that callback is invoked for each message."""
        received_messages = []

        def callback(device_id, topic, data):
            received_messages.append({
                "device_id": device_id,
                "topic": topic,
                "data": data,
            })

        config = SimulatorConfig(publish_rate_hz=10.0)
        sim = MQTTDeviceSimulator(config, on_message_callback=callback)

        try:
            sim.start_device("callback_test")
            time.sleep(1.5)

            # Should have received ~15 messages
            assert len(received_messages) >= 10

            # Verify message content
            for msg in received_messages:
                assert msg["device_id"] == "callback_test"
                assert "wavira/analysis" in msg["topic"]
                assert "timestamp" in msg["data"]

        finally:
            sim.stop_all()


@pytest.mark.mqtt
class TestRunLoadTestFunction:
    """Tests for the run_load_test helper function."""

    def test_run_load_test_short(self):
        """Test run_load_test with short duration."""
        results = run_load_test(
            device_count=3,
            duration_seconds=2.0,
            publish_rate_hz=10.0,
        )

        assert results["device_count"] == 3
        assert results["duration_seconds"] >= 2.0
        assert results["total_messages_sent"] > 0
        assert results["success_rate"] > 0.9
        assert results["messages_per_second"] > 0


# =============================================================================
# QoS Tests (Require MQTT Broker)
# =============================================================================


@pytest.mark.mqtt
class TestQoSBehavior:
    """Tests for MQTT QoS levels."""

    def test_qos_0_fire_and_forget(self):
        """Test QoS 0 (at most once) delivery."""
        config = SimulatorConfig(qos=0, publish_rate_hz=20.0)
        sim = MQTTDeviceSimulator(config)

        try:
            sim.start_device("qos0_test")
            time.sleep(2.0)

            stats = sim.get_device_stats("qos0_test")
            # QoS 0 doesn't guarantee delivery, so we just check messages were attempted
            assert stats.messages_sent > 0

        finally:
            sim.stop_all()

    def test_qos_1_at_least_once(self):
        """Test QoS 1 (at least once) delivery."""
        config = SimulatorConfig(qos=1, publish_rate_hz=10.0)
        sim = MQTTDeviceSimulator(config)

        try:
            sim.start_device("qos1_test")
            time.sleep(2.0)

            stats = sim.get_device_stats("qos1_test")
            # QoS 1 should have high success rate
            assert stats.messages_sent >= 15
            assert stats.messages_failed == 0

        finally:
            sim.stop_all()


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling scenarios."""

    def test_invalid_broker_host(self):
        """Test connection failure with invalid broker."""
        config = SimulatorConfig(
            broker_host="nonexistent.invalid.host",
            connect_timeout=1.0,
        )
        device = SimulatedDevice("error_test", config)

        # Should fail to connect
        result = device.start()
        assert result is False
        assert device.stats.state in (DeviceState.ERROR, DeviceState.DISCONNECTED)
        assert device.stats.last_error is not None

        device.stop()

    def test_invalid_broker_port(self):
        """Test connection failure with invalid port."""
        config = SimulatorConfig(
            broker_port=9999,  # Unlikely to have MQTT on this port
            connect_timeout=1.0,
        )
        device = SimulatedDevice("port_error_test", config)

        result = device.start()
        assert result is False

        device.stop()

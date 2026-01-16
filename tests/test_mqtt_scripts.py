"""
Unit tests for MQTT-based scripts.

Tests for:
- collect_crowd_mqtt.py
- monitor_crowd_mqtt.py
- serial_mqtt_bridge.py

These are unit tests that mock MQTT connections.
"""

import json
import time
from unittest.mock import Mock, MagicMock, patch
import pytest
import numpy as np
import tempfile
import os
import sys

# Add scripts directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestCollectCrowdMQTT:
    """Tests for collect_crowd_mqtt.py MQTTCrowdCollector."""

    def test_crowd_levels_defined(self):
        """Verify crowd level mappings exist."""
        from scripts.collect_crowd_mqtt import CROWD_LEVELS, DEFAULT_NUM_PEOPLE

        assert CROWD_LEVELS[0] == "empty"
        assert CROWD_LEVELS[1] == "moderate"
        assert CROWD_LEVELS[2] == "crowded"

        assert DEFAULT_NUM_PEOPLE[0] == 0
        assert DEFAULT_NUM_PEOPLE[1] == 3
        assert DEFAULT_NUM_PEOPLE[2] == 8

    def test_collector_initialization(self):
        """Test MQTTCrowdCollector initialization."""
        from scripts.collect_crowd_mqtt import MQTTCrowdCollector

        with tempfile.TemporaryDirectory() as tmpdir:
            collector = MQTTCrowdCollector(
                mqtt_host="localhost",
                mqtt_port=1883,
                output_dir=tmpdir,
                level=0,
                location="test_room",
                samples_per_file=50,
                num_files=5,
            )

            assert collector.level == 0
            assert collector.level_name == "empty"
            assert collector.location == "test_room"
            assert collector.samples_per_file == 50
            assert collector.num_files == 5
            assert collector.num_people == 0  # Default for level 0

    def test_collector_custom_num_people(self):
        """Test collector with custom num_people override."""
        from scripts.collect_crowd_mqtt import MQTTCrowdCollector

        with tempfile.TemporaryDirectory() as tmpdir:
            collector = MQTTCrowdCollector(
                mqtt_host="localhost",
                mqtt_port=1883,
                output_dir=tmpdir,
                level=1,
                location="test",
                num_people=10,  # Override default
            )

            assert collector.num_people == 10

    def test_on_message_processing(self):
        """Test CSI message processing."""
        from scripts.collect_crowd_mqtt import MQTTCrowdCollector

        with tempfile.TemporaryDirectory() as tmpdir:
            collector = MQTTCrowdCollector(
                mqtt_host="localhost",
                mqtt_port=1883,
                output_dir=tmpdir,
                level=0,
                location="test",
            )

            # Simulate incoming message with I/Q pairs
            csi_data = list(range(104))  # 52 subcarriers * 2 (I/Q)
            payload = json.dumps({
                "data": csi_data,
                "rssi": -45,
            }).encode()

            msg = Mock()
            msg.payload = payload

            # Process message
            collector._on_message(None, None, msg)

            assert len(collector.csi_buffer) == 1
            assert len(collector.rssi_buffer) == 1
            assert collector.rssi_buffer[0] == -45
            # Amplitude should be 52 values (from 104 I/Q pairs)
            assert len(collector.csi_buffer[0]) == 52

    def test_save_file(self):
        """Test HDF5 file saving."""
        from scripts.collect_crowd_mqtt import MQTTCrowdCollector
        import h5py

        with tempfile.TemporaryDirectory() as tmpdir:
            collector = MQTTCrowdCollector(
                mqtt_host="localhost",
                mqtt_port=1883,
                output_dir=tmpdir,
                level=1,
                location="office",
                samples_per_file=10,
            )

            # Add samples to buffer
            for i in range(15):
                collector.csi_buffer.append(np.random.randn(52).astype(np.float32))
                collector.rssi_buffer.append(-50 + i)

            # Save file
            result = collector._save_file()
            assert result is True
            assert collector.collected_files == 1
            assert len(collector.csi_buffer) == 5  # 15 - 10 = 5 remaining

            # Verify HDF5 content
            files = [f for f in os.listdir(tmpdir) if f.endswith('.h5')]
            assert len(files) == 1

            with h5py.File(os.path.join(tmpdir, files[0]), 'r') as f:
                assert 'amplitudes' in f
                assert 'rssi' in f
                assert f['amplitudes'].shape[0] == 10  # samples_per_file
                assert f.attrs['level'] == 1
                assert f.attrs['level_name'] == "moderate"
                assert f.attrs['location'] == "office"


class TestMonitorCrowdMQTT:
    """Tests for monitor_crowd_mqtt.py MQTTCrowdMonitor."""

    def test_crowd_labels_defined(self):
        """Verify crowd label mappings exist."""
        from scripts.monitor_crowd_mqtt import CROWD_LABELS

        assert "Empty" in CROWD_LABELS[0]
        assert "Low" in CROWD_LABELS[1]
        assert "Medium" in CROWD_LABELS[2]
        assert "High" in CROWD_LABELS[3]

    @patch('scripts.monitor_crowd_mqtt.torch.load')
    @patch('scripts.monitor_crowd_mqtt.CrowdEstimator')
    def test_monitor_initialization(self, mock_estimator_class, mock_torch_load):
        """Test MQTTCrowdMonitor initialization with mocked model."""
        from scripts.monitor_crowd_mqtt import MQTTCrowdMonitor

        # Mock checkpoint
        mock_state_dict = {
            "head.3.weight": Mock(shape=(4, 128)),  # 4 classes = classification
            "temporal_conv.0.conv.weight": Mock(shape=(64, 52, 3)),  # 52 subcarriers
        }
        mock_torch_load.return_value = mock_state_dict

        # Mock model
        mock_model = MagicMock()
        mock_estimator_class.return_value = mock_model

        monitor = MQTTCrowdMonitor(
            model_path="fake_model.pt",
            mqtt_host="localhost",
            mqtt_port=1883,
            window_size=50,
        )

        assert monitor.window_size == 50
        assert monitor.n_subcarriers == 52

    def test_message_processing(self):
        """Test CSI message processing in monitor."""
        from scripts.monitor_crowd_mqtt import MQTTCrowdMonitor

        # Create a minimal mock monitor
        with patch('scripts.monitor_crowd_mqtt.torch.load') as mock_load, \
             patch('scripts.monitor_crowd_mqtt.CrowdEstimator') as mock_cls:

            mock_state_dict = {
                "head.3.weight": Mock(shape=(4, 128)),
                "temporal_conv.0.conv.weight": Mock(shape=(64, 52, 3)),
            }
            mock_load.return_value = mock_state_dict
            mock_model = MagicMock()
            mock_cls.return_value = mock_model

            monitor = MQTTCrowdMonitor(
                model_path="fake.pt",
                window_size=10,
            )

            # Process messages
            for _ in range(5):
                csi_data = list(range(104))
                payload = json.dumps({"data": csi_data}).encode()
                msg = Mock()
                msg.payload = payload
                monitor._on_message(None, None, msg)

            assert len(monitor.csi_buffer) == 5


class TestSerialMQTTBridge:
    """Tests for serial_mqtt_bridge.py SerialMQTTBridge."""

    def test_bridge_initialization(self):
        """Test SerialMQTTBridge initialization."""
        from scripts.serial_mqtt_bridge import SerialMQTTBridge

        bridge = SerialMQTTBridge(
            port="/dev/ttyUSB0",
            baudrate=115200,
            mqtt_host="localhost",
            mqtt_port=1883,
            device_id="esp32-test",
        )

        assert bridge.port == "/dev/ttyUSB0"
        assert bridge.baudrate == 115200
        assert bridge.device_id == "esp32-test"

    def test_device_id_generation(self):
        """Test automatic device ID generation from port."""
        from scripts.serial_mqtt_bridge import SerialMQTTBridge

        bridge = SerialMQTTBridge(
            port="/dev/cu.usbmodem12345",
            mqtt_host="localhost",
            mqtt_port=1883,
        )

        assert "usbmodem12345" in bridge.device_id

    def test_parse_csi_line_valid(self):
        """Test parsing valid CSI_DATA line."""
        from scripts.serial_mqtt_bridge import SerialMQTTBridge

        bridge = SerialMQTTBridge(
            port="/dev/test",
            device_id="test_device",
        )

        # Valid CSI_DATA format
        line = 'CSI_DATA,123,AA:BB:CC:DD:EE:FF,-50,11,5,extra1,extra2,"[1,2,3,4,5,6]"'

        result = bridge._parse_csi_line(line)

        assert result is not None
        assert result["device_id"] == "test_device"
        assert result["seq"] == 123
        assert result["mac"] == "AA:BB:CC:DD:EE:FF"
        assert result["rssi"] == -50
        assert result["rate"] == 11
        assert result["noise_floor"] == 5
        assert result["data"] == [1, 2, 3, 4, 5, 6]
        assert "timestamp" in result

    def test_parse_csi_line_invalid(self):
        """Test parsing invalid lines returns None."""
        from scripts.serial_mqtt_bridge import SerialMQTTBridge

        bridge = SerialMQTTBridge(port="/dev/test")

        # Not a CSI_DATA line
        assert bridge._parse_csi_line("INFO: Boot complete") is None

        # Missing array
        assert bridge._parse_csi_line("CSI_DATA,1,mac,-50,11,5") is None

        # Malformed JSON
        assert bridge._parse_csi_line('CSI_DATA,1,mac,-50,11,5,"[broken"') is None

    def test_parse_csi_line_real_format(self):
        """Test parsing real ESP32 CSI output format."""
        from scripts.serial_mqtt_bridge import SerialMQTTBridge

        bridge = SerialMQTTBridge(
            port="/dev/test",
            device_id="esp32-001",
        )

        # Realistic ESP32 output with I/Q data
        csi_values = [10, 20, 15, 25, 12, 22] * 17  # 102 values (51 I/Q pairs)
        line = f'CSI_DATA,42,FF:FF:FF:FF:FF:FF,-45,11,-92,128,0,0,1,1,0,0,0,0,0,0,0,"{csi_values}"'

        result = bridge._parse_csi_line(line)

        assert result is not None
        assert result["seq"] == 42
        assert result["rssi"] == -45
        assert len(result["data"]) == len(csi_values)


class TestCSIDataConversion:
    """Tests for CSI data conversion (I/Q to amplitude)."""

    def test_iq_to_amplitude(self):
        """Test I/Q pair conversion to amplitude."""
        # I/Q pairs: [I0, Q0, I1, Q1, ...]
        csi_raw = [3, 4, 6, 8, 0, 5]  # Expected amplitudes: 5, 10, 5

        csi_array = np.array(csi_raw, dtype=np.float32)
        csi_complex = csi_array[0::2] + 1j * csi_array[1::2]
        amplitude = np.abs(csi_complex)

        np.testing.assert_array_almost_equal(amplitude, [5.0, 10.0, 5.0])

    def test_odd_length_passthrough(self):
        """Test odd-length CSI data is passed through as-is."""
        csi_raw = [1, 2, 3, 4, 5]  # Odd length

        csi_array = np.array(csi_raw, dtype=np.float32)
        if len(csi_array) % 2 != 0:
            amplitude = csi_array
        else:
            csi_complex = csi_array[0::2] + 1j * csi_array[1::2]
            amplitude = np.abs(csi_complex)

        np.testing.assert_array_equal(amplitude, [1, 2, 3, 4, 5])


class TestScriptImports:
    """Verify all scripts can be imported without errors."""

    def test_import_collect_crowd_mqtt(self):
        """Test collect_crowd_mqtt.py can be imported."""
        import scripts.collect_crowd_mqtt as module
        assert hasattr(module, 'MQTTCrowdCollector')
        assert hasattr(module, 'CROWD_LEVELS')

    def test_import_monitor_crowd_mqtt(self):
        """Test monitor_crowd_mqtt.py can be imported."""
        import scripts.monitor_crowd_mqtt as module
        assert hasattr(module, 'MQTTCrowdMonitor')
        assert hasattr(module, 'CROWD_LABELS')

    def test_import_serial_mqtt_bridge(self):
        """Test serial_mqtt_bridge.py can be imported."""
        import scripts.serial_mqtt_bridge as module
        assert hasattr(module, 'SerialMQTTBridge')

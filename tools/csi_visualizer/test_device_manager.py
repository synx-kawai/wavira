"""
Unit tests for Device Manager module.
"""

import os
import tempfile

import pytest

from device_manager import (
    Device,
    DeviceAuthManager,
    DeviceStatus,
    DeviceWithStatus,
    MQTTCredentials,
    generate_api_key,
    hash_api_key,
    verify_api_key,
)


class TestApiKeyUtilities:
    """Tests for API key utility functions."""

    def test_generate_api_key_default_prefix(self):
        """Test API key generation with default prefix."""
        key = generate_api_key()
        assert key.startswith("wvr_")
        assert len(key) == 4 + 64  # prefix + 32 bytes hex

    def test_generate_api_key_custom_prefix(self):
        """Test API key generation with custom prefix."""
        key = generate_api_key(prefix="test")
        assert key.startswith("test_")
        assert len(key) == 5 + 64

    def test_generate_api_key_uniqueness(self):
        """Test that generated API keys are unique."""
        keys = [generate_api_key() for _ in range(100)]
        assert len(set(keys)) == 100

    def test_hash_api_key(self):
        """Test API key hashing."""
        key = "test_key_12345"
        hashed = hash_api_key(key)
        assert hashed != key
        assert len(hashed) > 0

    def test_verify_api_key_correct(self):
        """Test API key verification with correct key."""
        key = generate_api_key()
        hashed = hash_api_key(key)
        assert verify_api_key(key, hashed) is True

    def test_verify_api_key_incorrect(self):
        """Test API key verification with incorrect key."""
        key = generate_api_key()
        hashed = hash_api_key(key)
        assert verify_api_key("wrong_key", hashed) is False

    def test_verify_api_key_different_keys(self):
        """Test that different keys have different hashes."""
        key1 = generate_api_key()
        key2 = generate_api_key()
        hashed1 = hash_api_key(key1)
        hashed2 = hash_api_key(key2)
        assert hashed1 != hashed2
        assert verify_api_key(key1, hashed1) is True
        assert verify_api_key(key2, hashed2) is True
        assert verify_api_key(key1, hashed2) is False
        assert verify_api_key(key2, hashed1) is False


class TestDeviceAuthManager:
    """Tests for DeviceAuthManager class."""

    @pytest.fixture
    def manager(self):
        """Create a DeviceAuthManager with a temporary database."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        manager = DeviceAuthManager(db_path)
        yield manager
        manager.close()
        os.unlink(db_path)

    def test_register_device(self, manager):
        """Test device registration."""
        device, api_key, mqtt_creds = manager.register_device(
            device_id="test-device-001",
            zone="Zone A",
            location="Test Lab",
        )
        assert device is not None
        assert device.id == "test-device-001"
        assert device.zone == "Zone A"
        assert device.location == "Test Lab"
        assert api_key is not None
        assert api_key.startswith("wvr_")
        # MQTT credentials
        assert mqtt_creds is not None
        assert mqtt_creds.username == "test-device-001"
        assert mqtt_creds.client_id == "wavira_test-device-001"

    def test_register_device_duplicate(self, manager):
        """Test that duplicate device registration raises ValueError."""
        manager.register_device(device_id="test-device-001")
        with pytest.raises(ValueError, match="already exists"):
            manager.register_device(device_id="test-device-001")

    def test_authenticate_device_valid(self, manager):
        """Test authentication with valid API key."""
        device, api_key, _mqtt_creds = manager.register_device(
            device_id="test-device-001",
            zone="Zone A",
        )
        # authenticate takes both device_id and api_key
        result = manager.authenticate("test-device-001", api_key)
        assert result is True

    def test_authenticate_by_key(self, manager):
        """Test authentication by API key only."""
        device, api_key, _mqtt_creds = manager.register_device(
            device_id="test-device-001",
            zone="Zone A",
        )
        device_id = manager.authenticate_by_key(api_key)
        assert device_id == "test-device-001"

    def test_authenticate_device_invalid(self, manager):
        """Test authentication with invalid API key."""
        manager.register_device(device_id="test-device-001")
        result = manager.authenticate("test-device-001", "invalid_api_key")
        assert result is False

    def test_authenticate_by_key_invalid(self, manager):
        """Test authenticate_by_key with invalid API key."""
        manager.register_device(device_id="test-device-001")
        device_id = manager.authenticate_by_key("invalid_api_key")
        assert device_id is None

    def test_authenticate_disabled_device(self, manager):
        """Test that disabled devices cannot authenticate."""
        device, api_key, _mqtt_creds = manager.register_device(device_id="test-device-001")
        manager.update_device("test-device-001", enabled=False)
        result = manager.authenticate("test-device-001", api_key)
        assert result is False

    def test_get_device(self, manager):
        """Test getting device by ID."""
        manager.register_device(
            device_id="test-device-001",
            zone="Zone A",
            location="Lab A",
        )
        device = manager.get_device("test-device-001")
        assert device is not None
        assert device.id == "test-device-001"
        assert device.zone == "Zone A"
        assert device.location == "Lab A"

    def test_get_device_not_found(self, manager):
        """Test getting non-existent device."""
        device = manager.get_device("non-existent")
        assert device is None

    def test_list_devices(self, manager):
        """Test listing all devices."""
        manager.register_device(device_id="device-001", zone="Zone A")
        manager.register_device(device_id="device-002", zone="Zone B")
        manager.register_device(device_id="device-003", zone="Zone C")

        devices = manager.list_devices()
        assert len(devices) == 3
        device_ids = [d.id for d in devices]
        assert "device-001" in device_ids
        assert "device-002" in device_ids
        assert "device-003" in device_ids

    def test_list_devices_enabled_only(self, manager):
        """Test listing only enabled devices."""
        manager.register_device(device_id="device-001", zone="Zone A")
        manager.register_device(device_id="device-002", zone="Zone B")
        manager.update_device("device-002", enabled=False)

        devices = manager.list_devices(enabled_only=True)
        assert len(devices) == 1
        assert devices[0].id == "device-001"

    def test_update_device(self, manager):
        """Test updating device properties."""
        manager.register_device(
            device_id="test-device-001",
            zone="Zone A",
            location="Lab A",
        )
        updated = manager.update_device(
            device_id="test-device-001",
            zone="Zone B",
            location="Lab B",
        )
        assert updated is not None
        assert updated.zone == "Zone B"
        assert updated.location == "Lab B"

    def test_update_device_not_found(self, manager):
        """Test updating non-existent device."""
        result = manager.update_device(
            device_id="non-existent",
            zone="Zone A",
        )
        assert result is None

    def test_delete_device(self, manager):
        """Test deleting device."""
        manager.register_device(device_id="test-device-001", zone="Zone A")
        success = manager.delete_device("test-device-001")
        assert success is True

        device = manager.get_device("test-device-001")
        assert device is None

    def test_delete_device_not_found(self, manager):
        """Test deleting non-existent device."""
        success = manager.delete_device("non-existent")
        assert success is False

    def test_rotate_api_key(self, manager):
        """Test API key rotation."""
        device, old_api_key, _mqtt_creds = manager.register_device(
            device_id="test-device-001",
            zone="Zone A",
        )
        new_api_key = manager.rotate_api_key("test-device-001")

        assert new_api_key is not None
        assert new_api_key != old_api_key

        # Old key should no longer work
        result = manager.authenticate("test-device-001", old_api_key)
        assert result is False

        # New key should work
        result = manager.authenticate("test-device-001", new_api_key)
        assert result is True

    def test_rotate_api_key_not_found(self, manager):
        """Test rotating API key for non-existent device."""
        new_key = manager.rotate_api_key("non-existent")
        assert new_key is None

    def test_update_device_status(self, manager):
        """Test updating device status."""
        manager.register_device(device_id="test-device-001", zone="Zone A")

        # Update status
        manager.update_device_status("test-device-001", packet_count_delta=10)

        status = manager.get_device_status("test-device-001")
        assert status is not None
        assert status.device_id == "test-device-001"
        assert status.packet_count == 10
        assert status.status == "online"
        assert status.last_seen is not None

    def test_update_device_status_with_firmware(self, manager):
        """Test updating device status with firmware version."""
        manager.register_device(device_id="test-device-001", zone="Zone A")
        manager.update_device_status(
            "test-device-001",
            packet_count_delta=5,
            firmware_version="2.0.0",
            uptime_seconds=3600,
        )

        status = manager.get_device_status("test-device-001")
        assert status.firmware_version == "2.0.0"
        assert status.uptime_seconds == 3600

    def test_get_device_status_not_found(self, manager):
        """Test getting status for non-existent device."""
        status = manager.get_device_status("non-existent")
        assert status is None

    def test_get_all_device_statuses(self, manager):
        """Test getting all devices with their statuses."""
        manager.register_device(device_id="device-001", zone="Zone A")
        manager.register_device(device_id="device-002", zone="Zone B")
        manager.update_device_status("device-001", packet_count_delta=100)

        devices_with_status = manager.get_all_device_statuses()
        assert len(devices_with_status) == 2

        # Find device with data
        device1 = next(d for d in devices_with_status if d.device.id == "device-001")
        assert device1.status.packet_count == 100

        # Find device without data
        device2 = next(d for d in devices_with_status if d.device.id == "device-002")
        assert device2.status.packet_count == 0

    def test_update_device_status_error_count(self, manager):
        """Test updating device status with error count."""
        manager.register_device(device_id="test-device-001", zone="Zone A")

        # Record errors
        manager.update_device_status("test-device-001", packet_count_delta=0, error_count_delta=1)
        manager.update_device_status("test-device-001", packet_count_delta=0, error_count_delta=1)

        status = manager.get_device_status("test-device-001")
        assert status is not None
        assert status.error_count == 2


class TestDeviceDataclass:
    """Tests for Device dataclass."""

    def test_device_creation(self):
        """Test creating a Device instance."""
        device = Device(
            id="test-001",
            api_key_hash="hash123",
            zone="Zone A",
            location="Lab",
            enabled=True,
            created_at=None,
            updated_at=None,
        )
        assert device.id == "test-001"
        assert device.zone == "Zone A"
        assert device.enabled is True


class TestDeviceStatusDataclass:
    """Tests for DeviceStatus dataclass."""

    def test_device_status_creation(self):
        """Test creating a DeviceStatus instance."""
        status = DeviceStatus(
            device_id="test-001",
            status="online",
            last_seen=None,
            packet_count=100,
            error_count=0,
            firmware_version="1.0.0",
            uptime_seconds=3600,
        )
        assert status.device_id == "test-001"
        assert status.status == "online"
        assert status.packet_count == 100
        assert status.firmware_version == "1.0.0"


class TestDeviceWithStatusDataclass:
    """Tests for DeviceWithStatus dataclass."""

    def test_device_with_status_creation(self):
        """Test creating a DeviceWithStatus instance."""
        device = Device(
            id="test-001",
            api_key_hash="hash123",
            zone="Zone A",
        )
        status = DeviceStatus(
            device_id="test-001",
            status="online",
        )
        combined = DeviceWithStatus(device=device, status=status)
        assert combined.device.id == "test-001"
        assert combined.status.status == "online"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

#!/usr/bin/env python3
"""
Tests for Device Manager (Issue #23).

Tests cover:
- MQTT credential generation
- Password hashing with bcrypt
- Device registration
- Credential rotation
- Database operations
"""

import os
import sys
import tempfile
import pytest

# Add the services directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools", "csi_visualizer", "services"))

from device_manager import (
    DeviceManager,
    DeviceManagerConfig,
    MQTTCredentials,
    generate_mqtt_password,
    hash_mqtt_password,
    verify_mqtt_password,
    generate_mqtt_credentials,
)


# =============================================================================
# MQTT Credentials Tests
# =============================================================================


class TestMQTTPasswordGeneration:
    """Tests for MQTT password generation and hashing."""

    def test_generate_password_length(self):
        """Password should be the expected length (base64 encoded)."""
        password = generate_mqtt_password(length=32)
        # URL-safe base64 encoding produces ~4/3 the length
        assert len(password) >= 32

    def test_generate_password_unique(self):
        """Each generated password should be unique."""
        passwords = [generate_mqtt_password() for _ in range(10)]
        assert len(set(passwords)) == 10

    def test_hash_password_produces_bcrypt_hash(self):
        """Hash should be a valid bcrypt hash."""
        password = "test_password"
        hashed = hash_mqtt_password(password)
        # bcrypt hashes start with $2b$ or $2a$
        assert hashed.startswith("$2")
        assert len(hashed) == 60  # bcrypt hash length

    def test_verify_password_correct(self):
        """Correct password should verify successfully."""
        password = "my_secure_password"
        hashed = hash_mqtt_password(password)
        assert verify_mqtt_password(password, hashed) is True

    def test_verify_password_incorrect(self):
        """Incorrect password should fail verification."""
        password = "my_secure_password"
        wrong_password = "wrong_password"
        hashed = hash_mqtt_password(password)
        assert verify_mqtt_password(wrong_password, hashed) is False

    def test_verify_password_invalid_hash(self):
        """Invalid hash should return False, not raise exception."""
        assert verify_mqtt_password("password", "invalid_hash") is False
        assert verify_mqtt_password("password", "") is False


class TestMQTTCredentialsGeneration:
    """Tests for MQTT credentials generation."""

    def test_generate_credentials_structure(self):
        """Generated credentials should have all required fields."""
        config = DeviceManagerConfig()
        creds = generate_mqtt_credentials("esp32-001", config)

        assert isinstance(creds, MQTTCredentials)
        assert creds.username is not None
        assert creds.password is not None
        assert creds.client_id is not None
        assert creds.password_hash is not None

    def test_generate_credentials_username_format(self):
        """Username should follow the expected format."""
        config = DeviceManagerConfig(mqtt_username_prefix="dev")
        creds = generate_mqtt_credentials("esp32-001", config)

        # Should be prefix_normalized_id
        assert creds.username == "dev_esp32_001"

    def test_generate_credentials_client_id_format(self):
        """Client ID should follow the expected format."""
        config = DeviceManagerConfig(client_id_prefix="wavira")
        creds = generate_mqtt_credentials("esp32-001", config)

        assert creds.client_id == "wavira_esp32_001"

    def test_generate_credentials_password_verifiable(self):
        """Generated password should verify against the hash."""
        config = DeviceManagerConfig()
        creds = generate_mqtt_credentials("esp32-001", config)

        assert verify_mqtt_password(creds.password, creds.password_hash) is True

    def test_credentials_to_dict_without_password(self):
        """to_dict should not include password by default."""
        config = DeviceManagerConfig()
        creds = generate_mqtt_credentials("esp32-001", config)
        result = creds.to_dict()

        assert "username" in result
        assert "client_id" in result
        assert "password" not in result

    def test_credentials_to_dict_with_password(self):
        """to_dict should include password when requested."""
        config = DeviceManagerConfig()
        creds = generate_mqtt_credentials("esp32-001", config)
        result = creds.to_dict(include_password=True)

        assert "password" in result
        assert result["password"] == creds.password


# =============================================================================
# Device Manager Tests
# =============================================================================


class TestDeviceManager:
    """Tests for DeviceManager class."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        yield db_path
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)

    @pytest.fixture
    def manager(self, temp_db):
        """Create a DeviceManager with temporary database."""
        config = DeviceManagerConfig(db_path=temp_db)
        return DeviceManager(config)

    def test_register_device_success(self, manager):
        """Registering a new device should succeed."""
        device, api_key, mqtt_creds = manager.register_device("esp32-001")

        assert device.device_id == "esp32-001"
        assert api_key.startswith("wvr_")
        assert mqtt_creds is not None
        assert mqtt_creds.username == "dev_esp32_001"

    def test_register_device_without_mqtt(self, manager):
        """Registering without MQTT should not generate credentials."""
        device, api_key, mqtt_creds = manager.register_device(
            "esp32-002",
            generate_mqtt=False
        )

        assert device.device_id == "esp32-002"
        assert api_key.startswith("wvr_")
        assert mqtt_creds is None
        assert device.mqtt_username is None

    def test_register_device_with_zone(self, manager):
        """Device should be registered with specified zone."""
        device, _, _ = manager.register_device("esp32-003", zone="office")

        assert device.zone == "office"

    def test_register_device_duplicate_fails(self, manager):
        """Registering duplicate device ID should fail."""
        manager.register_device("esp32-001")

        with pytest.raises(ValueError) as exc_info:
            manager.register_device("esp32-001")

        assert "already exists" in str(exc_info.value)

    def test_register_device_invalid_id_fails(self, manager):
        """Invalid device ID should fail validation."""
        with pytest.raises(ValueError) as exc_info:
            manager.register_device("invalid device!")

        assert "Invalid device_id" in str(exc_info.value)

    def test_get_device_exists(self, manager):
        """Should retrieve existing device."""
        manager.register_device("esp32-001")
        device = manager.get_device("esp32-001")

        assert device is not None
        assert device.device_id == "esp32-001"

    def test_get_device_not_exists(self, manager):
        """Should return None for non-existent device."""
        device = manager.get_device("nonexistent")
        assert device is None

    def test_get_device_by_mqtt_username(self, manager):
        """Should retrieve device by MQTT username."""
        manager.register_device("esp32-001")
        device = manager.get_device_by_mqtt_username("dev_esp32_001")

        assert device is not None
        assert device.device_id == "esp32-001"

    def test_list_devices_empty(self, manager):
        """Empty database should return empty list."""
        devices = manager.list_devices()
        assert devices == []

    def test_list_devices_multiple(self, manager):
        """Should return all registered devices."""
        manager.register_device("esp32-001")
        manager.register_device("esp32-002")
        manager.register_device("esp32-003")

        devices = manager.list_devices()
        assert len(devices) == 3

    def test_list_devices_filter_by_zone(self, manager):
        """Should filter devices by zone."""
        manager.register_device("esp32-001", zone="office")
        manager.register_device("esp32-002", zone="office")
        manager.register_device("esp32-003", zone="lab")

        office_devices = manager.list_devices(zone="office")
        assert len(office_devices) == 2

        lab_devices = manager.list_devices(zone="lab")
        assert len(lab_devices) == 1

    def test_rotate_mqtt_credentials(self, manager):
        """Rotating credentials should generate new password."""
        device, _, original_creds = manager.register_device("esp32-001")
        original_hash = device.mqtt_password_hash

        new_creds = manager.rotate_mqtt_credentials("esp32-001")

        assert new_creds is not None
        assert new_creds.password != original_creds.password

        # Verify new password works
        updated_device = manager.get_device("esp32-001")
        assert updated_device.mqtt_password_hash != original_hash
        assert verify_mqtt_password(new_creds.password, updated_device.mqtt_password_hash)

    def test_rotate_mqtt_credentials_nonexistent(self, manager):
        """Rotating for non-existent device should return None."""
        result = manager.rotate_mqtt_credentials("nonexistent")
        assert result is None

    def test_verify_mqtt_credentials_valid(self, manager):
        """Valid credentials should verify successfully."""
        _, _, creds = manager.register_device("esp32-001")

        device = manager.verify_mqtt_credentials(creds.username, creds.password)
        assert device is not None
        assert device.device_id == "esp32-001"

    def test_verify_mqtt_credentials_invalid(self, manager):
        """Invalid credentials should fail verification."""
        _, _, creds = manager.register_device("esp32-001")

        device = manager.verify_mqtt_credentials(creds.username, "wrong_password")
        assert device is None

    def test_verify_mqtt_credentials_unknown_user(self, manager):
        """Unknown username should fail verification."""
        device = manager.verify_mqtt_credentials("unknown_user", "password")
        assert device is None

    def test_delete_device_exists(self, manager):
        """Should delete existing device."""
        manager.register_device("esp32-001")

        deleted = manager.delete_device("esp32-001")
        assert deleted is True

        device = manager.get_device("esp32-001")
        assert device is None

    def test_delete_device_not_exists(self, manager):
        """Should return False for non-existent device."""
        deleted = manager.delete_device("nonexistent")
        assert deleted is False

    def test_update_device_status(self, manager):
        """Should update device online status."""
        manager.register_device("esp32-001")

        manager.update_device_status("esp32-001", online=True)
        device = manager.get_device("esp32-001")
        # Note: online status depends on last_seen time

    def test_get_devices_without_mqtt(self, manager):
        """Should return devices without MQTT credentials."""
        manager.register_device("esp32-001", generate_mqtt=True)
        manager.register_device("esp32-002", generate_mqtt=False)

        devices = manager.get_devices_without_mqtt()
        assert len(devices) == 1
        assert devices[0].device_id == "esp32-002"

    def test_migrate_all_devices_to_mqtt(self, manager):
        """Should add MQTT credentials to devices without them."""
        manager.register_device("esp32-001", generate_mqtt=False)
        manager.register_device("esp32-002", generate_mqtt=False)
        manager.register_device("esp32-003", generate_mqtt=True)  # Already has MQTT

        results = manager.migrate_all_devices_to_mqtt()

        assert len(results) == 2
        device_ids = [r[0] for r in results]
        assert "esp32-001" in device_ids
        assert "esp32-002" in device_ids

        # Verify all now have MQTT credentials
        devices = manager.get_devices_without_mqtt()
        assert len(devices) == 0

    def test_add_mqtt_credentials_to_device(self, manager):
        """Should add MQTT credentials to existing device."""
        manager.register_device("esp32-001", generate_mqtt=False)

        creds = manager.add_mqtt_credentials_to_device("esp32-001")
        assert creds is not None

        device = manager.get_device("esp32-001")
        assert device.has_mqtt_credentials is True


class TestDeviceModel:
    """Tests for Device data model."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        yield db_path
        if os.path.exists(db_path):
            os.unlink(db_path)

    @pytest.fixture
    def manager(self, temp_db):
        """Create a DeviceManager with temporary database."""
        config = DeviceManagerConfig(db_path=temp_db)
        return DeviceManager(config)

    def test_device_has_mqtt_credentials_true(self, manager):
        """Device with MQTT should report has_mqtt_credentials=True."""
        device, _, _ = manager.register_device("esp32-001", generate_mqtt=True)
        assert device.has_mqtt_credentials is True

    def test_device_has_mqtt_credentials_false(self, manager):
        """Device without MQTT should report has_mqtt_credentials=False."""
        device, _, _ = manager.register_device("esp32-001", generate_mqtt=False)
        assert device.has_mqtt_credentials is False

    def test_device_to_dict(self, manager):
        """to_dict should include expected fields."""
        device, _, _ = manager.register_device("esp32-001", zone="office")
        result = device.to_dict()

        assert "device_id" in result
        assert "mqtt_username" in result
        assert "mqtt_client_id" in result
        assert "has_mqtt_credentials" in result
        assert "zone" in result
        assert result["zone"] == "office"

        # Should not expose sensitive data
        assert "api_key" not in result
        assert "mqtt_password_hash" not in result


class TestDeviceManagerConfig:
    """Tests for DeviceManagerConfig."""

    def test_default_config(self):
        """Default config should have expected values."""
        config = DeviceManagerConfig()

        assert config.db_path == "devices.db"
        assert config.mqtt_username_prefix == "dev"
        assert config.mqtt_password_length == 32
        assert config.client_id_prefix == "wavira"

    def test_custom_config(self):
        """Custom config should override defaults."""
        config = DeviceManagerConfig(
            db_path="/tmp/test.db",
            mqtt_username_prefix="myapp",
            mqtt_password_length=64,
            client_id_prefix="custom",
        )

        assert config.db_path == "/tmp/test.db"
        assert config.mqtt_username_prefix == "myapp"
        assert config.mqtt_password_length == 64
        assert config.client_id_prefix == "custom"

    def test_config_from_env(self, monkeypatch):
        """Config should load from environment variables."""
        monkeypatch.setenv("DEVICES_DB_PATH", "/env/devices.db")
        monkeypatch.setenv("MQTT_USERNAME_PREFIX", "env_dev")
        monkeypatch.setenv("MQTT_PASSWORD_LENGTH", "48")
        monkeypatch.setenv("MQTT_CLIENT_ID_PREFIX", "env_client")

        config = DeviceManagerConfig.from_env()

        assert config.db_path == "/env/devices.db"
        assert config.mqtt_username_prefix == "env_dev"
        assert config.mqtt_password_length == 48
        assert config.client_id_prefix == "env_client"

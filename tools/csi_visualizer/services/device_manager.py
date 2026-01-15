#!/usr/bin/env python3
"""
Device Manager for Wavira MQTT Authentication.

Issue #23: デバイス認証のMQTT対応

This module manages device registration and MQTT credentials:
- Device registration with unique device IDs
- MQTT credential generation (username/password)
- Password hashing with bcrypt
- Credential rotation support
- Database migration for existing devices
"""

import hashlib
import hmac
import os
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import bcrypt

from security import validate_device_id


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class DeviceManagerConfig:
    """Device manager configuration."""
    db_path: str = "devices.db"
    mqtt_username_prefix: str = "dev"
    mqtt_password_length: int = 32
    client_id_prefix: str = "wavira"

    @classmethod
    def from_env(cls) -> "DeviceManagerConfig":
        """Load configuration from environment variables."""
        return cls(
            db_path=os.environ.get("DEVICES_DB_PATH", "devices.db"),
            mqtt_username_prefix=os.environ.get("MQTT_USERNAME_PREFIX", "dev"),
            mqtt_password_length=int(os.environ.get("MQTT_PASSWORD_LENGTH", "32")),
            client_id_prefix=os.environ.get("MQTT_CLIENT_ID_PREFIX", "wavira"),
        )


# =============================================================================
# MQTT Credentials
# =============================================================================


@dataclass
class MQTTCredentials:
    """MQTT connection credentials for a device."""
    username: str
    password: str  # Plain text, only returned once at generation
    client_id: str
    password_hash: str  # bcrypt hash for storage

    def to_dict(self, include_password: bool = False) -> Dict:
        """Convert to dictionary for API response."""
        result = {
            "username": self.username,
            "client_id": self.client_id,
        }
        if include_password:
            result["password"] = self.password
        return result


def generate_mqtt_password(length: int = 32) -> str:
    """Generate a secure random password for MQTT."""
    return secrets.token_urlsafe(length)


def hash_mqtt_password(password: str) -> str:
    """Hash MQTT password using bcrypt."""
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(password.encode(), salt).decode()


def verify_mqtt_password(password: str, password_hash: str) -> bool:
    """Verify MQTT password against bcrypt hash."""
    try:
        return bcrypt.checkpw(password.encode(), password_hash.encode())
    except (ValueError, TypeError):
        return False


def generate_mqtt_credentials(
    device_id: str,
    config: DeviceManagerConfig,
) -> MQTTCredentials:
    """Generate MQTT credentials for a device."""
    # Generate username: prefix_deviceid (normalized)
    normalized_id = device_id.replace("-", "_").lower()
    username = f"{config.mqtt_username_prefix}_{normalized_id}"

    # Generate secure password
    password = generate_mqtt_password(config.mqtt_password_length)
    password_hash = hash_mqtt_password(password)

    # Generate client ID
    client_id = f"{config.client_id_prefix}_{normalized_id}"

    return MQTTCredentials(
        username=username,
        password=password,
        client_id=client_id,
        password_hash=password_hash,
    )


# =============================================================================
# Device Data Models
# =============================================================================


@dataclass
class Device:
    """Device information with MQTT credentials."""
    device_id: str
    api_key: str
    api_key_hash: str
    mqtt_username: Optional[str] = None
    mqtt_password_hash: Optional[str] = None
    mqtt_client_id: Optional[str] = None
    last_seen: Optional[float] = None
    online: bool = False
    zone: str = "default"
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    @property
    def has_mqtt_credentials(self) -> bool:
        """Check if device has MQTT credentials."""
        return all([
            self.mqtt_username,
            self.mqtt_password_hash,
            self.mqtt_client_id,
        ])

    def to_dict(self) -> Dict:
        """Convert to dictionary for API response."""
        return {
            "device_id": self.device_id,
            "mqtt_username": self.mqtt_username,
            "mqtt_client_id": self.mqtt_client_id,
            "has_mqtt_credentials": self.has_mqtt_credentials,
            "last_seen": self.last_seen,
            "online": self.online,
            "zone": self.zone,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


# =============================================================================
# Device Manager
# =============================================================================


class DeviceManager:
    """
    Manages device registration and MQTT credentials.

    Thread-safe database operations using SQLite.
    """

    def __init__(self, config: Optional[DeviceManagerConfig] = None):
        self.config = config or DeviceManagerConfig.from_env()
        self.lock = threading.Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get a new database connection."""
        conn = sqlite3.connect(self.config.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initialize database schema with MQTT columns."""
        with self.lock:
            conn = self._get_conn()
            cursor = conn.cursor()

            # Create devices table with MQTT columns
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS devices (
                    device_id TEXT PRIMARY KEY,
                    api_key_hash TEXT NOT NULL,
                    mqtt_username TEXT,
                    mqtt_password_hash TEXT,
                    mqtt_client_id TEXT,
                    last_seen REAL,
                    online INTEGER DEFAULT 0,
                    zone TEXT DEFAULT 'default',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create index on mqtt_username for authentication lookups
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_devices_mqtt_username
                ON devices(mqtt_username)
            ''')

            # Migration: Add MQTT columns if they don't exist (for existing databases)
            try:
                cursor.execute('ALTER TABLE devices ADD COLUMN mqtt_username TEXT')
            except sqlite3.OperationalError:
                pass  # Column already exists

            try:
                cursor.execute('ALTER TABLE devices ADD COLUMN mqtt_password_hash TEXT')
            except sqlite3.OperationalError:
                pass

            try:
                cursor.execute('ALTER TABLE devices ADD COLUMN mqtt_client_id TEXT')
            except sqlite3.OperationalError:
                pass

            conn.commit()
            conn.close()

    def _generate_api_key(self) -> Tuple[str, str]:
        """Generate API key and its hash."""
        from security import generate_api_key, hash_api_key
        api_key = generate_api_key(prefix="wvr")
        api_key_hash = hash_api_key(api_key)
        return api_key, api_key_hash

    def register_device(
        self,
        device_id: str,
        zone: str = "default",
        generate_mqtt: bool = True,
    ) -> Tuple[Device, str, Optional[MQTTCredentials]]:
        """
        Register a new device.

        Args:
            device_id: Unique device identifier
            zone: Zone/location of the device
            generate_mqtt: Whether to generate MQTT credentials

        Returns:
            Tuple of (Device, api_key, MQTTCredentials or None)

        Raises:
            ValueError: If device_id is invalid or already exists
        """
        if not validate_device_id(device_id):
            raise ValueError(f"Invalid device_id: {device_id}")

        api_key, api_key_hash = self._generate_api_key()
        mqtt_creds = None

        if generate_mqtt:
            mqtt_creds = generate_mqtt_credentials(device_id, self.config)

        with self.lock:
            conn = self._get_conn()
            cursor = conn.cursor()

            # Check if device already exists
            cursor.execute(
                'SELECT device_id FROM devices WHERE device_id = ?',
                (device_id,)
            )
            if cursor.fetchone():
                conn.close()
                raise ValueError(f"Device already exists: {device_id}")

            # Insert new device
            cursor.execute('''
                INSERT INTO devices (
                    device_id, api_key_hash, mqtt_username, mqtt_password_hash,
                    mqtt_client_id, zone, online
                )
                VALUES (?, ?, ?, ?, ?, ?, 0)
            ''', (
                device_id,
                api_key_hash,
                mqtt_creds.username if mqtt_creds else None,
                mqtt_creds.password_hash if mqtt_creds else None,
                mqtt_creds.client_id if mqtt_creds else None,
                zone,
            ))

            conn.commit()

            # Fetch the created device
            cursor.execute('SELECT * FROM devices WHERE device_id = ?', (device_id,))
            row = cursor.fetchone()
            conn.close()

            device = self._row_to_device(row, api_key_hash)
            return device, api_key, mqtt_creds

    def get_device(self, device_id: str) -> Optional[Device]:
        """Get device by ID."""
        with self.lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM devices WHERE device_id = ?', (device_id,))
            row = cursor.fetchone()
            conn.close()

            if row:
                return self._row_to_device(row)
            return None

    def get_device_by_mqtt_username(self, mqtt_username: str) -> Optional[Device]:
        """Get device by MQTT username for authentication."""
        with self.lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute(
                'SELECT * FROM devices WHERE mqtt_username = ?',
                (mqtt_username,)
            )
            row = cursor.fetchone()
            conn.close()

            if row:
                return self._row_to_device(row)
            return None

    def list_devices(
        self,
        zone: Optional[str] = None,
        online_only: bool = False,
    ) -> List[Device]:
        """List all devices with optional filtering."""
        with self.lock:
            conn = self._get_conn()
            cursor = conn.cursor()

            query = 'SELECT * FROM devices WHERE 1=1'
            params = []

            if zone:
                query += ' AND zone = ?'
                params.append(zone)

            if online_only:
                # Consider device online if seen within last 60 seconds
                cutoff = time.time() - 60
                query += ' AND last_seen > ?'
                params.append(cutoff)

            query += ' ORDER BY last_seen DESC'
            cursor.execute(query, params)

            rows = cursor.fetchall()
            conn.close()

            return [self._row_to_device(row) for row in rows]

    def rotate_mqtt_credentials(self, device_id: str) -> Optional[MQTTCredentials]:
        """
        Rotate MQTT credentials for a device.

        Returns new credentials or None if device not found.
        """
        if not validate_device_id(device_id):
            raise ValueError(f"Invalid device_id: {device_id}")

        mqtt_creds = generate_mqtt_credentials(device_id, self.config)

        with self.lock:
            conn = self._get_conn()
            cursor = conn.cursor()

            cursor.execute('''
                UPDATE devices SET
                    mqtt_username = ?,
                    mqtt_password_hash = ?,
                    mqtt_client_id = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE device_id = ?
            ''', (
                mqtt_creds.username,
                mqtt_creds.password_hash,
                mqtt_creds.client_id,
                device_id,
            ))

            if cursor.rowcount == 0:
                conn.close()
                return None

            conn.commit()
            conn.close()
            return mqtt_creds

    def add_mqtt_credentials_to_device(self, device_id: str) -> Optional[MQTTCredentials]:
        """
        Add MQTT credentials to an existing device that doesn't have them.

        Returns new credentials or None if device not found.
        """
        return self.rotate_mqtt_credentials(device_id)

    def update_device_status(
        self,
        device_id: str,
        online: bool,
        last_seen: Optional[float] = None,
    ):
        """Update device online status."""
        with self.lock:
            conn = self._get_conn()
            cursor = conn.cursor()

            cursor.execute('''
                UPDATE devices SET
                    online = ?,
                    last_seen = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE device_id = ?
            ''', (
                1 if online else 0,
                last_seen or time.time(),
                device_id,
            ))

            conn.commit()
            conn.close()

    def verify_mqtt_credentials(
        self,
        username: str,
        password: str,
    ) -> Optional[Device]:
        """
        Verify MQTT credentials for authentication.

        Returns Device if valid, None otherwise.
        """
        device = self.get_device_by_mqtt_username(username)
        if not device or not device.mqtt_password_hash:
            return None

        if verify_mqtt_password(password, device.mqtt_password_hash):
            return device
        return None

    def delete_device(self, device_id: str) -> bool:
        """Delete a device. Returns True if deleted, False if not found."""
        with self.lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute('DELETE FROM devices WHERE device_id = ?', (device_id,))
            deleted = cursor.rowcount > 0
            conn.commit()
            conn.close()
            return deleted

    def get_devices_without_mqtt(self) -> List[Device]:
        """Get devices that don't have MQTT credentials (for migration)."""
        with self.lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM devices
                WHERE mqtt_username IS NULL OR mqtt_password_hash IS NULL
                ORDER BY created_at
            ''')
            rows = cursor.fetchall()
            conn.close()
            return [self._row_to_device(row) for row in rows]

    def migrate_all_devices_to_mqtt(self) -> List[Tuple[str, MQTTCredentials]]:
        """
        Add MQTT credentials to all devices that don't have them.

        Returns list of (device_id, credentials) tuples.
        """
        devices = self.get_devices_without_mqtt()
        results = []

        for device in devices:
            creds = self.add_mqtt_credentials_to_device(device.device_id)
            if creds:
                results.append((device.device_id, creds))

        return results

    def _row_to_device(
        self,
        row: sqlite3.Row,
        api_key_hash: Optional[str] = None,
    ) -> Device:
        """Convert database row to Device object."""
        # Determine online status based on last_seen
        last_seen = row["last_seen"]
        online = False
        if last_seen:
            online = time.time() - last_seen < 60

        return Device(
            device_id=row["device_id"],
            api_key="",  # Never expose
            api_key_hash=api_key_hash or row["api_key_hash"],
            mqtt_username=row["mqtt_username"],
            mqtt_password_hash=row["mqtt_password_hash"],
            mqtt_client_id=row["mqtt_client_id"],
            last_seen=last_seen,
            online=online,
            zone=row["zone"] or "default",
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


# =============================================================================
# Mosquitto ACL Generator
# =============================================================================


def generate_mosquitto_password_file(
    manager: DeviceManager,
    output_path: str = "mosquitto_passwd",
) -> int:
    """
    Generate Mosquitto password file from device credentials.

    Note: This generates a file in Mosquitto's password format.
    The passwords are already hashed with bcrypt.

    Returns: Number of entries written
    """
    devices = manager.list_devices()
    count = 0

    with open(output_path, "w") as f:
        for device in devices:
            if device.mqtt_username and device.mqtt_password_hash:
                # Mosquitto format: username:password_hash
                f.write(f"{device.mqtt_username}:{device.mqtt_password_hash}\n")
                count += 1

    return count


def generate_mosquitto_acl_file(
    manager: DeviceManager,
    output_path: str = "mosquitto_acl.conf",
) -> int:
    """
    Generate Mosquitto ACL file for device access control.

    Returns: Number of entries written
    """
    devices = manager.list_devices()
    count = 0

    with open(output_path, "w") as f:
        f.write("# Wavira MQTT ACL Configuration\n")
        f.write("# Auto-generated by device_manager.py\n\n")

        for device in devices:
            if device.mqtt_username:
                normalized_id = device.device_id.replace("-", "_").lower()
                f.write(f"# Device: {device.device_id}\n")
                f.write(f"user {device.mqtt_username}\n")
                # Allow publish to device-specific topics
                f.write(f"topic write wavira/csi/{device.device_id}\n")
                f.write(f"topic write wavira/device/{device.device_id}/#\n")
                # Allow subscribe to control topics
                f.write(f"topic read wavira/control/{device.device_id}/#\n")
                f.write("\n")
                count += 1

        # Server user with full access
        f.write("# Server user - full access\n")
        f.write("user wavira_server\n")
        f.write("topic readwrite wavira/#\n")

    return count


# =============================================================================
# CLI Interface
# =============================================================================


def main():
    """CLI interface for device management."""
    import argparse

    parser = argparse.ArgumentParser(description="Wavira Device Manager")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Register command
    register_parser = subparsers.add_parser("register", help="Register a new device")
    register_parser.add_argument("device_id", help="Device ID")
    register_parser.add_argument("--zone", default="default", help="Device zone")
    register_parser.add_argument(
        "--no-mqtt", action="store_true", help="Don't generate MQTT credentials"
    )

    # List command
    list_parser = subparsers.add_parser("list", help="List devices")
    list_parser.add_argument("--zone", help="Filter by zone")
    list_parser.add_argument("--online", action="store_true", help="Only online devices")

    # Rotate command
    rotate_parser = subparsers.add_parser("rotate", help="Rotate MQTT credentials")
    rotate_parser.add_argument("device_id", help="Device ID")

    # Migrate command
    migrate_parser = subparsers.add_parser(
        "migrate", help="Add MQTT credentials to devices without them"
    )

    # Generate Mosquitto files
    mosquitto_parser = subparsers.add_parser(
        "mosquitto", help="Generate Mosquitto config files"
    )
    mosquitto_parser.add_argument(
        "--passwd-file", default="mosquitto_passwd", help="Password file path"
    )
    mosquitto_parser.add_argument(
        "--acl-file", default="mosquitto_acl.conf", help="ACL file path"
    )

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a device")
    delete_parser.add_argument("device_id", help="Device ID")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    config = DeviceManagerConfig.from_env()
    manager = DeviceManager(config)

    if args.command == "register":
        try:
            device, api_key, mqtt_creds = manager.register_device(
                args.device_id,
                zone=args.zone,
                generate_mqtt=not args.no_mqtt,
            )
            print(f"Device registered: {device.device_id}")
            print(f"  API Key: {api_key}")
            if mqtt_creds:
                print(f"  MQTT Username: {mqtt_creds.username}")
                print(f"  MQTT Password: {mqtt_creds.password}")
                print(f"  MQTT Client ID: {mqtt_creds.client_id}")
            print("\nIMPORTANT: Save these credentials securely. "
                  "The password cannot be retrieved later.")
        except ValueError as e:
            print(f"Error: {e}")

    elif args.command == "list":
        devices = manager.list_devices(zone=args.zone, online_only=args.online)
        if not devices:
            print("No devices found")
            return

        print(f"{'Device ID':<20} {'MQTT User':<25} {'Zone':<10} {'Online':<8}")
        print("-" * 70)
        for device in devices:
            mqtt_user = device.mqtt_username or "(none)"
            online = "Yes" if device.online else "No"
            print(f"{device.device_id:<20} {mqtt_user:<25} {device.zone:<10} {online:<8}")

    elif args.command == "rotate":
        creds = manager.rotate_mqtt_credentials(args.device_id)
        if creds:
            print(f"MQTT credentials rotated for: {args.device_id}")
            print(f"  MQTT Username: {creds.username}")
            print(f"  MQTT Password: {creds.password}")
            print(f"  MQTT Client ID: {creds.client_id}")
            print("\nIMPORTANT: Update device firmware with new credentials.")
        else:
            print(f"Device not found: {args.device_id}")

    elif args.command == "migrate":
        results = manager.migrate_all_devices_to_mqtt()
        if not results:
            print("No devices need migration")
            return

        print(f"Migrated {len(results)} devices:")
        for device_id, creds in results:
            print(f"\n{device_id}:")
            print(f"  MQTT Username: {creds.username}")
            print(f"  MQTT Password: {creds.password}")
            print(f"  MQTT Client ID: {creds.client_id}")

    elif args.command == "mosquitto":
        passwd_count = generate_mosquitto_password_file(manager, args.passwd_file)
        acl_count = generate_mosquitto_acl_file(manager, args.acl_file)
        print(f"Generated {args.passwd_file} with {passwd_count} entries")
        print(f"Generated {args.acl_file} with {acl_count} entries")

    elif args.command == "delete":
        if manager.delete_device(args.device_id):
            print(f"Device deleted: {args.device_id}")
        else:
            print(f"Device not found: {args.device_id}")


if __name__ == "__main__":
    main()

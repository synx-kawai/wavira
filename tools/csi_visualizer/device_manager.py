#!/usr/bin/env python3
"""
Device Management System for Wavira CSI Platform

ESP32デバイスの認証・管理システム。
Issue #16: ESP32デバイス認証・設定管理システムの実装

Features:
- Device registration with API key generation
- API key hashing with bcrypt for secure storage
- Device status monitoring (online/offline detection)
- Device metadata management (zone, location)
"""

import hashlib
import logging
import secrets
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# Try to import bcrypt, fall back to hashlib if not available
try:
    import bcrypt
    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False
    logging.warning("bcrypt not available, using SHA256 for API key hashing (less secure)")

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class Device:
    """デバイス情報"""
    id: str
    api_key_hash: str
    zone: Optional[str] = None
    location: Optional[str] = None
    enabled: bool = True
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class DeviceStatus:
    """デバイス状態"""
    device_id: str
    status: str  # "online", "offline", "unknown"
    last_seen: Optional[datetime] = None
    packet_count: int = 0
    error_count: int = 0
    firmware_version: Optional[str] = None
    uptime_seconds: Optional[int] = None


@dataclass
class DeviceWithStatus:
    """デバイス情報と状態"""
    device: Device
    status: DeviceStatus


# =============================================================================
# API Key Utilities
# =============================================================================


def generate_api_key(prefix: str = "wvr") -> str:
    """
    Generate a secure API key.

    Format: {prefix}_{random_32_bytes_hex}
    Example: wvr_a1b2c3d4e5f6...
    """
    random_bytes = secrets.token_hex(32)
    return f"{prefix}_{random_bytes}"


def hash_api_key(api_key: str) -> str:
    """
    Hash an API key for secure storage.

    Uses bcrypt if available, otherwise SHA256.
    """
    if BCRYPT_AVAILABLE:
        # bcrypt handles salt internally
        return bcrypt.hashpw(api_key.encode(), bcrypt.gensalt()).decode()
    else:
        # Fallback to SHA256 with salt
        salt = secrets.token_hex(16)
        hashed = hashlib.sha256((salt + api_key).encode()).hexdigest()
        return f"sha256:{salt}:{hashed}"


def verify_api_key(api_key: str, hashed: str) -> bool:
    """
    Verify an API key against its hash.
    """
    if BCRYPT_AVAILABLE and not hashed.startswith("sha256:"):
        try:
            return bcrypt.checkpw(api_key.encode(), hashed.encode())
        except Exception:
            return False
    elif hashed.startswith("sha256:"):
        # SHA256 fallback verification
        parts = hashed.split(":")
        if len(parts) != 3:
            return False
        _, salt, stored_hash = parts
        computed_hash = hashlib.sha256((salt + api_key).encode()).hexdigest()
        return secrets.compare_digest(computed_hash, stored_hash)
    return False


def create_lookup_hash(api_key: str) -> str:
    """
    Create a fast lookup hash for API key indexing.

    Uses SHA256 without salt for consistent, indexed lookups.
    This is NOT for security verification - use verify_api_key for that.
    """
    return hashlib.sha256(api_key.encode()).hexdigest()


# =============================================================================
# Device Manager
# =============================================================================


class DeviceAuthManager:
    """
    デバイス認証・管理システム

    SQLiteデータベースでデバイス情報を管理し、
    APIキーによる認証機能を提供。
    """

    # オフライン判定の閾値（秒）
    OFFLINE_THRESHOLD_SECONDS = 60

    def __init__(self, db_path: str = "devices.db"):
        self.db_path = db_path
        self._lock = threading.RLock()  # RLock for re-entrant calls
        # For in-memory databases, we need to keep a persistent connection
        # otherwise each connect() creates a new empty database
        self._persistent_conn: Optional[sqlite3.Connection] = None
        if db_path == ":memory:":
            self._persistent_conn = sqlite3.connect(":memory:")
            self._persistent_conn.row_factory = sqlite3.Row
        self._init_database()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with row factory."""
        if self._persistent_conn is not None:
            return self._persistent_conn
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _close_connection(self, conn: sqlite3.Connection) -> None:
        """Close a database connection if not using persistent connection."""
        if self._persistent_conn is None:
            conn.close()

    def _init_database(self):
        """Initialize database schema."""
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()

                # デバイス管理テーブル
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS devices (
                        id TEXT PRIMARY KEY,
                        api_key_hash TEXT NOT NULL,
                        api_key_lookup TEXT NOT NULL,
                        zone TEXT,
                        location TEXT,
                        enabled BOOLEAN DEFAULT TRUE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # デバイス状態テーブル
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS device_status (
                        device_id TEXT PRIMARY KEY REFERENCES devices(id) ON DELETE CASCADE,
                        last_seen TIMESTAMP,
                        packet_count INTEGER DEFAULT 0,
                        error_count INTEGER DEFAULT 0,
                        firmware_version TEXT,
                        uptime_seconds INTEGER
                    )
                """)

                # インデックス作成
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_devices_zone ON devices(zone)
                """)
                cursor.execute("""
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_devices_api_key_lookup
                    ON devices(api_key_lookup)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_device_status_last_seen
                    ON device_status(last_seen)
                """)

                conn.commit()
                logger.info(f"Device database initialized: {self.db_path}")
            finally:
                self._close_connection(conn)

    # -------------------------------------------------------------------------
    # Device CRUD Operations
    # -------------------------------------------------------------------------

    def register_device(
        self,
        device_id: str,
        zone: Optional[str] = None,
        location: Optional[str] = None,
    ) -> tuple[Device, str]:
        """
        Register a new device and generate its API key.

        Returns:
            Tuple of (Device, plain_api_key)
        """
        # Generate API key
        api_key = generate_api_key()
        api_key_hash = hash_api_key(api_key)
        api_key_lookup = create_lookup_hash(api_key)

        now = datetime.utcnow()

        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()

                # Check if device already exists
                cursor.execute("SELECT id FROM devices WHERE id = ?", (device_id,))
                if cursor.fetchone():
                    raise ValueError(f"Device '{device_id}' already exists")

                # Insert device
                cursor.execute("""
                    INSERT INTO devices (id, api_key_hash, api_key_lookup, zone, location, enabled, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, TRUE, ?, ?)
                """, (device_id, api_key_hash, api_key_lookup, zone, location, now, now))

                # Initialize device status
                cursor.execute("""
                    INSERT INTO device_status (device_id, packet_count, error_count)
                    VALUES (?, 0, 0)
                """, (device_id,))

                conn.commit()

                device = Device(
                    id=device_id,
                    api_key_hash=api_key_hash,
                    zone=zone,
                    location=location,
                    enabled=True,
                    created_at=now,
                    updated_at=now,
                )

                logger.info(f"Device registered: {device_id}")
                return device, api_key
            finally:
                self._close_connection(conn)

    def get_device(self, device_id: str) -> Optional[Device]:
        """Get device by ID."""
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM devices WHERE id = ?", (device_id,))
                row = cursor.fetchone()
                if row:
                    return Device(
                        id=row["id"],
                        api_key_hash=row["api_key_hash"],
                        zone=row["zone"],
                        location=row["location"],
                        enabled=bool(row["enabled"]),
                        created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
                        updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None,
                    )
                return None
            finally:
                self._close_connection(conn)

    def list_devices(self, zone: Optional[str] = None, enabled_only: bool = False) -> List[Device]:
        """List all devices, optionally filtered by zone."""
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()

                query = "SELECT * FROM devices WHERE 1=1"
                params = []

                if zone:
                    query += " AND zone = ?"
                    params.append(zone)

                if enabled_only:
                    query += " AND enabled = TRUE"

                query += " ORDER BY created_at DESC"

                cursor.execute(query, params)

                devices = []
                for row in cursor.fetchall():
                    devices.append(Device(
                        id=row["id"],
                        api_key_hash=row["api_key_hash"],
                        zone=row["zone"],
                        location=row["location"],
                        enabled=bool(row["enabled"]),
                        created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
                        updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None,
                    ))
                return devices
            finally:
                self._close_connection(conn)

    def update_device(
        self,
        device_id: str,
        zone: Optional[str] = None,
        location: Optional[str] = None,
        enabled: Optional[bool] = None,
    ) -> Optional[Device]:
        """Update device information."""
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()

                # Build update query dynamically
                updates = []
                params = []

                if zone is not None:
                    updates.append("zone = ?")
                    params.append(zone)

                if location is not None:
                    updates.append("location = ?")
                    params.append(location)

                if enabled is not None:
                    updates.append("enabled = ?")
                    params.append(enabled)

                if not updates:
                    return self.get_device(device_id)

                updates.append("updated_at = ?")
                params.append(datetime.utcnow())
                params.append(device_id)

                query = f"UPDATE devices SET {', '.join(updates)} WHERE id = ?"
                cursor.execute(query, params)
                conn.commit()

                if cursor.rowcount == 0:
                    return None

                logger.info(f"Device updated: {device_id}")
                return self.get_device(device_id)
            finally:
                self._close_connection(conn)

    def delete_device(self, device_id: str) -> bool:
        """Delete a device."""
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM devices WHERE id = ?", (device_id,))
                conn.commit()

                deleted = cursor.rowcount > 0
                if deleted:
                    logger.info(f"Device deleted: {device_id}")
                return deleted
            finally:
                self._close_connection(conn)

    def rotate_api_key(self, device_id: str) -> Optional[str]:
        """
        Rotate (regenerate) the API key for a device.

        Returns:
            The new plain API key, or None if device not found.
        """
        api_key = generate_api_key()
        api_key_hash = hash_api_key(api_key)
        api_key_lookup = create_lookup_hash(api_key)

        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE devices
                    SET api_key_hash = ?, api_key_lookup = ?, updated_at = ?
                    WHERE id = ?
                """, (api_key_hash, api_key_lookup, datetime.utcnow(), device_id))
                conn.commit()

                if cursor.rowcount == 0:
                    return None

                logger.info(f"API key rotated for device: {device_id}")
                return api_key
            finally:
                self._close_connection(conn)

    # -------------------------------------------------------------------------
    # Authentication
    # -------------------------------------------------------------------------

    def authenticate(self, device_id: str, api_key: str) -> bool:
        """
        Authenticate a device with its API key.

        Returns:
            True if authentication successful, False otherwise.
        """
        device = self.get_device(device_id)
        if not device:
            return False

        if not device.enabled:
            logger.warning(f"Authentication failed: Device '{device_id}' is disabled")
            return False

        return verify_api_key(api_key, device.api_key_hash)

    def authenticate_by_key(self, api_key: str) -> Optional[str]:
        """
        Authenticate by API key only (find device by key).

        Uses indexed lookup hash for O(1) performance, then verifies with bcrypt.

        Returns:
            Device ID if found and authenticated, None otherwise.
        """
        # Create lookup hash for indexed search
        lookup_hash = create_lookup_hash(api_key)

        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                # O(1) indexed lookup by hash
                cursor.execute(
                    "SELECT id, api_key_hash, enabled FROM devices WHERE api_key_lookup = ?",
                    (lookup_hash,)
                )
                row = cursor.fetchone()

                if not row:
                    return None

                if not row["enabled"]:
                    logger.warning(f"Authentication failed: Device '{row['id']}' is disabled")
                    return None

                # Verify with bcrypt for security
                if verify_api_key(api_key, row["api_key_hash"]):
                    return row["id"]

                return None
            finally:
                self._close_connection(conn)

    # -------------------------------------------------------------------------
    # Device Status
    # -------------------------------------------------------------------------

    def update_device_status(
        self,
        device_id: str,
        packet_count_delta: int = 1,
        error_count_delta: int = 0,
        firmware_version: Optional[str] = None,
        uptime_seconds: Optional[int] = None,
    ):
        """Update device status after receiving data."""
        now = datetime.utcnow()

        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()

                # Upsert device status
                cursor.execute("""
                    INSERT INTO device_status (device_id, last_seen, packet_count, error_count, firmware_version, uptime_seconds)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(device_id) DO UPDATE SET
                        last_seen = excluded.last_seen,
                        packet_count = device_status.packet_count + ?,
                        error_count = device_status.error_count + ?,
                        firmware_version = COALESCE(excluded.firmware_version, device_status.firmware_version),
                        uptime_seconds = COALESCE(excluded.uptime_seconds, device_status.uptime_seconds)
                """, (
                    device_id, now, packet_count_delta, error_count_delta,
                    firmware_version, uptime_seconds,
                    packet_count_delta, error_count_delta
                ))

                conn.commit()
            finally:
                self._close_connection(conn)

    def get_device_status(self, device_id: str) -> Optional[DeviceStatus]:
        """Get device status."""
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM device_status WHERE device_id = ?", (device_id,))
                row = cursor.fetchone()

                if not row:
                    return None

                last_seen = datetime.fromisoformat(row["last_seen"]) if row["last_seen"] else None

                # Determine online/offline status
                if last_seen:
                    threshold = datetime.utcnow() - timedelta(seconds=self.OFFLINE_THRESHOLD_SECONDS)
                    status = "online" if last_seen > threshold else "offline"
                else:
                    status = "unknown"

                return DeviceStatus(
                    device_id=row["device_id"],
                    status=status,
                    last_seen=last_seen,
                    packet_count=row["packet_count"] or 0,
                    error_count=row["error_count"] or 0,
                    firmware_version=row["firmware_version"],
                    uptime_seconds=row["uptime_seconds"],
                )
            finally:
                self._close_connection(conn)

    def get_all_device_statuses(self) -> List[DeviceWithStatus]:
        """Get all devices with their statuses."""
        devices = self.list_devices()
        result = []

        for device in devices:
            status = self.get_device_status(device.id)
            if not status:
                status = DeviceStatus(
                    device_id=device.id,
                    status="unknown",
                )
            result.append(DeviceWithStatus(device=device, status=status))

        return result

    def get_offline_devices(self) -> List[DeviceWithStatus]:
        """Get all devices that are currently offline."""
        all_devices = self.get_all_device_statuses()
        return [d for d in all_devices if d.status.status == "offline"]

    def get_online_device_count(self) -> int:
        """Get count of online devices."""
        all_devices = self.get_all_device_statuses()
        return sum(1 for d in all_devices if d.status.status == "online")

    def has_registered_devices(self) -> bool:
        """Check if there are any registered devices in the database.

        This is used to determine if device authentication should be enforced.
        If there are no registered devices and no static API keys, authentication
        can be disabled for development/testing purposes.
        """
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.execute("SELECT COUNT(*) FROM devices")
                count = cursor.fetchone()[0]
                return count > 0
            except Exception as e:
                logger.error(f"Error checking registered devices: {e}")
                return False
            finally:
                self._close_connection(conn)

    def close(self):
        """Close the database connection."""
        if self._persistent_conn is not None:
            self._persistent_conn.close()
            self._persistent_conn = None
        logger.info("Device manager closed")


# =============================================================================
# CLI Interface
# =============================================================================


def main():
    """CLI for device management."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Wavira Device Management CLI")
    parser.add_argument("--db", default="devices.db", help="Database path")

    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Register command
    register_parser = subparsers.add_parser("register", help="Register a new device")
    register_parser.add_argument("device_id", help="Device ID")
    register_parser.add_argument("--zone", help="Device zone")
    register_parser.add_argument("--location", help="Device location")

    # List command
    list_parser = subparsers.add_parser("list", help="List devices")
    list_parser.add_argument("--zone", help="Filter by zone")
    list_parser.add_argument("--enabled-only", action="store_true", help="Show only enabled devices")
    list_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Get command
    get_parser = subparsers.add_parser("get", help="Get device details")
    get_parser.add_argument("device_id", help="Device ID")
    get_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Update command
    update_parser = subparsers.add_parser("update", help="Update device")
    update_parser.add_argument("device_id", help="Device ID")
    update_parser.add_argument("--zone", help="New zone")
    update_parser.add_argument("--location", help="New location")
    update_parser.add_argument("--enable", action="store_true", help="Enable device")
    update_parser.add_argument("--disable", action="store_true", help="Disable device")

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete device")
    delete_parser.add_argument("device_id", help="Device ID")
    delete_parser.add_argument("--force", action="store_true", help="Skip confirmation")

    # Rotate key command
    rotate_parser = subparsers.add_parser("rotate-key", help="Rotate API key")
    rotate_parser.add_argument("device_id", help="Device ID")

    # Status command
    status_parser = subparsers.add_parser("status", help="Show device status")
    status_parser.add_argument("device_id", nargs="?", help="Device ID (optional, shows all if not specified)")
    status_parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    manager = DeviceAuthManager(args.db)

    try:
        if args.command == "register":
            device, api_key = manager.register_device(
                args.device_id,
                zone=args.zone,
                location=args.location,
            )
            print(f"Device registered successfully!")
            print(f"  Device ID: {device.id}")
            print(f"  Zone: {device.zone or 'N/A'}")
            print(f"  Location: {device.location or 'N/A'}")
            print(f"\n⚠️  IMPORTANT: Save this API key - it will not be shown again!")
            print(f"  API Key: {api_key}")

        elif args.command == "list":
            devices = manager.list_devices(
                zone=args.zone,
                enabled_only=args.enabled_only,
            )

            if args.json:
                output = [
                    {
                        "id": d.id,
                        "zone": d.zone,
                        "location": d.location,
                        "enabled": d.enabled,
                        "created_at": d.created_at.isoformat() if d.created_at else None,
                    }
                    for d in devices
                ]
                print(json.dumps(output, indent=2))
            else:
                if not devices:
                    print("No devices found.")
                else:
                    print(f"{'ID':<20} {'Zone':<15} {'Location':<20} {'Enabled':<10}")
                    print("-" * 65)
                    for d in devices:
                        print(f"{d.id:<20} {d.zone or 'N/A':<15} {d.location or 'N/A':<20} {'Yes' if d.enabled else 'No':<10}")

        elif args.command == "get":
            device = manager.get_device(args.device_id)
            status = manager.get_device_status(args.device_id)

            if not device:
                print(f"Device '{args.device_id}' not found.")
                return

            if args.json:
                output = {
                    "id": device.id,
                    "zone": device.zone,
                    "location": device.location,
                    "enabled": device.enabled,
                    "created_at": device.created_at.isoformat() if device.created_at else None,
                    "updated_at": device.updated_at.isoformat() if device.updated_at else None,
                    "status": status.status if status else "unknown",
                    "last_seen": status.last_seen.isoformat() if status and status.last_seen else None,
                    "packet_count": status.packet_count if status else 0,
                    "error_count": status.error_count if status else 0,
                }
                print(json.dumps(output, indent=2))
            else:
                print(f"Device: {device.id}")
                print(f"  Zone: {device.zone or 'N/A'}")
                print(f"  Location: {device.location or 'N/A'}")
                print(f"  Enabled: {'Yes' if device.enabled else 'No'}")
                print(f"  Created: {device.created_at}")
                print(f"  Updated: {device.updated_at}")
                if status:
                    print(f"  Status: {status.status}")
                    print(f"  Last Seen: {status.last_seen or 'Never'}")
                    print(f"  Packets: {status.packet_count}")
                    print(f"  Errors: {status.error_count}")

        elif args.command == "update":
            enabled = None
            if args.enable:
                enabled = True
            elif args.disable:
                enabled = False

            device = manager.update_device(
                args.device_id,
                zone=args.zone,
                location=args.location,
                enabled=enabled,
            )

            if device:
                print(f"Device '{args.device_id}' updated successfully.")
            else:
                print(f"Device '{args.device_id}' not found.")

        elif args.command == "delete":
            if not args.force:
                confirm = input(f"Delete device '{args.device_id}'? (y/N): ")
                if confirm.lower() != "y":
                    print("Cancelled.")
                    return

            if manager.delete_device(args.device_id):
                print(f"Device '{args.device_id}' deleted.")
            else:
                print(f"Device '{args.device_id}' not found.")

        elif args.command == "rotate-key":
            api_key = manager.rotate_api_key(args.device_id)
            if api_key:
                print(f"API key rotated for device '{args.device_id}'")
                print(f"\n⚠️  IMPORTANT: Save this API key - it will not be shown again!")
                print(f"  New API Key: {api_key}")
            else:
                print(f"Device '{args.device_id}' not found.")

        elif args.command == "status":
            if args.device_id:
                status = manager.get_device_status(args.device_id)
                if not status:
                    print(f"No status for device '{args.device_id}'")
                    return

                if args.json:
                    output = {
                        "device_id": status.device_id,
                        "status": status.status,
                        "last_seen": status.last_seen.isoformat() if status.last_seen else None,
                        "packet_count": status.packet_count,
                        "error_count": status.error_count,
                        "firmware_version": status.firmware_version,
                        "uptime_seconds": status.uptime_seconds,
                    }
                    print(json.dumps(output, indent=2))
                else:
                    print(f"Device: {status.device_id}")
                    print(f"  Status: {status.status}")
                    print(f"  Last Seen: {status.last_seen or 'Never'}")
                    print(f"  Packets: {status.packet_count}")
                    print(f"  Errors: {status.error_count}")
                    print(f"  Firmware: {status.firmware_version or 'Unknown'}")
                    print(f"  Uptime: {status.uptime_seconds or 'Unknown'}s")
            else:
                all_statuses = manager.get_all_device_statuses()

                if args.json:
                    output = [
                        {
                            "device_id": ds.status.device_id,
                            "status": ds.status.status,
                            "last_seen": ds.status.last_seen.isoformat() if ds.status.last_seen else None,
                            "packet_count": ds.status.packet_count,
                            "enabled": ds.device.enabled,
                        }
                        for ds in all_statuses
                    ]
                    print(json.dumps(output, indent=2))
                else:
                    online = sum(1 for ds in all_statuses if ds.status.status == "online")
                    offline = sum(1 for ds in all_statuses if ds.status.status == "offline")
                    unknown = sum(1 for ds in all_statuses if ds.status.status == "unknown")

                    print(f"Device Status Summary: {online} online, {offline} offline, {unknown} unknown")
                    print()
                    print(f"{'ID':<20} {'Status':<10} {'Packets':<12} {'Last Seen':<25}")
                    print("-" * 70)
                    for ds in all_statuses:
                        last_seen = ds.status.last_seen.strftime("%Y-%m-%d %H:%M:%S") if ds.status.last_seen else "Never"
                        print(f"{ds.status.device_id:<20} {ds.status.status:<10} {ds.status.packet_count:<12} {last_seen:<25}")

    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()

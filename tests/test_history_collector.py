#!/usr/bin/env python3
"""
Unit tests for History Collector Service
Tests the REST API and database functionality of the MQTT-based history collector.
"""

import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# Add the tools/csi_visualizer/services directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "csi_visualizer" / "services"))

from history_collector import HistoryDatabase, app


# =============================================================================
# Database Tests
# =============================================================================


class TestHistoryDatabase:
    """HistoryDatabase unit tests"""

    @pytest.fixture
    def db(self):
        """Create a temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        db = HistoryDatabase(db_path)
        yield db
        os.unlink(db_path)

    def test_init_creates_tables(self, db):
        """Database initialization creates required tables."""
        import sqlite3
        conn = sqlite3.connect(db.db_path)
        cursor = conn.cursor()

        # Check tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}

        assert "csi_history" in tables
        assert "devices" in tables
        assert "zone_history" in tables
        conn.close()

    def test_add_csi_entry(self, db):
        """Adding CSI entry stores data correctly."""
        db.add_csi_entry(
            device_id="esp32-001",
            timestamp=time.time(),
            rssi=-50,
            avg_amplitude=1.5,
            variance=0.1,
            present=True,
            breath_ratio=0.8,
            breathing=True,
        )

        history = db.get_device_history("esp32-001", limit=10)
        assert len(history) == 1
        assert history[0]["rssi"] == -50
        assert history[0]["avg_amplitude"] == 1.5
        assert history[0]["present"] is True

    def test_get_device_history_limit(self, db):
        """History respects limit parameter."""
        for i in range(10):
            db.add_csi_entry(
                device_id="esp32-001",
                timestamp=time.time() + i,
                rssi=-50 + i,
                avg_amplitude=1.0,
                variance=0.1,
                present=True,
                breath_ratio=0.5,
                breathing=False,
            )

        history = db.get_device_history("esp32-001", limit=5)
        assert len(history) == 5

    def test_get_device_history_since(self, db):
        """History filters by timestamp."""
        now = time.time()
        for i in range(5):
            db.add_csi_entry(
                device_id="esp32-001",
                timestamp=now + i,
                rssi=-50,
                avg_amplitude=1.0,
                variance=0.1,
                present=True,
                breath_ratio=0.5,
                breathing=False,
            )

        history = db.get_device_history("esp32-001", since=now + 2)
        assert len(history) == 2  # Only entries after now + 2

    def test_get_devices(self, db):
        """Device list includes all known devices."""
        db.add_csi_entry(
            device_id="esp32-001",
            timestamp=time.time(),
            rssi=-50,
            avg_amplitude=1.0,
            variance=0.1,
            present=True,
            breath_ratio=0.5,
            breathing=False,
        )
        db.add_csi_entry(
            device_id="esp32-002",
            timestamp=time.time(),
            rssi=-60,
            avg_amplitude=1.2,
            variance=0.2,
            present=False,
            breath_ratio=0.3,
            breathing=False,
        )

        devices = db.get_devices()
        assert len(devices) == 2
        device_ids = {d["device_id"] for d in devices}
        assert "esp32-001" in device_ids
        assert "esp32-002" in device_ids

    def test_device_online_status(self, db):
        """Device online status based on last_seen timestamp."""
        # Device seen recently -> online
        db.add_csi_entry(
            device_id="esp32-online",
            timestamp=time.time(),
            rssi=-50,
            avg_amplitude=1.0,
            variance=0.1,
            present=True,
            breath_ratio=0.5,
            breathing=False,
        )

        # Device seen long ago -> offline
        db.add_csi_entry(
            device_id="esp32-offline",
            timestamp=time.time() - 120,  # 2 minutes ago
            rssi=-50,
            avg_amplitude=1.0,
            variance=0.1,
            present=True,
            breath_ratio=0.5,
            breathing=False,
        )

        devices = db.get_devices()
        online_device = next(d for d in devices if d["device_id"] == "esp32-online")
        offline_device = next(d for d in devices if d["device_id"] == "esp32-offline")

        assert online_device["online"] is True
        assert offline_device["online"] is False

    def test_cleanup_old_data(self, db):
        """Cleanup removes old entries."""
        old_timestamp = time.time() - (8 * 24 * 3600)  # 8 days ago
        recent_timestamp = time.time()

        db.add_csi_entry(
            device_id="esp32-old",
            timestamp=old_timestamp,
            rssi=-50,
            avg_amplitude=1.0,
            variance=0.1,
            present=True,
            breath_ratio=0.5,
            breathing=False,
        )
        db.add_csi_entry(
            device_id="esp32-recent",
            timestamp=recent_timestamp,
            rssi=-50,
            avg_amplitude=1.0,
            variance=0.1,
            present=True,
            breath_ratio=0.5,
            breathing=False,
        )

        db.cleanup_old_data(retention_days=7)

        # Old entry should be removed
        old_history = db.get_device_history("esp32-old")
        recent_history = db.get_device_history("esp32-recent")

        assert len(old_history) == 0
        assert len(recent_history) == 1

    def test_get_hourly_summary(self, db):
        """Hourly summary aggregates data correctly."""
        now = time.time()
        for i in range(10):
            db.add_csi_entry(
                device_id="esp32-001",
                timestamp=now - i * 60,  # Every minute
                rssi=-50,
                avg_amplitude=1.0 + i * 0.1,
                variance=0.1,
                present=i % 2 == 0,  # Alternating presence
                breath_ratio=0.5,
                breathing=False,
            )

        summary = db.get_hourly_summary("esp32-001", hours=1)
        assert len(summary) >= 1
        # Total samples should be 10 across all hourly buckets
        total_samples = sum(s["sample_count"] for s in summary)
        assert total_samples == 10


# =============================================================================
# REST API Tests
# =============================================================================


class TestHistoryAPI:
    """REST API endpoint tests"""

    @pytest_asyncio.fixture
    async def client(self):
        """Create test client with mocked database."""
        import history_collector

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        # Initialize global db directly for testing
        history_collector.db = HistoryDatabase(db_path)

        # Set environment variables before creating app context
        with patch.dict(os.environ, {
            "DB_PATH": db_path,
            "MQTT_HOST": "localhost",
            "MQTT_PORT": "1883",
        }):
            # Patch MQTT to avoid actual connection
            with patch("history_collector.MQTTCollector") as mock_mqtt:
                mock_mqtt.return_value.start.return_value = None
                mock_mqtt.return_value.stop.return_value = None

                async with AsyncClient(
                    transport=ASGITransport(app=app),
                    base_url="http://test",
                ) as client:
                    yield client

        history_collector.db = None
        os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_health_endpoint(self, client):
        """Health endpoint returns ok status."""
        response = await client.get("/api/v1/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "ok"
        assert "timestamp" in data

    @pytest.mark.asyncio
    async def test_devices_empty(self, client):
        """Devices endpoint returns empty list when no devices."""
        response = await client.get("/api/v1/devices")
        assert response.status_code == 200
        assert response.json() == []

    @pytest.mark.asyncio
    async def test_history_empty(self, client):
        """History endpoint returns empty list for unknown device."""
        response = await client.get("/api/v1/history/unknown-device")
        assert response.status_code == 200
        assert response.json() == []

    @pytest.mark.asyncio
    async def test_history_limit_validation(self, client):
        """History limit parameter is validated."""
        # Too large limit
        response = await client.get("/api/v1/history/esp32-001?limit=100000")
        assert response.status_code == 422

        # Negative limit
        response = await client.get("/api/v1/history/esp32-001?limit=-1")
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_hourly_summary_endpoint(self, client):
        """Hourly summary endpoint works."""
        response = await client.get("/api/v1/history/esp32-001/hourly")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_all_hourly_summary_endpoint(self, client):
        """All devices hourly summary endpoint works."""
        response = await client.get("/api/v1/hourly")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_hourly_hours_validation(self, client):
        """Hourly hours parameter is validated."""
        # Too large hours
        response = await client.get("/api/v1/hourly?hours=1000")
        assert response.status_code == 422

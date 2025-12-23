#!/usr/bin/env python3
"""
Unit tests for ESP32 CSI API Server
Issue #14: サーバー側にESP32からのCSIデータ受信用HTTPエンドポイントを追加
"""

import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# Add the tools/csi_visualizer directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "csi_visualizer"))

from api_server import (
    AppState,
    CSIDataRequest,
    DeviceManager,
    RateLimiter,
    ServerConfig,
    create_app,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def server_config():
    """テスト用のサーバー設定"""
    return ServerConfig(
        host="127.0.0.1",
        port=8080,
        api_prefix="/api/v1",
        api_keys=["test-api-key"],
        rate_limit_per_second=10,
        db_path=":memory:",
    )


@pytest.fixture
def server_config_no_auth():
    """認証なしのサーバー設定"""
    return ServerConfig(
        host="127.0.0.1",
        port=8080,
        api_prefix="/api/v1",
        api_keys=[],
        rate_limit_per_second=10,
        db_path=":memory:",
    )


@pytest.fixture
def app(server_config):
    """テスト用のFastAPIアプリ"""
    return create_app(server_config)


@pytest.fixture
def app_no_auth(server_config_no_auth):
    """認証なしのテスト用FastAPIアプリ"""
    return create_app(server_config_no_auth)


@pytest_asyncio.fixture
async def client(app):
    """テスト用の非同期HTTPクライアント"""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        yield client


@pytest_asyncio.fixture
async def client_no_auth(app_no_auth):
    """認証なしのテスト用HTTPクライアント"""
    async with AsyncClient(
        transport=ASGITransport(app=app_no_auth),
        base_url="http://test",
    ) as client:
        yield client


@pytest.fixture
def valid_csi_data():
    """有効なCSIデータ"""
    return {
        "device_id": "esp32-001",
        "timestamp": int(time.time() * 1000),
        "seq": 100,
        "mac": "aa:bb:cc:dd:ee:ff",
        "rssi": -50,
        "rate": 11,
        "channel": 11,
        "csi_data": [1.0, 2.0, 3.0, 4.0, 5.0],
        "metadata": {
            "firmware_version": "1.0.0",
            "uptime_ms": 12345678,
        },
    }


# =============================================================================
# Health Endpoint Tests
# =============================================================================


class TestHealthEndpoint:
    """ヘルスチェックエンドポイントのテスト"""

    @pytest.mark.asyncio
    async def test_health_check_returns_healthy(self, client):
        """ヘルスチェックが正常なレスポンスを返す"""
        response = await client.get("/api/v1/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "uptime_seconds" in data
        assert "version" in data
        assert data["devices_seen"] == 0
        assert data["total_packets"] == 0

    @pytest.mark.asyncio
    async def test_health_check_no_auth_required(self, client):
        """ヘルスチェックは認証不要"""
        response = await client.get("/api/v1/health")
        assert response.status_code == 200


# =============================================================================
# CSI Endpoint Tests
# =============================================================================


class TestCSIEndpoint:
    """CSIデータ受信エンドポイントのテスト"""

    @pytest.mark.asyncio
    async def test_receive_csi_success(self, client, valid_csi_data):
        """CSIデータの正常受信"""
        response = await client.post(
            "/api/v1/csi",
            json=valid_csi_data,
            headers={"X-API-Key": "test-api-key"},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "ok"
        assert "received_at" in data

    @pytest.mark.asyncio
    async def test_receive_csi_without_auth(self, client, valid_csi_data):
        """認証なしでCSIデータを送信すると401エラー"""
        response = await client.post(
            "/api/v1/csi",
            json=valid_csi_data,
        )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_receive_csi_invalid_api_key(self, client, valid_csi_data):
        """無効なAPIキーで401エラー"""
        response = await client.post(
            "/api/v1/csi",
            json=valid_csi_data,
            headers={"X-API-Key": "invalid-key"},
        )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_receive_csi_no_auth_mode(self, client_no_auth, valid_csi_data):
        """認証無効モードでは認証なしで送信可能"""
        response = await client_no_auth.post(
            "/api/v1/csi",
            json=valid_csi_data,
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_receive_csi_invalid_mac(self, client, valid_csi_data):
        """無効なMACアドレスで422エラー"""
        valid_csi_data["mac"] = "invalid-mac"
        response = await client.post(
            "/api/v1/csi",
            json=valid_csi_data,
            headers={"X-API-Key": "test-api-key"},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_receive_csi_invalid_channel(self, client, valid_csi_data):
        """無効なチャネルで422エラー"""
        valid_csi_data["channel"] = 20  # 1-14のみ有効
        response = await client.post(
            "/api/v1/csi",
            json=valid_csi_data,
            headers={"X-API-Key": "test-api-key"},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_receive_csi_empty_data(self, client, valid_csi_data):
        """空のCSIデータで422エラー"""
        valid_csi_data["csi_data"] = []
        response = await client.post(
            "/api/v1/csi",
            json=valid_csi_data,
            headers={"X-API-Key": "test-api-key"},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_receive_csi_missing_required_field(self, client):
        """必須フィールドがない場合は422エラー"""
        incomplete_data = {
            "device_id": "esp32-001",
            "timestamp": int(time.time() * 1000),
            # seq, mac, rssi, rate, channel, csi_data が欠落
        }
        response = await client.post(
            "/api/v1/csi",
            json=incomplete_data,
            headers={"X-API-Key": "test-api-key"},
        )
        assert response.status_code == 422


# =============================================================================
# CSI Batch Endpoint Tests
# =============================================================================


class TestCSIBatchEndpoint:
    """CSIバッチ受信エンドポイントのテスト"""

    @pytest.mark.asyncio
    async def test_receive_batch_success(self, client, valid_csi_data):
        """バッチデータの正常受信"""
        batch_data = {
            "device_id": "esp32-001",
            "data": [valid_csi_data, valid_csi_data],
        }
        response = await client.post(
            "/api/v1/csi/batch",
            json=batch_data,
            headers={"X-API-Key": "test-api-key"},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "ok"
        assert data["received_count"] == 2
        assert "received_at" in data

    @pytest.mark.asyncio
    async def test_receive_batch_empty(self, client):
        """空のバッチデータで422エラー"""
        batch_data = {
            "device_id": "esp32-001",
            "data": [],
        }
        response = await client.post(
            "/api/v1/csi/batch",
            json=batch_data,
            headers={"X-API-Key": "test-api-key"},
        )
        assert response.status_code == 422


# =============================================================================
# Device Endpoint Tests
# =============================================================================


class TestDeviceEndpoints:
    """デバイス管理エンドポイントのテスト"""

    @pytest.mark.asyncio
    async def test_list_devices_empty(self, client):
        """デバイスがない場合は空リストを返す"""
        response = await client.get(
            "/api/v1/devices",
            headers={"X-API-Key": "test-api-key"},
        )
        assert response.status_code == 200
        assert response.json() == []

    @pytest.mark.asyncio
    async def test_list_devices_after_csi(self, client, valid_csi_data):
        """CSIデータ送信後にデバイスがリストに表示される"""
        # CSIデータを送信
        await client.post(
            "/api/v1/csi",
            json=valid_csi_data,
            headers={"X-API-Key": "test-api-key"},
        )

        # デバイス一覧を取得
        response = await client.get(
            "/api/v1/devices",
            headers={"X-API-Key": "test-api-key"},
        )
        assert response.status_code == 200

        devices = response.json()
        assert len(devices) == 1
        assert devices[0]["device_id"] == "esp32-001"
        assert devices[0]["packet_count"] == 1

    @pytest.mark.asyncio
    async def test_get_device_status(self, client, valid_csi_data):
        """特定デバイスの状態を取得"""
        # CSIデータを送信
        await client.post(
            "/api/v1/csi",
            json=valid_csi_data,
            headers={"X-API-Key": "test-api-key"},
        )

        # デバイス状態を取得
        response = await client.get(
            "/api/v1/devices/esp32-001",
            headers={"X-API-Key": "test-api-key"},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["device_id"] == "esp32-001"
        assert data["status"] == "online"
        assert data["packet_count"] == 1

    @pytest.mark.asyncio
    async def test_get_device_not_found(self, client):
        """存在しないデバイスは404エラー"""
        response = await client.get(
            "/api/v1/devices/non-existent",
            headers={"X-API-Key": "test-api-key"},
        )
        assert response.status_code == 404


# =============================================================================
# Rate Limiter Tests
# =============================================================================


class TestRateLimiter:
    """レート制限のテスト"""

    @pytest.mark.asyncio
    async def test_rate_limiter_allows_under_limit(self):
        """制限内のリクエストは許可される"""
        limiter = RateLimiter(max_requests_per_second=5)
        for _ in range(5):
            result = await limiter.check("device-1")
            assert result is True

    @pytest.mark.asyncio
    async def test_rate_limiter_blocks_over_limit(self):
        """制限を超えたリクエストはブロックされる"""
        limiter = RateLimiter(max_requests_per_second=3)
        for _ in range(3):
            await limiter.check("device-1")

        result = await limiter.check("device-1")
        assert result is False

    @pytest.mark.asyncio
    async def test_rate_limiter_per_device(self):
        """デバイスごとに独立したレート制限"""
        limiter = RateLimiter(max_requests_per_second=2)

        # device-1で2回
        await limiter.check("device-1")
        await limiter.check("device-1")

        # device-1は制限超過
        result1 = await limiter.check("device-1")
        assert result1 is False

        # device-2はまだOK
        result2 = await limiter.check("device-2")
        assert result2 is True

    @pytest.mark.asyncio
    async def test_rate_limiter_cleanup(self):
        """クリーンアップで古いエントリが削除される"""
        limiter = RateLimiter(max_requests_per_second=10)
        await limiter.check("device-1")

        # 時間を経過させてクリーンアップ
        await asyncio.sleep(0.1)
        await limiter.cleanup()

        # デバイスのエントリがまだ存在することを確認（時間経過が短いため）
        assert "device-1" in limiter.states


# =============================================================================
# Device Manager Tests
# =============================================================================


class TestDeviceManager:
    """デバイスマネージャーのテスト"""

    @pytest.mark.asyncio
    async def test_device_manager_update(self):
        """デバイス情報の更新"""
        manager = DeviceManager()
        await manager.update("esp32-001")

        device = await manager.get_status("esp32-001")
        assert device is not None
        assert device.device_id == "esp32-001"
        assert device.packet_count == 1

    @pytest.mark.asyncio
    async def test_device_manager_multiple_updates(self):
        """複数回の更新でカウントが増加"""
        manager = DeviceManager()
        await manager.update("esp32-001")
        await manager.update("esp32-001")
        await manager.update("esp32-001")

        device = await manager.get_status("esp32-001")
        assert device.packet_count == 3

    @pytest.mark.asyncio
    async def test_device_manager_error_count(self):
        """エラーカウントの記録"""
        manager = DeviceManager()
        await manager.record_error("esp32-001")
        await manager.record_error("esp32-001")

        device = await manager.get_status("esp32-001")
        assert device.error_count == 2

    @pytest.mark.asyncio
    async def test_device_manager_total_packets(self):
        """総パケット数の取得"""
        manager = DeviceManager()
        await manager.update("esp32-001")
        await manager.update("esp32-001")
        await manager.update("esp32-002")

        total = await manager.get_total_packets()
        assert total == 3


# =============================================================================
# Server Config Tests
# =============================================================================


class TestServerConfig:
    """サーバー設定のテスト"""

    def test_default_config(self):
        """デフォルト設定"""
        config = ServerConfig()
        assert config.host == "0.0.0.0"
        assert config.port == 8080
        assert config.api_prefix == "/api/v1"
        assert config.api_keys == []
        assert config.rate_limit_per_second == 100

    def test_config_from_yaml_not_found(self, tmp_path):
        """存在しないYAMLファイルはデフォルト設定を返す"""
        config = ServerConfig.from_yaml(str(tmp_path / "not_found.yaml"))
        assert config.host == "0.0.0.0"

    def test_config_from_yaml(self, tmp_path):
        """YAMLファイルから設定を読み込む"""
        yaml_content = """
api_server:
  host: "127.0.0.1"
  port: 9090
  api_keys:
    - "key1"
    - "key2"
  rate_limit_per_second: 50
"""
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(yaml_content)

        config = ServerConfig.from_yaml(str(yaml_path))
        assert config.host == "127.0.0.1"
        assert config.port == 9090
        assert config.api_keys == ["key1", "key2"]
        assert config.rate_limit_per_second == 50

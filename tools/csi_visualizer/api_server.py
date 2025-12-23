#!/usr/bin/env python3
"""
HTTP API Server for ESP32 CSI Data Reception

ESP32デバイスからWi-Fi経由でCSIデータを受信するためのHTTP APIサーバー。
Issue #14: サーバー側にESP32からのCSIデータ受信用HTTPエンドポイントを追加

Usage:
    python api_server.py --config config.yaml --port 8080
    python api_server.py --api-key your-secret-key
"""

import asyncio
import json
import logging
import os
import sys
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, field_validator

# Import DataRecorder from server_multi
sys.path.insert(0, str(Path(__file__).parent))
from server_multi import DataRecorder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic Models
# =============================================================================


class CSIMetadata(BaseModel):
    """CSIデータのメタデータ"""

    firmware_version: Optional[str] = None
    uptime_ms: Optional[int] = None


class CSIDataRequest(BaseModel):
    """CSIデータ受信リクエスト"""

    device_id: str = Field(..., min_length=1, max_length=64, description="デバイスID")
    timestamp: int = Field(..., ge=0, description="タイムスタンプ (ms)")
    seq: int = Field(..., ge=0, description="シーケンス番号")
    mac: str = Field(..., pattern=r"^([0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}$", description="MACアドレス")
    rssi: int = Field(..., ge=-100, le=0, description="RSSI値")
    rate: int = Field(..., ge=0, description="データレート")
    channel: int = Field(..., ge=1, le=14, description="Wi-Fiチャネル")
    csi_data: List[float] = Field(..., min_length=1, description="CSI振幅データ")
    metadata: Optional[CSIMetadata] = None

    @field_validator("csi_data")
    @classmethod
    def validate_csi_data(cls, v: List[float]) -> List[float]:
        if len(v) > 256:
            raise ValueError("csi_data must have at most 256 elements")
        return v


class CSIDataResponse(BaseModel):
    """CSIデータ受信レスポンス"""

    status: str = "ok"
    received_at: int


class CSIBatchRequest(BaseModel):
    """CSIデータバッチ受信リクエスト"""

    device_id: str = Field(..., min_length=1, max_length=64)
    data: List[CSIDataRequest] = Field(..., min_length=1, max_length=100)


class CSIBatchResponse(BaseModel):
    """CSIデータバッチ受信レスポンス"""

    status: str = "ok"
    received_count: int
    received_at: int


class HealthResponse(BaseModel):
    """ヘルスチェックレスポンス"""

    status: str = "healthy"
    timestamp: int
    uptime_seconds: float
    version: str = "1.0.0"
    devices_seen: int
    total_packets: int


class DeviceStatusResponse(BaseModel):
    """デバイス状態レスポンス"""

    device_id: str
    status: str
    last_seen: Optional[str] = None
    packet_count: int
    error_count: int


class ErrorResponse(BaseModel):
    """エラーレスポンス"""

    error: str
    detail: Optional[str] = None


# =============================================================================
# Rate Limiter
# =============================================================================


@dataclass
class RateLimitState:
    """レート制限の状態"""

    requests: List[float] = field(default_factory=list)
    window_seconds: int = 1
    max_requests: int = 100


class RateLimiter:
    """デバイスごとのレート制限"""

    def __init__(self, max_requests_per_second: int = 100):
        self.max_requests = max_requests_per_second
        self.states: Dict[str, RateLimitState] = defaultdict(
            lambda: RateLimitState(max_requests=self.max_requests)
        )
        self._lock = asyncio.Lock()

    async def check(self, device_id: str) -> bool:
        """レート制限をチェック。制限内ならTrue、超過ならFalse"""
        async with self._lock:
            state = self.states[device_id]
            now = time.time()

            # 古いリクエストを削除
            state.requests = [t for t in state.requests if now - t < state.window_seconds]

            if len(state.requests) >= state.max_requests:
                return False

            state.requests.append(now)
            return True

    async def cleanup(self):
        """古いエントリをクリーンアップ"""
        async with self._lock:
            now = time.time()
            to_delete = []
            for device_id, state in self.states.items():
                state.requests = [t for t in state.requests if now - t < state.window_seconds]
                if not state.requests:
                    to_delete.append(device_id)
            for device_id in to_delete:
                del self.states[device_id]


# =============================================================================
# Device State Manager
# =============================================================================


@dataclass
class DeviceInfo:
    """デバイス情報"""

    device_id: str
    last_seen: float = 0
    packet_count: int = 0
    error_count: int = 0
    firmware_version: Optional[str] = None


class DeviceManager:
    """デバイス状態管理"""

    def __init__(self):
        self.devices: Dict[str, DeviceInfo] = {}
        self._lock = asyncio.Lock()

    async def update(self, device_id: str, metadata: Optional[CSIMetadata] = None):
        """デバイス情報を更新"""
        async with self._lock:
            if device_id not in self.devices:
                self.devices[device_id] = DeviceInfo(device_id=device_id)
            device = self.devices[device_id]
            device.last_seen = time.time()
            device.packet_count += 1
            if metadata and metadata.firmware_version:
                device.firmware_version = metadata.firmware_version

    async def record_error(self, device_id: str):
        """エラーを記録"""
        async with self._lock:
            if device_id not in self.devices:
                self.devices[device_id] = DeviceInfo(device_id=device_id)
            self.devices[device_id].error_count += 1

    async def get_status(self, device_id: str) -> Optional[DeviceInfo]:
        """デバイス状態を取得"""
        async with self._lock:
            return self.devices.get(device_id)

    async def get_all_devices(self) -> List[DeviceInfo]:
        """全デバイス情報を取得"""
        async with self._lock:
            return list(self.devices.values())

    async def get_total_packets(self) -> int:
        """総パケット数を取得"""
        async with self._lock:
            return sum(d.packet_count for d in self.devices.values())


# =============================================================================
# Server Configuration
# =============================================================================


@dataclass
class ServerConfig:
    """サーバー設定"""

    host: str = "0.0.0.0"
    port: int = 8080
    api_prefix: str = "/api/v1"
    api_keys: List[str] = field(default_factory=list)
    rate_limit_per_second: int = 100
    db_path: str = "history.db"
    cors_origins: List[str] = field(default_factory=lambda: ["*"])

    @classmethod
    def from_yaml(cls, path: str) -> "ServerConfig":
        """YAMLファイルから設定を読み込む"""
        config_path = Path(path)
        if not config_path.exists():
            logger.warning(f"Config file not found: {path}")
            return cls()

        with open(config_path) as f:
            data = yaml.safe_load(f)

        api_config = data.get("api_server", {})
        return cls(
            host=api_config.get("host", "0.0.0.0"),
            port=api_config.get("port", 8080),
            api_prefix=api_config.get("api_prefix", "/api/v1"),
            api_keys=api_config.get("api_keys", []),
            rate_limit_per_second=api_config.get("rate_limit_per_second", 100),
            db_path=api_config.get("db_path", "history.db"),
            cors_origins=api_config.get("cors_origins", ["*"]),
        )


# =============================================================================
# Application State
# =============================================================================


class AppState:
    """アプリケーション状態"""

    def __init__(self, config: ServerConfig):
        self.config = config
        self.start_time = time.time()
        self.recorder = DataRecorder(config.db_path)
        self.rate_limiter = RateLimiter(config.rate_limit_per_second)
        self.device_manager = DeviceManager()
        self._cleanup_task: Optional[asyncio.Task] = None

    async def start(self):
        """バックグラウンドタスクを開始"""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self):
        """クリーンアップ"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        self.recorder.close()

    async def _cleanup_loop(self):
        """定期的なクリーンアップ"""
        while True:
            await asyncio.sleep(60)
            await self.rate_limiter.cleanup()
            await self.recorder.cleanup_old_data_async()


# =============================================================================
# FastAPI Application
# =============================================================================


def create_app(config: Optional[ServerConfig] = None) -> FastAPI:
    """FastAPIアプリケーションを作成"""

    if config is None:
        config = ServerConfig()

    app_state = AppState(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """アプリケーションのライフサイクル管理"""
        logger.info("Starting API server...")
        await app_state.start()
        yield
        logger.info("Shutting down API server...")
        await app_state.stop()

    app = FastAPI(
        title="ESP32 CSI API Server",
        description="ESP32デバイスからWi-Fi経由でCSIデータを受信するためのAPI",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS設定
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # APIキー認証
    api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

    async def verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> str:
        """APIキーを検証"""
        # APIキーが設定されていない場合は認証をスキップ
        if not config.api_keys:
            return "anonymous"

        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key is required",
            )

        if api_key not in config.api_keys:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
            )

        return api_key

    async def check_rate_limit(request: Request) -> None:
        """レート制限をチェック"""
        # device_idはリクエストボディから取得するため、ここではIPアドレスで制限
        client_ip = request.client.host if request.client else "unknown"
        if not await app_state.rate_limiter.check(client_ip):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
            )

    # ==========================================================================
    # Endpoints
    # ==========================================================================

    @app.get(
        f"{config.api_prefix}/health",
        response_model=HealthResponse,
        tags=["Health"],
        summary="ヘルスチェック",
    )
    async def health_check() -> HealthResponse:
        """サーバーの状態を確認"""
        devices = await app_state.device_manager.get_all_devices()
        total_packets = await app_state.device_manager.get_total_packets()

        return HealthResponse(
            status="healthy",
            timestamp=int(time.time() * 1000),
            uptime_seconds=time.time() - app_state.start_time,
            devices_seen=len(devices),
            total_packets=total_packets,
        )

    @app.post(
        f"{config.api_prefix}/csi",
        response_model=CSIDataResponse,
        tags=["CSI"],
        summary="CSIデータ受信",
        responses={
            401: {"model": ErrorResponse, "description": "認証エラー"},
            429: {"model": ErrorResponse, "description": "レート制限超過"},
            500: {"model": ErrorResponse, "description": "サーバーエラー"},
        },
    )
    async def receive_csi(
        request: Request,
        data: CSIDataRequest,
        api_key: str = Depends(verify_api_key),
    ) -> CSIDataResponse:
        """ESP32からCSIデータを受信"""
        # レート制限チェック
        if not await app_state.rate_limiter.check(data.device_id):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded for device {data.device_id}",
            )

        try:
            # デバイス情報を更新
            await app_state.device_manager.update(data.device_id, data.metadata)

            # DataRecorderに保存
            record_data = {
                "device_id": data.device_id,
                "timestamp": data.timestamp / 1000,  # msからsに変換
                "rssi": data.rssi,
                "zone": "default",  # 将来的にはデバイス設定から取得
                "breath": {
                    "present": True,  # CSIデータがあれば存在とみなす
                    "breath_rate": 0,
                    "breath_ratio": 0,
                    "confidence": 0,
                },
            }
            await app_state.recorder.save_device_data(record_data)

            received_at = int(time.time() * 1000)
            logger.debug(
                f"Received CSI from {data.device_id}: seq={data.seq}, rssi={data.rssi}"
            )

            return CSIDataResponse(status="ok", received_at=received_at)

        except Exception as e:
            logger.error(f"Error processing CSI data: {e}")
            await app_state.device_manager.record_error(data.device_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e),
            )

    @app.post(
        f"{config.api_prefix}/csi/batch",
        response_model=CSIBatchResponse,
        tags=["CSI"],
        summary="CSIデータバッチ受信",
        responses={
            401: {"model": ErrorResponse, "description": "認証エラー"},
            429: {"model": ErrorResponse, "description": "レート制限超過"},
            500: {"model": ErrorResponse, "description": "サーバーエラー"},
        },
    )
    async def receive_csi_batch(
        request: Request,
        batch: CSIBatchRequest,
        api_key: str = Depends(verify_api_key),
    ) -> CSIBatchResponse:
        """ESP32からCSIデータをバッチで受信"""
        # レート制限チェック（バッチ全体で1リクエストとしてカウント）
        if not await app_state.rate_limiter.check(batch.device_id):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded for device {batch.device_id}",
            )

        received_count = 0
        try:
            for data in batch.data:
                # デバイス情報を更新
                await app_state.device_manager.update(data.device_id, data.metadata)

                # DataRecorderに保存
                record_data = {
                    "device_id": data.device_id,
                    "timestamp": data.timestamp / 1000,
                    "rssi": data.rssi,
                    "zone": "default",
                    "breath": {
                        "present": True,
                        "breath_rate": 0,
                        "breath_ratio": 0,
                        "confidence": 0,
                    },
                }
                await app_state.recorder.save_device_data(record_data)
                received_count += 1

            received_at = int(time.time() * 1000)
            logger.debug(f"Received {received_count} CSI packets from {batch.device_id}")

            return CSIBatchResponse(
                status="ok",
                received_count=received_count,
                received_at=received_at,
            )

        except Exception as e:
            logger.error(f"Error processing batch CSI data: {e}")
            await app_state.device_manager.record_error(batch.device_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e),
            )

    @app.get(
        f"{config.api_prefix}/devices",
        response_model=List[DeviceStatusResponse],
        tags=["Devices"],
        summary="デバイス一覧",
    )
    async def list_devices(
        api_key: str = Depends(verify_api_key),
    ) -> List[DeviceStatusResponse]:
        """登録されたデバイスの一覧を取得"""
        devices = await app_state.device_manager.get_all_devices()
        now = time.time()

        result = []
        for device in devices:
            # 30秒以上データがない場合はオフライン
            status = "online" if now - device.last_seen < 30 else "offline"
            last_seen = (
                datetime.fromtimestamp(device.last_seen).isoformat()
                if device.last_seen > 0
                else None
            )
            result.append(
                DeviceStatusResponse(
                    device_id=device.device_id,
                    status=status,
                    last_seen=last_seen,
                    packet_count=device.packet_count,
                    error_count=device.error_count,
                )
            )

        return result

    @app.get(
        config.api_prefix + "/devices/{device_id}",
        response_model=DeviceStatusResponse,
        tags=["Devices"],
        summary="デバイス状態",
        responses={404: {"model": ErrorResponse, "description": "デバイスが見つからない"}},
    )
    async def get_device_status(
        device_id: str,
        api_key: str = Depends(verify_api_key),
    ) -> DeviceStatusResponse:
        """指定されたデバイスの状態を取得"""
        device = await app_state.device_manager.get_status(device_id)
        if not device:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Device {device_id} not found",
            )

        now = time.time()
        device_status = "online" if now - device.last_seen < 30 else "offline"
        last_seen = (
            datetime.fromtimestamp(device.last_seen).isoformat()
            if device.last_seen > 0
            else None
        )

        return DeviceStatusResponse(
            device_id=device.device_id,
            status=device_status,
            last_seen=last_seen,
            packet_count=device.packet_count,
            error_count=device.error_count,
        )

    return app


# =============================================================================
# Main
# =============================================================================


def main():
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="ESP32 CSI API Server")
    parser.add_argument(
        "--config", "-c", default="config.yaml", help="Path to config file"
    )
    parser.add_argument("--host", default=None, help="Host to bind to")
    parser.add_argument("--port", "-p", type=int, default=None, help="Port to bind to")
    parser.add_argument("--api-key", default=None, help="API key for authentication")
    parser.add_argument("--db", "-d", default=None, help="Path to database file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    # 設定を読み込む
    config = ServerConfig.from_yaml(args.config)

    # コマンドライン引数で上書き
    if args.host:
        config.host = args.host
    if args.port:
        config.port = args.port
    if args.api_key:
        config.api_keys = [args.api_key]
    if args.db:
        config.db_path = args.db

    # 環境変数からAPIキーを読み込む
    env_api_key = os.environ.get("WAVIRA_API_KEY")
    if env_api_key and env_api_key not in config.api_keys:
        config.api_keys.append(env_api_key)

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    print("=" * 60)
    print("ESP32 CSI API Server")
    print("=" * 60)
    print(f"Host: {config.host}")
    print(f"Port: {config.port}")
    print(f"API Prefix: {config.api_prefix}")
    print(f"Database: {config.db_path}")
    print(f"Auth: {'Enabled' if config.api_keys else 'Disabled'}")
    print(f"Rate Limit: {config.rate_limit_per_second} req/s per device")
    print("=" * 60)

    app = create_app(config)

    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level="debug" if args.debug else "info",
    )


if __name__ == "__main__":
    main()

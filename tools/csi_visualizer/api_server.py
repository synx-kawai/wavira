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
import logging
import os
import secrets
import sys
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, field_validator

# Import DataRecorder from server_multi
sys.path.insert(0, str(Path(__file__).parent))
from server_multi import DataRecorder
from device_manager import DeviceAuthManager, verify_api_key as verify_api_key_hash

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
# Device Management Models (Issue #16)
# =============================================================================


class DeviceRegisterRequest(BaseModel):
    """デバイス登録リクエスト"""

    device_id: str = Field(..., min_length=1, max_length=64, description="デバイスID")
    zone: Optional[str] = Field(None, max_length=64, description="ゾーン")
    location: Optional[str] = Field(None, max_length=128, description="設置場所")


class DeviceRegisterResponse(BaseModel):
    """デバイス登録レスポンス"""

    device_id: str
    api_key: str
    zone: Optional[str] = None
    location: Optional[str] = None
    message: str = "Device registered successfully. Save the API key - it will not be shown again."


class DeviceUpdateRequest(BaseModel):
    """デバイス更新リクエスト"""

    zone: Optional[str] = Field(None, max_length=64, description="ゾーン")
    location: Optional[str] = Field(None, max_length=128, description="設置場所")
    enabled: Optional[bool] = Field(None, description="有効/無効")


class DeviceDetailResponse(BaseModel):
    """デバイス詳細レスポンス"""

    device_id: str
    zone: Optional[str] = None
    location: Optional[str] = None
    enabled: bool = True
    status: str  # online, offline, unknown
    last_seen: Optional[str] = None
    packet_count: int = 0
    error_count: int = 0
    firmware_version: Optional[str] = None
    uptime_seconds: Optional[int] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class DeviceListResponse(BaseModel):
    """デバイス一覧レスポンス"""

    devices: List[DeviceDetailResponse]
    total: int
    online_count: int
    offline_count: int


class ApiKeyRotateResponse(BaseModel):
    """APIキーローテーションレスポンス"""

    device_id: str
    new_api_key: str
    message: str = "API key rotated successfully. Save the new API key - it will not be shown again."


class DeviceDeleteResponse(BaseModel):
    """デバイス削除レスポンス"""

    device_id: str
    message: str = "Device deleted successfully"


# =============================================================================
# Rate Limiter
# =============================================================================


@dataclass
class RateLimitState:
    """レート制限の状態"""

    requests: deque = field(default_factory=deque)
    window_seconds: int = 1
    max_requests: int = 100


class RateLimiter:
    """デバイスごとのレート制限"""

    # メモリ枯渇防止のための最大エントリ数（認証なしモードでの攻撃対策）
    MAX_DEVICE_ENTRIES = 10000

    def __init__(self, max_requests_per_second: int = 100):
        self.max_requests = max_requests_per_second
        self.states: Dict[str, RateLimitState] = defaultdict(
            lambda: RateLimitState(max_requests=self.max_requests)
        )
        self._lock = asyncio.Lock()

    async def check(self, device_id: str) -> bool:
        """レート制限をチェック。制限内ならTrue、超過ならFalse"""
        async with self._lock:
            # メモリ枯渇防止: エントリ数が上限に達した場合、新規デバイスは拒否
            if device_id not in self.states and len(self.states) >= self.MAX_DEVICE_ENTRIES:
                logger.warning(
                    f"Rate limiter max entries ({self.MAX_DEVICE_ENTRIES}) reached. "
                    f"Rejecting new device: {device_id}"
                )
                return False

            state = self.states[device_id]
            now = time.time()

            # 古いリクエストを効率的に削除（deque.popleft()）
            while state.requests and now - state.requests[0] >= state.window_seconds:
                state.requests.popleft()

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
                # 古いリクエストを効率的に削除（deque.popleft()）
                while state.requests and now - state.requests[0] >= state.window_seconds:
                    state.requests.popleft()
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
    device_db_path: str = "devices.db"  # Device authentication database
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    # Admin API key for device management operations (required for POST/PUT/DELETE on /devices)
    admin_api_key: Optional[str] = None

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
            device_db_path=api_config.get("device_db_path", "devices.db"),
            cors_origins=api_config.get("cors_origins", ["*"]),
            admin_api_key=api_config.get("admin_api_key"),
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
        # Device authentication manager (Issue #16)
        self.device_auth_manager = DeviceAuthManager(config.device_db_path)
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

        # データベース接続を安全にクローズ
        try:
            self.recorder.close()
            self.device_auth_manager.close()
            logger.info("Database connections closed successfully")
        except Exception as e:
            logger.error(f"Error closing database connection: {e}")

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

        # セキュリティ警告: 認証が無効な場合
        if not config.api_keys:
            logger.critical(
                "=" * 60 + "\n"
                "⚠️  SECURITY WARNING: API authentication is DISABLED!\n"
                "Any client can send data to this server without authentication.\n"
                "This is acceptable for development/testing but NOT for production.\n"
                "To enable authentication, set api_keys in config.yaml or use --api-key.\n"
                + "=" * 60
            )

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
        """
        APIキーを検証

        Returns:
            - "anonymous": 認証無効時
            - "admin:{key}": 静的APIキー（後方互換）
            - "device:{device_id}": デバイス固有APIキー
        """
        # APIキーが設定されていない場合は認証をスキップ
        # 静的APIキーがなく、かつ登録済みデバイスもない場合のみ認証を無効化
        has_static_keys = bool(config.api_keys)
        has_registered_devices = (
            app_state.device_auth_manager
            and app_state.device_auth_manager.has_registered_devices()
        )
        if not has_static_keys and not has_registered_devices:
            return "anonymous"

        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key is required",
            )

        # 1. 静的APIキーをチェック（後方互換性）
        for valid_key in config.api_keys:
            if secrets.compare_digest(api_key, valid_key):
                return f"admin:{api_key}"

        # 2. デバイス固有APIキーをチェック
        if app_state.device_auth_manager:
            device_id = app_state.device_auth_manager.authenticate_by_key(api_key)
            if device_id:
                return f"device:{device_id}"

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

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
        auth_result: str = Depends(verify_api_key),
    ) -> CSIDataResponse:
        """ESP32からCSIデータを受信"""
        # デバイス認証時はdevice_idの一致を確認（なりすまし防止）
        if auth_result.startswith("device:"):
            auth_device_id = auth_result.split(":", 1)[1]
            if auth_device_id != data.device_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Device ID mismatch: authenticated as {auth_device_id}",
                )

        # レート制限チェック
        if not await app_state.rate_limiter.check(data.device_id):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded for device {data.device_id}",
            )

        try:
            # デバイス情報を更新（メモリ上）
            await app_state.device_manager.update(data.device_id, data.metadata)

            # デバイスステータスを更新（永続化DB）
            if app_state.device_auth_manager:
                firmware_ver = data.metadata.firmware_version if data.metadata else None
                app_state.device_auth_manager.update_device_status(
                    device_id=data.device_id,
                    packet_count_delta=1,
                    firmware_version=firmware_ver,
                )

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
        auth_result: str = Depends(verify_api_key),
    ) -> CSIBatchResponse:
        """ESP32からCSIデータをバッチで受信"""
        # デバイス認証時はdevice_idの一致を確認（なりすまし防止）
        if auth_result.startswith("device:"):
            auth_device_id = auth_result.split(":", 1)[1]
            if auth_device_id != batch.device_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Device ID mismatch: authenticated as {auth_device_id}",
                )

        # レート制限チェック（バッチ全体で1リクエストとしてカウント）
        if not await app_state.rate_limiter.check(batch.device_id):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded for device {batch.device_id}",
            )

        received_count = 0
        try:
            for data in batch.data:
                # デバイス情報を更新（メモリ上）
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

            # デバイスステータスを更新（永続化DB）- バッチ全体で1回
            if app_state.device_auth_manager and received_count > 0:
                app_state.device_auth_manager.update_device_status(
                    device_id=batch.device_id,
                    packet_count_delta=received_count,
                )

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
            device_status = "online" if now - device.last_seen < 30 else "offline"
            last_seen = (
                datetime.fromtimestamp(device.last_seen).isoformat()
                if device.last_seen > 0
                else None
            )
            result.append(
                DeviceStatusResponse(
                    device_id=device.device_id,
                    status=device_status,
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

    # ==========================================================================
    # Device Management Endpoints (Issue #16)
    # ==========================================================================

    async def verify_admin_key(api_key: Optional[str] = Security(api_key_header)) -> str:
        """管理者APIキーを検証（デバイス管理操作用）"""
        if not config.admin_api_key:
            # admin_api_keyが設定されていない場合はエンドポイントを無効化
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Device management is disabled. Set admin_api_key to enable.",
            )

        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Admin API key is required for this operation",
            )

        if not secrets.compare_digest(api_key, config.admin_api_key):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid admin API key",
            )

        return api_key

    @app.get(
        f"{config.api_prefix}/admin/devices",
        response_model=DeviceListResponse,
        tags=["Device Management"],
        summary="登録デバイス一覧（管理用）",
    )
    async def list_registered_devices(
        zone: Optional[str] = None,
        enabled_only: bool = False,
        api_key: str = Depends(verify_admin_key),
    ) -> DeviceListResponse:
        """登録されたデバイスの一覧を取得（認証情報含む）"""
        all_devices = app_state.device_auth_manager.get_all_device_statuses()

        # フィルタリング
        if zone:
            all_devices = [d for d in all_devices if d.device.zone == zone]
        if enabled_only:
            all_devices = [d for d in all_devices if d.device.enabled]

        devices = []
        online_count = 0
        offline_count = 0

        for ds in all_devices:
            device = ds.device
            status_info = ds.status

            if status_info.status == "online":
                online_count += 1
            elif status_info.status == "offline":
                offline_count += 1

            devices.append(DeviceDetailResponse(
                device_id=device.id,
                zone=device.zone,
                location=device.location,
                enabled=device.enabled,
                status=status_info.status,
                last_seen=status_info.last_seen.isoformat() if status_info.last_seen else None,
                packet_count=status_info.packet_count,
                error_count=status_info.error_count,
                firmware_version=status_info.firmware_version,
                uptime_seconds=status_info.uptime_seconds,
                created_at=device.created_at.isoformat() if device.created_at else None,
                updated_at=device.updated_at.isoformat() if device.updated_at else None,
            ))

        return DeviceListResponse(
            devices=devices,
            total=len(devices),
            online_count=online_count,
            offline_count=offline_count,
        )

    @app.post(
        f"{config.api_prefix}/admin/devices",
        response_model=DeviceRegisterResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["Device Management"],
        summary="デバイス登録",
        responses={
            400: {"model": ErrorResponse, "description": "デバイスが既に存在する"},
        },
    )
    async def register_device(
        request: DeviceRegisterRequest,
        api_key: str = Depends(verify_admin_key),
    ) -> DeviceRegisterResponse:
        """新規デバイスを登録しAPIキーを発行"""
        try:
            device, plain_api_key = app_state.device_auth_manager.register_device(
                device_id=request.device_id,
                zone=request.zone,
                location=request.location,
            )

            logger.info(f"Device registered via API: {device.id}")

            return DeviceRegisterResponse(
                device_id=device.id,
                api_key=plain_api_key,
                zone=device.zone,
                location=device.location,
            )
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            )

    @app.get(
        f"{config.api_prefix}/admin/devices/{{device_id}}",
        response_model=DeviceDetailResponse,
        tags=["Device Management"],
        summary="デバイス詳細（管理用）",
        responses={404: {"model": ErrorResponse, "description": "デバイスが見つからない"}},
    )
    async def get_device_detail(
        device_id: str,
        api_key: str = Depends(verify_admin_key),
    ) -> DeviceDetailResponse:
        """登録デバイスの詳細情報を取得"""
        device = app_state.device_auth_manager.get_device(device_id)
        if not device:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Device {device_id} not found",
            )

        status_info = app_state.device_auth_manager.get_device_status(device_id)

        return DeviceDetailResponse(
            device_id=device.id,
            zone=device.zone,
            location=device.location,
            enabled=device.enabled,
            status=status_info.status if status_info else "unknown",
            last_seen=status_info.last_seen.isoformat() if status_info and status_info.last_seen else None,
            packet_count=status_info.packet_count if status_info else 0,
            error_count=status_info.error_count if status_info else 0,
            firmware_version=status_info.firmware_version if status_info else None,
            uptime_seconds=status_info.uptime_seconds if status_info else None,
            created_at=device.created_at.isoformat() if device.created_at else None,
            updated_at=device.updated_at.isoformat() if device.updated_at else None,
        )

    @app.put(
        f"{config.api_prefix}/admin/devices/{{device_id}}",
        response_model=DeviceDetailResponse,
        tags=["Device Management"],
        summary="デバイス更新",
        responses={404: {"model": ErrorResponse, "description": "デバイスが見つからない"}},
    )
    async def update_device(
        device_id: str,
        request: DeviceUpdateRequest,
        api_key: str = Depends(verify_admin_key),
    ) -> DeviceDetailResponse:
        """デバイス情報を更新"""
        updated = app_state.device_auth_manager.update_device(
            device_id=device_id,
            zone=request.zone,
            location=request.location,
            enabled=request.enabled,
        )

        if not updated:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Device {device_id} not found",
            )

        logger.info(f"Device updated via API: {device_id}")

        status_info = app_state.device_auth_manager.get_device_status(device_id)

        return DeviceDetailResponse(
            device_id=updated.id,
            zone=updated.zone,
            location=updated.location,
            enabled=updated.enabled,
            status=status_info.status if status_info else "unknown",
            last_seen=status_info.last_seen.isoformat() if status_info and status_info.last_seen else None,
            packet_count=status_info.packet_count if status_info else 0,
            error_count=status_info.error_count if status_info else 0,
            firmware_version=status_info.firmware_version if status_info else None,
            uptime_seconds=status_info.uptime_seconds if status_info else None,
            created_at=updated.created_at.isoformat() if updated.created_at else None,
            updated_at=updated.updated_at.isoformat() if updated.updated_at else None,
        )

    @app.delete(
        f"{config.api_prefix}/admin/devices/{{device_id}}",
        response_model=DeviceDeleteResponse,
        tags=["Device Management"],
        summary="デバイス削除",
        responses={404: {"model": ErrorResponse, "description": "デバイスが見つからない"}},
    )
    async def delete_device(
        device_id: str,
        api_key: str = Depends(verify_admin_key),
    ) -> DeviceDeleteResponse:
        """デバイスを削除"""
        deleted = app_state.device_auth_manager.delete_device(device_id)

        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Device {device_id} not found",
            )

        logger.info(f"Device deleted via API: {device_id}")

        return DeviceDeleteResponse(device_id=device_id)

    @app.post(
        f"{config.api_prefix}/admin/devices/{{device_id}}/rotate-key",
        response_model=ApiKeyRotateResponse,
        tags=["Device Management"],
        summary="APIキーローテーション",
        responses={404: {"model": ErrorResponse, "description": "デバイスが見つからない"}},
    )
    async def rotate_device_api_key(
        device_id: str,
        api_key: str = Depends(verify_admin_key),
    ) -> ApiKeyRotateResponse:
        """デバイスのAPIキーをローテーション（再生成）"""
        new_api_key = app_state.device_auth_manager.rotate_api_key(device_id)

        if not new_api_key:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Device {device_id} not found",
            )

        logger.info(f"API key rotated via API for device: {device_id}")

        return ApiKeyRotateResponse(
            device_id=device_id,
            new_api_key=new_api_key,
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
    parser.add_argument(
        "--device-db", default=None, help="Path to device database file"
    )
    parser.add_argument(
        "--admin-api-key", default=None, help="Admin API key for device management"
    )
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
    if args.device_db:
        config.device_db_path = args.device_db
    if args.admin_api_key:
        config.admin_api_key = args.admin_api_key

    # 環境変数からAPIキーを読み込む
    env_api_key = os.environ.get("WAVIRA_API_KEY")
    if env_api_key and env_api_key not in config.api_keys:
        config.api_keys.append(env_api_key)

    # 環境変数から管理者APIキーを読み込む
    env_admin_api_key = os.environ.get("WAVIRA_ADMIN_API_KEY")
    if env_admin_api_key and not config.admin_api_key:
        config.admin_api_key = env_admin_api_key

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    print("=" * 60)
    print("ESP32 CSI API Server")
    print("=" * 60)
    print(f"Host: {config.host}")
    print(f"Port: {config.port}")
    print(f"API Prefix: {config.api_prefix}")
    print(f"Database: {config.db_path}")
    print(f"Device DB: {config.device_db_path}")
    print(f"Auth: {'Enabled' if config.api_keys else 'Disabled'}")
    print(f"Admin API: {'Enabled' if config.admin_api_key else 'Disabled'}")
    print(f"Rate Limit: {config.rate_limit_per_second} req/s per device")
    print("=" * 60)

    # セキュリティ警告を表示
    if not config.api_keys:
        print("\n" + "!" * 60)
        print("⚠️  WARNING: Authentication is DISABLED!")
        print("   This server is accepting unauthenticated requests.")
        print("   For production use, enable authentication with:")
        print("   --api-key YOUR_SECRET_KEY")
        print("   or set api_keys in config.yaml")
        print("!" * 60 + "\n")

    app = create_app(config)

    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level="debug" if args.debug else "info",
    )


if __name__ == "__main__":
    main()

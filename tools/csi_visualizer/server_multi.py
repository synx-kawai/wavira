#!/usr/bin/env python3
"""
Multi-ESP32 CSI Server

複数のESP32デバイスからCSIデータを収集し、
ゾーン別に人数推定を行うサーバー。

Issue #9: 複数ESP32によるマルチゾーン計測対応
"""

import asyncio
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set
import logging

import serial as pyserial
import websockets
import yaml

# Import breathing detector
sys.path.insert(0, str(Path(__file__).parent))
from breathing_detector import BreathingDetector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class DeviceConfig:
    """ESP32デバイス設定"""
    id: str
    name: str
    port: str
    baud: int = 115200
    zone: str = "default"
    position: Dict[str, float] = field(default_factory=lambda: {"x": 0, "y": 0})
    color: str = "#3fb950"
    enabled: bool = True


@dataclass
class ZoneConfig:
    """ゾーン設定"""
    name: str
    capacity: int = 10
    alert_threshold: int = 8


@dataclass
class DeviceState:
    """デバイスの状態"""
    device: DeviceConfig
    serial_conn: Optional[pyserial.Serial] = None
    breathing_detector: Optional[BreathingDetector] = None
    last_data: Optional[Dict] = None
    last_update: float = 0
    packet_count: int = 0
    connected: bool = False
    error_count: int = 0


class MultiDeviceServer:
    """複数ESP32デバイス対応サーバー"""

    CSI_PATTERN = re.compile(
        r'CSI_DATA,(\d+),([0-9a-fA-F:]+),(-?\d+),(\d+),'
        r'.*?\[([^\]]+)\]'
    )

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.devices: Dict[str, DeviceState] = {}
        self.zones: Dict[str, ZoneConfig] = {}
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.ws_port = 8765
        self.running = False

        self._load_config()

    def _load_config(self):
        """設定ファイルを読み込む"""
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}")
            return

        with open(self.config_path) as f:
            config = yaml.safe_load(f)

        # サーバー設定
        if "server" in config:
            self.ws_port = config["server"].get("websocket_port", 8765)

        # デバイス設定
        for dev_conf in config.get("devices", []):
            if not dev_conf.get("enabled", True):
                continue

            device = DeviceConfig(
                id=dev_conf["id"],
                name=dev_conf["name"],
                port=dev_conf["port"],
                baud=dev_conf.get("baud", 115200),
                zone=dev_conf.get("zone", "default"),
                position=dev_conf.get("position", {"x": 0, "y": 0}),
                color=dev_conf.get("color", "#3fb950"),
                enabled=dev_conf.get("enabled", True),
            )
            self.devices[device.id] = DeviceState(
                device=device,
                breathing_detector=BreathingDetector(sample_rate=10.0),
            )
            logger.info(f"Loaded device: {device.id} ({device.name}) on {device.port}")

        # ゾーン設定
        for zone_id, zone_conf in config.get("zones", {}).items():
            self.zones[zone_id] = ZoneConfig(
                name=zone_conf.get("name", zone_id),
                capacity=zone_conf.get("capacity", 10),
                alert_threshold=zone_conf.get("alert_threshold", 8),
            )
            logger.info(f"Loaded zone: {zone_id} ({self.zones[zone_id].name})")

    def _connect_device(self, state: DeviceState) -> bool:
        """デバイスに接続"""
        try:
            if state.serial_conn and state.serial_conn.is_open:
                state.serial_conn.close()

            state.serial_conn = pyserial.Serial(
                state.device.port,
                state.device.baud,
                timeout=0.1
            )
            state.connected = True
            state.error_count = 0
            logger.info(f"Connected to {state.device.id} on {state.device.port}")
            return True

        except Exception as e:
            state.connected = False
            state.error_count += 1
            if state.error_count <= 3:
                logger.error(f"Failed to connect to {state.device.id}: {e}")
            return False

    def _parse_csi(self, line: str, device_id: str) -> Optional[Dict]:
        """CSIデータをパース"""
        match = self.CSI_PATTERN.search(line)
        if not match:
            return None

        try:
            pkt = int(match.group(1))
            mac = match.group(2)
            rssi = int(match.group(3))
            channel = int(match.group(4))
            raw_data = match.group(5)

            # Parse amplitude data
            values = [int(x) for x in raw_data.replace(" ", "").split(",") if x.lstrip("-").isdigit()]

            # Extract amplitudes (every other value is I, Q)
            amps = []
            for i in range(0, len(values) - 1, 2):
                amp = (values[i] ** 2 + values[i + 1] ** 2) ** 0.5
                amps.append(amp)

            if not amps:
                return None

            return {
                "device_id": device_id,
                "pkt": pkt,
                "mac": mac,
                "rssi": rssi,
                "ch": channel,
                "amps": amps,
                "timestamp": time.time(),
            }

        except Exception as e:
            return None

    async def _read_device(self, state: DeviceState):
        """デバイスからデータを読み取る"""
        while self.running:
            if not state.connected:
                if not self._connect_device(state):
                    await asyncio.sleep(5)  # 再接続待ち
                    continue

            try:
                if state.serial_conn and state.serial_conn.in_waiting:
                    line = state.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                    if "CSI_DATA" in line:
                        data = self._parse_csi(line, state.device.id)
                        if data:
                            # Update breathing detector
                            if state.breathing_detector:
                                breath_state = state.breathing_detector.update(
                                    data["amps"],
                                    data["timestamp"]
                                )
                                data["breath"] = {
                                    "breathing": bool(breath_state.is_breathing),
                                    "present": bool(breath_state.is_present),
                                    "breath_ratio": float(breath_state.breath_ratio),
                                    "breath_rate": float(breath_state.breath_rate),
                                    "confidence": float(breath_state.confidence),
                                }

                            # Add device metadata
                            data["device_name"] = state.device.name
                            data["zone"] = state.device.zone
                            data["color"] = state.device.color

                            state.last_data = data
                            state.last_update = time.time()
                            state.packet_count += 1

                            # Broadcast to clients
                            await self._broadcast(data)

                await asyncio.sleep(0.01)

            except pyserial.SerialException as e:
                logger.error(f"Serial error on {state.device.id}: {e}")
                state.connected = False
                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Error reading {state.device.id}: {e}")
                await asyncio.sleep(0.1)

    async def _broadcast(self, data: Dict):
        """全クライアントにデータを送信"""
        if not self.clients:
            return

        message = json.dumps(data, ensure_ascii=False)
        dead_clients = set()

        for client in self.clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                dead_clients.add(client)

        self.clients -= dead_clients

    async def _handle_client(self, websocket: websockets.WebSocketServerProtocol):
        """WebSocketクライアントを処理"""
        self.clients.add(websocket)
        client_id = id(websocket)
        logger.info(f"Client connected: {client_id}")

        try:
            # 初期状態を送信
            status = self._get_status()
            await websocket.send(json.dumps(status, ensure_ascii=False))

            # 接続維持
            async for message in websocket:
                # クライアントからのコマンドを処理（将来の拡張用）
                try:
                    cmd = json.loads(message)
                    if cmd.get("type") == "get_status":
                        await websocket.send(json.dumps(self._get_status(), ensure_ascii=False))
                except json.JSONDecodeError:
                    pass

        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.discard(websocket)
            logger.info(f"Client disconnected: {client_id}")

    def _get_status(self) -> Dict:
        """現在の状態を取得"""
        devices_status = []
        zones_status = {}

        for device_id, state in self.devices.items():
            dev_status = {
                "id": device_id,
                "name": state.device.name,
                "zone": state.device.zone,
                "color": state.device.color,
                "connected": state.connected,
                "packet_count": state.packet_count,
                "last_update": state.last_update,
            }

            if state.last_data:
                dev_status["rssi"] = state.last_data.get("rssi")
                dev_status["breath"] = state.last_data.get("breath")

            devices_status.append(dev_status)

            # ゾーン別集計
            zone_id = state.device.zone
            if zone_id not in zones_status:
                zone_config = self.zones.get(zone_id, ZoneConfig(name=zone_id))
                zones_status[zone_id] = {
                    "name": zone_config.name,
                    "capacity": zone_config.capacity,
                    "alert_threshold": zone_config.alert_threshold,
                    "devices": [],
                    "total_present": 0,
                }

            zones_status[zone_id]["devices"].append(device_id)
            if state.last_data and state.last_data.get("breath", {}).get("present"):
                zones_status[zone_id]["total_present"] += 1

        return {
            "type": "status",
            "timestamp": time.time(),
            "devices": devices_status,
            "zones": zones_status,
            "total_clients": len(self.clients),
        }

    async def _status_broadcaster(self):
        """定期的にステータスを送信"""
        while self.running:
            await asyncio.sleep(5)  # 5秒ごと
            if self.clients:
                status = self._get_status()
                await self._broadcast(status)

    async def run(self):
        """サーバーを実行"""
        self.running = True

        print("=" * 50)
        print("Multi-ESP32 CSI Server")
        print("=" * 50)
        print(f"Devices: {len(self.devices)}")
        for dev_id, state in self.devices.items():
            print(f"  - {dev_id}: {state.device.name} ({state.device.port})")
        print(f"WebSocket: ws://0.0.0.0:{self.ws_port}")
        print("=" * 50)

        # WebSocketサーバー起動
        async with websockets.serve(self._handle_client, "0.0.0.0", self.ws_port):
            logger.info(f"WebSocket server started on port {self.ws_port}")

            # 各デバイスの読み取りタスクを開始
            tasks = []
            for state in self.devices.values():
                tasks.append(asyncio.create_task(self._read_device(state)))

            # ステータス送信タスク
            tasks.append(asyncio.create_task(self._status_broadcaster()))

            try:
                await asyncio.gather(*tasks)
            except asyncio.CancelledError:
                pass

        self.running = False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Multi-ESP32 CSI Server")
    parser.add_argument("--config", "-c", default="config.yaml",
                        help="Path to config file")
    args = parser.parse_args()

    server = MultiDeviceServer(args.config)

    if not server.devices:
        logger.error("No devices configured. Check config.yaml")
        sys.exit(1)

    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        logger.info("Server stopped")


if __name__ == "__main__":
    main()

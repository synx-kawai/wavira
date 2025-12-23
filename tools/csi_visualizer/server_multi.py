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
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
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


class DataRecorder:
    """履歴データをSQLiteに保存するクラス"""

    def __init__(self, db_path: str = "history.db"):
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self):
        """データベースを初期化"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # デバイス別の詳細データ
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS device_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                datetime TEXT NOT NULL,
                device_id TEXT NOT NULL,
                device_name TEXT,
                zone TEXT,
                rssi INTEGER,
                is_present INTEGER,
                breath_rate REAL,
                breath_ratio REAL,
                confidence REAL
            )
        ''')

        # ゾーン別の集計データ（1分ごと）
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS zone_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                datetime TEXT NOT NULL,
                zone TEXT NOT NULL,
                zone_name TEXT,
                device_count INTEGER,
                present_count INTEGER,
                avg_rssi REAL,
                crowd_level REAL
            )
        ''')

        # インデックス
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_device_timestamp ON device_history(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_zone_timestamp ON zone_history(timestamp)')

        conn.commit()
        conn.close()
        logger.info(f"Database initialized: {self.db_path}")

    def save_device_data(self, data: Dict):
        """デバイスデータを保存"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            breath = data.get("breath", {})
            cursor.execute('''
                INSERT INTO device_history
                (timestamp, datetime, device_id, device_name, zone, rssi,
                 is_present, breath_rate, breath_ratio, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data.get("timestamp", time.time()),
                datetime.now().isoformat(),
                data.get("device_id"),
                data.get("device_name"),
                data.get("zone"),
                data.get("rssi"),
                1 if breath.get("present") else 0,
                breath.get("breath_rate", 0),
                breath.get("breath_ratio", 0),
                breath.get("confidence", 0),
            ))

            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to save device data: {e}")

    def save_zone_summary(self, zone_id: str, zone_name: str,
                          device_count: int, present_count: int,
                          avg_rssi: float, crowd_level: float):
        """ゾーン集計データを保存"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO zone_history
                (timestamp, datetime, zone, zone_name, device_count,
                 present_count, avg_rssi, crowd_level)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                time.time(),
                datetime.now().isoformat(),
                zone_id,
                zone_name,
                device_count,
                present_count,
                avg_rssi,
                crowd_level,
            ))

            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to save zone summary: {e}")

    def get_device_history(self, device_id: str = None,
                           hours: int = 24, limit: int = 1000) -> List[Dict]:
        """デバイス履歴を取得"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        since = time.time() - (hours * 3600)

        if device_id:
            cursor.execute('''
                SELECT * FROM device_history
                WHERE timestamp > ? AND device_id = ?
                ORDER BY timestamp DESC LIMIT ?
            ''', (since, device_id, limit))
        else:
            cursor.execute('''
                SELECT * FROM device_history
                WHERE timestamp > ?
                ORDER BY timestamp DESC LIMIT ?
            ''', (since, limit))

        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return rows

    def get_zone_history(self, zone_id: str = None,
                         hours: int = 24, limit: int = 1000) -> List[Dict]:
        """ゾーン履歴を取得"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        since = time.time() - (hours * 3600)

        if zone_id:
            cursor.execute('''
                SELECT * FROM zone_history
                WHERE timestamp > ? AND zone = ?
                ORDER BY timestamp DESC LIMIT ?
            ''', (since, zone_id, limit))
        else:
            cursor.execute('''
                SELECT * FROM zone_history
                WHERE timestamp > ?
                ORDER BY timestamp DESC LIMIT ?
            ''', (since, limit))

        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return rows

    def get_hourly_summary(self, hours: int = 24) -> List[Dict]:
        """時間別サマリーを取得"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        since = time.time() - (hours * 3600)

        cursor.execute('''
            SELECT
                strftime('%Y-%m-%d %H:00', datetime) as hour,
                zone,
                AVG(present_count) as avg_present,
                MAX(present_count) as max_present,
                AVG(crowd_level) as avg_crowd_level
            FROM zone_history
            WHERE timestamp > ?
            GROUP BY hour, zone
            ORDER BY hour DESC
        ''', (since,))

        rows = []
        for row in cursor.fetchall():
            rows.append({
                "hour": row[0],
                "zone": row[1],
                "avg_present": row[2],
                "max_present": row[3],
                "avg_crowd_level": row[4],
            })

        conn.close()
        return rows


class MultiDeviceServer:
    """複数ESP32デバイス対応サーバー"""

    CSI_PATTERN = re.compile(
        r'CSI_DATA,(\d+),([0-9a-fA-F:]+),(-?\d+),(\d+),'
        r'.*?\[([^\]]+)\]'
    )

    def __init__(self, config_path: str = "config.yaml", db_path: str = "history.db"):
        self.config_path = Path(config_path)
        self.devices: Dict[str, DeviceState] = {}
        self.zones: Dict[str, ZoneConfig] = {}
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.ws_port = 8765
        self.running = False
        self.recorder = DataRecorder(db_path)
        self.last_save_time = 0
        self.save_interval = 10  # デバイスデータ保存間隔（秒）
        self.zone_save_interval = 60  # ゾーンサマリー保存間隔（秒）
        self.last_zone_save_time = 0

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

                            # Save to database (throttled)
                            current_time = time.time()
                            if current_time - self.last_save_time >= self.save_interval:
                                self.recorder.save_device_data(data)
                                self.last_save_time = current_time

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
                # クライアントからのコマンドを処理
                try:
                    cmd = json.loads(message)
                    cmd_type = cmd.get("type")

                    if cmd_type == "get_status":
                        await websocket.send(json.dumps(self._get_status(), ensure_ascii=False))

                    elif cmd_type == "get_history":
                        # 履歴データを取得
                        hours = cmd.get("hours", 24)
                        history_type = cmd.get("history_type", "zone")  # "zone" or "device"
                        zone_id = cmd.get("zone_id")
                        device_id = cmd.get("device_id")

                        if history_type == "zone":
                            data = self.recorder.get_zone_history(zone_id, hours)
                        else:
                            data = self.recorder.get_device_history(device_id, hours)

                        response = {
                            "type": "history",
                            "history_type": history_type,
                            "data": data,
                        }
                        await websocket.send(json.dumps(response, ensure_ascii=False))

                    elif cmd_type == "get_hourly_summary":
                        hours = cmd.get("hours", 24)
                        data = self.recorder.get_hourly_summary(hours)
                        response = {
                            "type": "hourly_summary",
                            "data": data,
                        }
                        await websocket.send(json.dumps(response, ensure_ascii=False))

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

    async def _zone_recorder(self):
        """定期的にゾーンサマリーを保存"""
        while self.running:
            await asyncio.sleep(self.zone_save_interval)

            # 各ゾーンの集計を保存
            zones_data = {}
            for device_id, state in self.devices.items():
                zone_id = state.device.zone
                if zone_id not in zones_data:
                    zone_config = self.zones.get(zone_id, ZoneConfig(name=zone_id))
                    zones_data[zone_id] = {
                        "name": zone_config.name,
                        "capacity": zone_config.capacity,
                        "devices": [],
                        "rssi_values": [],
                        "present_count": 0,
                    }

                zones_data[zone_id]["devices"].append(device_id)

                if state.last_data:
                    rssi = state.last_data.get("rssi")
                    if rssi is not None:
                        zones_data[zone_id]["rssi_values"].append(rssi)

                    if state.last_data.get("breath", {}).get("present"):
                        zones_data[zone_id]["present_count"] += 1

            # 保存
            for zone_id, zdata in zones_data.items():
                device_count = len(zdata["devices"])
                present_count = zdata["present_count"]
                avg_rssi = sum(zdata["rssi_values"]) / len(zdata["rssi_values"]) if zdata["rssi_values"] else 0
                crowd_level = present_count / zdata["capacity"] if zdata["capacity"] > 0 else 0

                self.recorder.save_zone_summary(
                    zone_id=zone_id,
                    zone_name=zdata["name"],
                    device_count=device_count,
                    present_count=present_count,
                    avg_rssi=avg_rssi,
                    crowd_level=crowd_level,
                )

            logger.info(f"Saved zone summary for {len(zones_data)} zones")

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

            # ゾーンサマリー保存タスク
            tasks.append(asyncio.create_task(self._zone_recorder()))

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
    parser.add_argument("--db", "-d", default="history.db",
                        help="Path to history database")
    args = parser.parse_args()

    server = MultiDeviceServer(args.config, args.db)

    if not server.devices:
        logger.error("No devices configured. Check config.yaml")
        sys.exit(1)

    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        logger.info("Server stopped")


if __name__ == "__main__":
    main()

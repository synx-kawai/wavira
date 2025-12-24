#!/usr/bin/env python3
"""
MQTT to WebSocket Bridge for Wavira CSI Dashboard

ESP32デバイスからMQTT経由で受信したCSIデータを
WebSocketクライアント（ダッシュボード）に転送するブリッジサーバー。

Usage:
    python mqtt_ws_bridge.py --mqtt-host 192.168.2.197 --ws-port 8765
"""

import argparse
import asyncio
import json
import logging
import signal
import sys
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from queue import Queue

import paho.mqtt.client as mqtt
import websockets

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class DeviceState:
    """デバイスの状態"""
    id: str
    name: str
    zone: str = "default"
    color: str = "#3fb950"
    rssi: int = -100
    amps: list = field(default_factory=list)
    breath: dict = field(default_factory=dict)
    last_update: float = 0
    connected: bool = True


class MQTTWebSocketBridge:
    """MQTT to WebSocket Bridge"""

    # 履歴データの最大保存数（2分間 x 約10Hz = 1200データポイント）
    HISTORY_MAX_SIZE = 1200

    def __init__(self, mqtt_host: str, mqtt_port: int, ws_port: int):
        self.mqtt_host = mqtt_host
        self.mqtt_port = mqtt_port
        self.ws_port = ws_port

        self.devices: Dict[str, DeviceState] = {}
        self.zones = {
            "default": {
                "name": "デフォルトゾーン",
                "capacity": 10,
                "alert_threshold": 8,
                "total_present": 0
            }
        }
        self.ws_clients: Set = set()
        self.mqtt_client: Optional[mqtt.Client] = None
        self.running = False
        self.message_queue: Queue = Queue()
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        # デバイスごとの履歴データ（リングバッファ）
        self.device_history: Dict[str, deque] = {}

    def _on_mqtt_connect(self, client, userdata, flags, rc):
        """MQTT接続時コールバック"""
        if rc == 0:
            logger.info(f"Connected to MQTT broker at {self.mqtt_host}:{self.mqtt_port}")
            client.subscribe("wavira/#")
            logger.info("Subscribed to wavira/#")
        else:
            logger.error(f"MQTT connection failed with code {rc}")

    def _on_mqtt_message(self, client, userdata, msg):
        """MQTTメッセージ受信コールバック"""
        # キューに追加してasyncioループで処理
        self.message_queue.put((msg.topic, msg.payload))

    def _update_device(self, device_id: str, data: dict):
        """デバイスデータを更新"""
        now = time.time()

        if device_id not in self.devices:
            self.devices[device_id] = DeviceState(
                id=device_id,
                name=device_id,
            )
            logger.info(f"New device registered: {device_id}")

        dev = self.devices[device_id]
        dev.rssi = data.get("rssi", -100)
        dev.last_update = now
        dev.connected = True

        # CSIデータから振幅を計算
        csi_data = data.get("data", [])
        if csi_data:
            # I/Qペアから振幅を計算
            amps = []
            for i in range(0, len(csi_data) - 1, 2):
                i_val = csi_data[i]
                q_val = csi_data[i + 1] if i + 1 < len(csi_data) else 0
                amp = (i_val ** 2 + q_val ** 2) ** 0.5
                amps.append(amp)
            dev.amps = amps

        # 簡易的な人検出（振幅の分散が大きければ人がいると判定）
        if dev.amps:
            avg_amp = sum(dev.amps) / len(dev.amps)
            variance = sum((a - avg_amp) ** 2 for a in dev.amps) / len(dev.amps)
            dev.breath = {
                "present": variance > 50,
                "breath_ratio": min(variance / 500, 1.0)
            }

        # 履歴データに追加
        self._add_to_history(device_id, dev)

        return dev

    def _add_to_history(self, device_id: str, dev: DeviceState):
        """履歴データに追加"""
        if device_id not in self.device_history:
            self.device_history[device_id] = deque(maxlen=self.HISTORY_MAX_SIZE)

        history_entry = {
            "device_id": dev.id,
            "device_name": dev.name,
            "zone": dev.zone,
            "color": dev.color,
            "rssi": dev.rssi,
            "amps": dev.amps[:64] if dev.amps else [],
            "breath": dev.breath,
            "timestamp": time.time()
        }
        self.device_history[device_id].append(history_entry)

    async def _broadcast_device_data(self, dev: DeviceState):
        """WebSocketクライアントにデバイスデータをブロードキャスト"""
        if not self.ws_clients:
            return

        data = {
            "device_id": dev.id,
            "device_name": dev.name,
            "zone": dev.zone,
            "color": dev.color,
            "rssi": dev.rssi,
            "amps": dev.amps[:64] if dev.amps else [],
            "breath": dev.breath,
        }

        message = json.dumps(data)
        disconnected = set()

        for ws in list(self.ws_clients):
            try:
                await ws.send(message)
            except Exception:
                disconnected.add(ws)

        self.ws_clients -= disconnected

    async def _send_status(self, ws):
        """ステータス情報を送信"""
        devices_list = []
        for dev in self.devices.values():
            devices_list.append({
                "id": dev.id,
                "name": dev.name,
                "zone": dev.zone,
                "color": dev.color,
                "rssi": dev.rssi,
                "connected": dev.connected,
            })

        status = {
            "type": "status",
            "devices": devices_list,
            "zones": self.zones,
        }

        try:
            await ws.send(json.dumps(status))
        except Exception:
            pass

    async def _send_history(self, ws):
        """履歴データを送信"""
        for device_id, history in self.device_history.items():
            if not history:
                continue

            # 履歴データをまとめて送信
            history_data = {
                "type": "history",
                "device_id": device_id,
                "data": list(history)
            }

            try:
                await ws.send(json.dumps(history_data))
                logger.info(f"Sent {len(history)} history entries for {device_id}")
            except Exception as e:
                logger.error(f"Failed to send history: {e}")

    async def _handle_ws_client(self, ws):
        """WebSocketクライアントを処理"""
        self.ws_clients.add(ws)
        client_addr = ws.remote_address
        logger.info(f"WebSocket client connected: {client_addr}")

        try:
            # 初期ステータスを送信
            await self._send_status(ws)

            # 履歴データを送信（グラフ復元用）
            await self._send_history(ws)

            # クライアントからのメッセージを処理
            async for message in ws:
                try:
                    data = json.loads(message)
                    if data.get("type") == "get_hourly_summary":
                        await ws.send(json.dumps({
                            "type": "hourly_summary",
                            "data": []
                        }))
                except json.JSONDecodeError:
                    pass

        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.ws_clients.discard(ws)
            logger.info(f"WebSocket client disconnected: {client_addr}")

    async def _process_mqtt_messages(self):
        """MQTTメッセージを処理"""
        while self.running:
            try:
                # キューからメッセージを取得（非ブロッキング）
                while not self.message_queue.empty():
                    topic, payload = self.message_queue.get_nowait()
                    try:
                        data = json.loads(payload.decode())
                        if topic.startswith("wavira/csi/"):
                            device_id = data.get("id", topic.split("/")[-1])
                            dev = self._update_device(device_id, data)
                            await self._broadcast_device_data(dev)
                        elif topic.startswith("wavira/device/") and topic.endswith("/status"):
                            device_id = topic.split("/")[2]
                            logger.info(f"Device {device_id} status: {data}")
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
            except Exception:
                pass
            await asyncio.sleep(0.05)

    async def _periodic_status_broadcast(self):
        """定期的にステータスをブロードキャスト"""
        while self.running:
            await asyncio.sleep(1)

            # デバイスの接続状態を更新
            now = time.time()
            for dev in self.devices.values():
                if now - dev.last_update > 5:
                    dev.connected = False

            # ステータスをブロードキャスト
            for ws in list(self.ws_clients):
                await self._send_status(ws)

    def _start_mqtt(self):
        """MQTTクライアントを開始"""
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_connect = self._on_mqtt_connect
        self.mqtt_client.on_message = self._on_mqtt_message

        try:
            self.mqtt_client.connect(self.mqtt_host, self.mqtt_port, 60)
            self.mqtt_client.loop_start()
            logger.info("MQTT client started")
        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {e}")
            raise

    async def run(self):
        """ブリッジサーバーを実行"""
        self.running = True
        self.loop = asyncio.get_event_loop()

        # MQTTクライアントを開始
        self._start_mqtt()

        # WebSocketサーバーを開始
        server = await websockets.serve(
            self._handle_ws_client,
            "0.0.0.0",
            self.ws_port
        )
        logger.info(f"WebSocket server started on ws://0.0.0.0:{self.ws_port}")

        # バックグラウンドタスクを開始
        mqtt_task = asyncio.create_task(self._process_mqtt_messages())
        status_task = asyncio.create_task(self._periodic_status_broadcast())

        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            pass
        finally:
            self.running = False
            mqtt_task.cancel()
            status_task.cancel()
            server.close()
            await server.wait_closed()
            if self.mqtt_client:
                self.mqtt_client.loop_stop()
                self.mqtt_client.disconnect()


def main():
    parser = argparse.ArgumentParser(description="MQTT to WebSocket Bridge")
    parser.add_argument("--mqtt-host", default="192.168.2.197", help="MQTT broker host")
    parser.add_argument("--mqtt-port", type=int, default=1883, help="MQTT broker port")
    parser.add_argument("--ws-port", type=int, default=8765, help="WebSocket server port")
    args = parser.parse_args()

    bridge = MQTTWebSocketBridge(args.mqtt_host, args.mqtt_port, args.ws_port)

    def signal_handler(sig, frame):
        logger.info("Shutting down...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info(f"Starting MQTT-WebSocket Bridge")
    logger.info(f"  MQTT: {args.mqtt_host}:{args.mqtt_port}")
    logger.info(f"  WebSocket: ws://0.0.0.0:{args.ws_port}")

    asyncio.run(bridge.run())


if __name__ == "__main__":
    main()

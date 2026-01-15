# ESP32 CSI Receiver Firmware

ESP32でWi-Fi CSI（Channel State Information）を収集し、MQTTでブローカーに送信するファームウェアです。

## 必要条件

- ESP-IDF v5.0以上
- ESP32/ESP32-S3/ESP32-C3デバイス
- MQTTブローカー（Mosquitto推奨）

## アーキテクチャ

```
ESP32 (CSI収集) --MQTT--> Mosquitto ---> CSI Processor / History Collector
                              |
                              +--> Dashboard (WebSocket)
```

## クイックスタート

### 1. ESP-IDF環境のセットアップ

```bash
# ESP-IDFを有効化
source $HOME/esp/esp-idf/export.sh
```

### 2. ファームウェアの設定

```bash
cd esp-csi/examples/get-started/csi_recv
idf.py menuconfig
```

### 3. 必須設定項目

menuconfigで以下を設定してください：

#### Wi-Fi設定
`Wavira CSI Configuration` → `Wi-Fi Configuration`

| 項目 | 説明 | 例 |
|------|------|-----|
| `WiFi SSID` | 接続するWi-FiのSSID | `MyHomeWiFi` |
| `WiFi Password` | Wi-Fiパスワード | `password123` |

#### MQTT設定
`Wavira CSI Configuration` → `MQTT Configuration`

| 項目 | 説明 | 例 |
|------|------|-----|
| `MQTT Broker URL` | MQTTブローカーのURL | `mqtt://192.168.1.100:1883` |
| `MQTT Username` | ユーザー名（オプション） | `wavira` |
| `MQTT Password` | パスワード（オプション） | `secret` |

#### デバイス設定
`Wavira CSI Configuration` → `Device Configuration`

| 項目 | 説明 | 例 |
|------|------|-----|
| `Device ID` | デバイス識別子 | `esp32-living-room` |
| `Device Zone` | 設置ゾーン | `living-room` |

### 4. ビルドとフラッシュ

```bash
# ビルド
idf.py build

# フラッシュ（ポートは環境に合わせて変更）
idf.py -p /dev/cu.usbserial-10 flash

# シリアルモニター
idf.py -p /dev/cu.usbserial-10 monitor
```

## MQTTブローカーのセットアップ

### Docker Compose（推奨）

```bash
cd tools/csi_visualizer
docker-compose up -d
```

これにより以下のサービスが起動します：
- Mosquitto（MQTT: 1883, WebSocket: 9001）
- CSI Processor
- History Collector（REST API: 8080）
- Dashboard（HTTP: 80）

### 手動セットアップ

```bash
# Mosquittoのインストール（macOS）
brew install mosquitto

# 起動
mosquitto -c /usr/local/etc/mosquitto/mosquitto.conf
```

## MQTTトピック

| トピック | 方向 | 説明 |
|----------|------|------|
| `wavira/csi/{device_id}` | Publish | CSI生データ |
| `wavira/device/{device_id}/status` | Publish | デバイスステータス |
| `wavira/analysis/{device_id}` | Subscribe | 解析結果 |

## LEDステータスインジケーター

| 状態 | LED | 説明 |
|------|-----|------|
| 起動中 | 赤点灯 | ブート処理中 |
| Wi-Fi待機 | 黄点滅 | Wi-Fi接続待ち |
| MQTT待機 | 青点滅 | MQTT接続待ち |
| 接続済み | 緑点灯 | 正常動作中 |
| データ送信 | 緑パルス | CSIデータ送信時 |
| エラー | 赤点滅 | エラー発生 |

## 動作モード

### CSIトリガーモード

| モード | 説明 | 設定 |
|--------|------|------|
| Router Ping | ルーターにpingしてCSIを取得（推奨） | `CONFIG_CSI_TRIGGER_ROUTER=y` |
| ESP-NOW | 別のESP32からパケットを送信 | `CONFIG_CSI_TRIGGER_ESPNOW=y` |

Router Pingモードは単一のESP32で動作するため、開発・テストに便利です。

## トラブルシューティング

### Wi-Fi接続失敗

```
E (xxxx) wavira_csi: Failed to connect to Wi-Fi SSID: xxx
```

- SSIDとパスワードが正しいか確認
- ルーターが2.4GHz帯に対応しているか確認
- ESP32とルーターの距離を近づける

### MQTT接続失敗

```
E (xxxx) wavira_csi: MQTT connection failed
```

- MQTTブローカーが起動しているか確認
- URLとポートが正しいか確認
- ファイアウォール設定を確認

### スタックオーバーフロー

```
***ERROR*** A stack overflow in task xxx has been detected.
```

このファームウェアでは適切なスタックサイズが設定済みです。
カスタマイズする場合は`app_main.c`の`xTaskCreate`を確認してください。

## 設定例

### 開発環境

```
CONFIG_WAVIRA_WIFI_SSID="DevNetwork"
CONFIG_WAVIRA_WIFI_PASSWORD="devpass123"
CONFIG_WAVIRA_MQTT_BROKER_URL="mqtt://192.168.1.100:1883"
CONFIG_WAVIRA_DEVICE_ID="esp32-dev-01"
CONFIG_CSI_TRIGGER_ROUTER=y
```

### 本番環境

```
CONFIG_WAVIRA_WIFI_SSID="ProductionNetwork"
CONFIG_WAVIRA_WIFI_PASSWORD="securepassword"
CONFIG_WAVIRA_MQTT_BROKER_URL="mqtt://wavira-mqtt.local:1883"
CONFIG_WAVIRA_DEVICE_ID="esp32-zone-a-01"
CONFIG_WAVIRA_DEVICE_ZONE="zone-a"
```

## ファイル構成

```
csi_recv/
├── main/
│   ├── app_main.c          # メインアプリケーション
│   ├── CMakeLists.txt      # ビルド設定
│   ├── Kconfig.projbuild   # menuconfig定義
│   └── idf_component.yml   # 依存コンポーネント
├── sdkconfig.defaults      # デフォルト設定
└── README.md               # このファイル
```

## 関連ドキュメント

- [Wavira CSI Visualizer](../../../tools/csi_visualizer/README.md)
- [ESP-IDF ドキュメント](https://docs.espressif.com/projects/esp-idf/)
- [ESP32 CSI Guide](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-guides/wifi.html#wi-fi-channel-state-information)

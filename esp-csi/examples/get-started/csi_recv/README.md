# ESP32 CSI Receiver Firmware

ESP32でWi-Fi CSI（Channel State Information）を収集し、HTTPでサーバーに送信するファームウェアです。

## 必要条件

- ESP-IDF v5.0以上
- ESP32/ESP32-S3/ESP32-C3デバイス
- Wavira APIサーバー（`tools/csi_visualizer/api_server.py`）

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

#### サーバー設定
`Wavira CSI Configuration` → `HTTP Server Configuration`

| 項目 | 説明 | 例 |
|------|------|-----|
| `Server URL` | CSIデータ送信先 | `http://192.168.1.100:8080/api/v1/csi` |
| `Batch Endpoint` | バッチ送信先 | `http://192.168.1.100:8080/api/v1/csi/batch` |
| `API Key` | 認証用APIキー | `wvr_xxxx...` |

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

## APIキーの取得方法

### 方法1: サーバー起動時に自動生成

```bash
cd tools/csi_visualizer
python api_server.py --port 8080
```

起動ログに表示されるAdmin APIキーを使用できます。

### 方法2: デバイス登録APIで取得

```bash
# デバイスを登録してAPIキーを取得
curl -X POST http://localhost:8080/api/v1/admin/devices \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_ADMIN_API_KEY" \
  -d '{
    "device_id": "esp32-001",
    "name": "Living Room ESP32",
    "zone": "living-room"
  }'
```

レスポンスに含まれる`api_key`をファームウェアに設定します。

## 動作モード

### CSIトリガーモード

| モード | 説明 | 設定 |
|--------|------|------|
| Router Ping | ルーターにpingしてCSIを取得（推奨） | `CONFIG_CSI_TRIGGER_ROUTER=y` |
| ESP-NOW | 別のESP32からパケットを送信 | `CONFIG_CSI_TRIGGER_ESPNOW=y` |

Router Pingモードは単一のESP32で動作するため、開発・テストに便利です。

### 出力モード

| モード | 説明 | 設定 |
|--------|------|------|
| HTTP | Wi-Fi経由でサーバーに送信 | `CONFIG_CSI_OUTPUT_HTTP=y` |
| Serial | UARTで出力（デバッグ用） | `CONFIG_CSI_OUTPUT_SERIAL=y` |

## トラブルシューティング

### Wi-Fi接続失敗

```
E (xxxx) wavira_csi: Failed to connect to Wi-Fi SSID: xxx
```

- SSIDとパスワードが正しいか確認
- ルーターが2.4GHz帯に対応しているか確認
- ESP32とルーターの距離を近づける

### サーバー接続失敗

```
E (xxxx) wavira_csi: HTTP POST failed
```

- サーバーが起動しているか確認
- IPアドレスとポートが正しいか確認
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
CONFIG_WAVIRA_SERVER_URL="http://192.168.1.100:8080/api/v1/csi"
CONFIG_WAVIRA_DEVICE_ID="esp32-dev-01"
CONFIG_CSI_TRIGGER_ROUTER=y
```

### 本番環境

```
CONFIG_WAVIRA_WIFI_SSID="ProductionNetwork"
CONFIG_WAVIRA_WIFI_PASSWORD="securepassword"
CONFIG_WAVIRA_SERVER_URL="http://wavira-server.local:8080/api/v1/csi"
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

- [Wavira APIサーバー](../../../tools/csi_visualizer/README.md)
- [ESP-IDF ドキュメント](https://docs.espressif.com/projects/esp-idf/)
- [ESP32 CSI Guide](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-guides/wifi.html#wi-fi-channel-state-information)

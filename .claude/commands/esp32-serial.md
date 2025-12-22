# ESP32 シリアルモニタリング ガイド

ESP32のシリアル接続とCSIデータ収集を支援するスキルです。

## 使用方法

このスキルは以下のタスクをサポートします：
1. ESP32接続テスト
2. シリアルポート検出
3. CSIデータ収集
4. トラブルシューティング

## タスク別手順

### 1. シリアルポートの検出

```bash
# 利用可能なシリアルポートを確認
ls /dev/cu.usb*
```

### 2. ESP32接続テスト

```python
# Pythonで接続テスト
python3 -c "
from wavira.utils.esp32_serial import test_connection
test_connection()
"
```

### 3. CSIデータ収集（混雑レベル）

```bash
# レベル0（空き）のデータ収集
python scripts/collect_crowd.py --level 0 --location office

# レベル1（やや混雑）のデータ収集
python scripts/collect_crowd.py --level 1 --location office

# レベル2（混雑）のデータ収集
python scripts/collect_crowd.py --level 2 --location office
```

### 4. Wi-Fi設定の変更

ESP32のWi-Fi設定を変更する場合：

```bash
# sdkconfigのWi-Fi設定を確認
grep -E "WIFI_SSID|WIFI_PASSWORD" esp-csi/examples/get-started/csi_recv_router/sdkconfig

# 設定を変更後、再ビルド＆フラッシュ
cd esp-csi/examples/get-started/csi_recv_router
source ~/esp/esp-idf/export.sh
idf.py build
idf.py -p /dev/cu.usbserial-XXXX flash
```

## トラブルシューティング

### フリーズ防止のベストプラクティス

ESP32Serialモジュールは以下のフリーズ防止機能を実装しています：

1. **非ブロッキング読み取り**
   - `in_waiting` でデータ存在を確認後に読み取り
   - 50ms間隔でシャットダウンフラグをチェック
   - `readline()` のブロッキングを回避

2. **スレッド終了処理**
   - `threading.Event` によるシャットダウン通知
   - 3秒タイムアウトでスレッド終了を待機
   - 終了しない場合は警告ログを出力

3. **Ctrl+C 対応**
   - シグナルハンドラーは呼び出し元で設定
   - ESP32Serialはシグナルを直接ハンドルしない
   - `stop()` を呼び出すだけで安全に終了

### フリーズした場合の対処

1. **Ctrl+C で終了できない場合**
   ```bash
   # プロセスIDを確認
   ps aux | grep python

   # 強制終了
   kill -9 <PID>
   ```

2. **シリアルポートがロックされている場合**
   ```bash
   # ポートを使用しているプロセスを確認
   lsof /dev/cu.usbserial-*

   # プロセスを終了
   kill <PID>
   ```

3. **ESP32のリセット**
   - USBケーブルを抜き差し
   - または EN ボタンを押す

4. **バッファのクリア**
   ```python
   import serial
   s = serial.Serial('/dev/cu.usbserial-XXXX', 115200)
   s.reset_input_buffer()
   s.reset_output_buffer()
   s.close()
   ```

### Wi-Fi接続に失敗する場合

1. **SSIDとパスワードを確認**
   ```bash
   grep -E "WIFI_SSID|WIFI_PASSWORD" esp-csi/.../sdkconfig
   ```

2. **2.4GHz帯であることを確認**
   - ESP32は5GHz帯に非対応

3. **認証方式を確認**
   - WPA2-PSKまたはWPA/WPA2混合が推奨

### CSIデータが来ない場合

1. **ESP32がWi-Fiに接続されているか確認**
   - シリアルモニターで "got ip" を確認

2. **ルーターからのパケットがあるか確認**
   - Ping応答でCSIが生成される

## ESP32Serialモジュールの機能

`ESP32Serial`モジュールには以下の機能があります：

- **自動ポート検出**: ポート指定不要
- **自動再接続**: 接続切断時に自動復旧
- **非ブロッキング読み取り**: フリーズ防止
- **スレッドセーフな終了**: Ctrl+Cで安全に終了
- **コールバック機能**: CSIデータ/エラー/状態変更を通知

## 使用例

### 基本的な使い方

```python
from wavira.utils.esp32_serial import ESP32Serial, CSIPacket
import signal

# ハンドラを作成
esp32 = ESP32Serial(auto_reconnect=True)

# コールバックを登録
def on_csi(packet: CSIPacket):
    print(f"CSI received: RSSI={packet.rssi}")

esp32.add_csi_callback(on_csi)

# Ctrl+C でクリーンに終了するためのシグナルハンドラー
def shutdown(signum, frame):
    print("Shutting down...")
    esp32.stop()

signal.signal(signal.SIGINT, shutdown)

# 接続して開始
esp32.connect()
esp32.start()

# メインループ (実際の処理)
import time
try:
    while esp32.is_connected:
        time.sleep(0.1)
except KeyboardInterrupt:
    pass
finally:
    esp32.stop()
```

### 接続テストのみ

```python
from wavira.utils.esp32_serial import test_connection

# Ctrl+Cで中断可能な接続テスト
result = test_connection(timeout=10.0)
print(f"Test result: {'OK' if result else 'FAILED'}")
```

## 注意事項

- シグナルハンドラーは必ず呼び出し元で設定してください
- `esp32.stop()` を finally ブロックで呼び出して確実にクリーンアップ
- 長時間のタイムアウト設定は避け、短いタイムアウトでループを使用

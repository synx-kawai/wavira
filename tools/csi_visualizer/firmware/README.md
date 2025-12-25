# ESP32 CSI Firmware

## esp32_csi_firmware.bin

正常に動作するESP32 CSIファームウェアのバックアップです。

### 仕様
- ボーレート: 115200
- WiFi SSID: msdfreeap
- MQTT Broker: mqtt://52.196.197.127
- 出力形式: JSON over MQTT

### MQTT設定 (v2 - 2025-12-25)
- Keepalive: 30秒 (変更前: 120秒)
- Network Timeout: 15秒 (変更前: 30秒)
- Reconnect Timeout: 5秒 (変更前: 10秒)

> 接続安定性向上のため、keepalive間隔を短縮しました

### 書き込み方法

```bash
esptool --chip esp32 --port /dev/cu.usbserial-XX --baud 460800 \
  write-flash 0x0 esp32_csi_firmware.bin
```

### 注意
- esp-csi/examples/get-started/csi_recv_router/ のビルドは使用しないでください
- 新しいESP32を追加する場合は、このファームウェアを使用してください

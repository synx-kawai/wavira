# ESP32 CSI Firmware

## esp32_csi_firmware.bin

正常に動作するESP32 CSIファームウェアのバックアップです。

### 仕様
- ボーレート: 115200
- WiFi SSID: <YOUR_WIFI_SSID>
- WiFi Password: <YOUR_WIFI_PASSWORD>
- 出力形式: CSI_DATA CSV形式

### 書き込み方法

```bash
esptool --chip esp32 --port /dev/cu.usbserial-XX --baud 460800 \
  write-flash 0x0 esp32_csi_firmware.bin
```

### 注意
- esp-csi/examples/get-started/csi_recv_router/ のビルドは使用しないでください
- 新しいESP32を追加する場合は、このファームウェアを使用してください

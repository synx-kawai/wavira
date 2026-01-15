# Troubleshooting Guide

Common issues and their solutions when working with Wavira.

## ESP32 Issues

### Device Not Detected

**Symptoms**: Serial port not showing up, device not responding

**Solutions**:
1. Check USB cable (use data cable, not charge-only)
2. Try different USB port
3. Install USB-to-Serial drivers:
   - macOS: Usually automatic
   - Windows: [CP210x driver](https://www.silabs.com/developers/usb-to-uart-bridge-vcp-drivers)
   - Linux: Usually built-in, check `dmesg` for device
4. Check port permissions (Linux):
   ```bash
   sudo usermod -a -G dialout $USER
   # Log out and back in
   ```

### No CSI Data

**Symptoms**: Device connected but no CSI output

**Solutions**:
1. Verify Wi-Fi connection:
   ```
   # Check serial output for WiFi status
   WiFi connected, IP: 192.168.x.x
   ```
2. Check MQTT broker is running:
   ```bash
   mosquitto_sub -h localhost -t "wavira/#" -v
   ```
3. Verify device configuration in menuconfig:
   - WiFi SSID/Password
   - MQTT broker address
4. LED status indicators:
   - Blue blinking: Connecting
   - Green solid: Connected
   - Red: Error

### Firmware Flash Failure

**Symptoms**: esptool.py errors, flash write failed

**Solutions**:
1. Hold BOOT button while flashing
2. Reduce baud rate:
   ```bash
   idf.py -p /dev/cu.usbserial-0001 -b 115200 flash
   ```
3. Erase flash first:
   ```bash
   esptool.py --chip esp32 erase_flash
   ```

---

## MQTT Issues

### Connection Refused

**Symptoms**: `Connection refused` error

**Solutions**:
1. Check broker is running:
   ```bash
   docker ps | grep mosquitto
   # or
   systemctl status mosquitto
   ```
2. Verify port availability:
   ```bash
   netstat -an | grep 1883
   ```
3. Check firewall rules:
   ```bash
   sudo ufw allow 1883
   ```

### Authentication Failed

**Symptoms**: `Connection refused: not authorized`

**Solutions**:
1. Check username/password in configuration
2. Verify password file:
   ```bash
   mosquitto_passwd -c /path/to/passwords.txt username
   ```
3. Ensure `allow_anonymous false` is set if using auth

### WebSocket Connection Failed

**Symptoms**: Dashboard can't connect to MQTT

**Solutions**:
1. Verify WebSocket listener (port 9001) is enabled
2. Check CORS if behind proxy:
   ```nginx
   location /mqtt {
       proxy_pass http://localhost:9001;
       proxy_http_version 1.1;
       proxy_set_header Upgrade $http_upgrade;
       proxy_set_header Connection "upgrade";
   }
   ```
3. Browser console for specific errors

---

## API Issues

### 401 Unauthorized

**Symptoms**: API returns 401

**Solutions**:
1. Include API key in header:
   ```bash
   curl -H "X-API-Key: wvr_your_key" http://localhost:8080/api/v1/devices
   ```
2. Verify key is in `API_KEYS` environment variable
3. Check `REQUIRE_API_KEY=true` is set

### 429 Too Many Requests

**Symptoms**: Rate limit exceeded

**Solutions**:
1. Reduce request frequency
2. Adjust rate limits:
   ```bash
   export RATE_LIMIT_REQUESTS=200
   export RATE_LIMIT_WINDOW=60
   ```
3. Use batch endpoints when available

### 503 Service Unavailable

**Symptoms**: Service not responding

**Solutions**:
1. Check service logs:
   ```bash
   docker logs history_collector
   ```
2. Verify database file is accessible
3. Check disk space

---

## Training Issues

### CUDA Out of Memory

**Symptoms**: `CUDA out of memory` error

**Solutions**:
1. Reduce batch size:
   ```bash
   python scripts/train_crowd.py --batch_size 16
   ```
2. Use gradient accumulation
3. Enable mixed precision:
   ```python
   scaler = torch.cuda.amp.GradScaler()
   ```
4. Free GPU memory:
   ```python
   torch.cuda.empty_cache()
   ```

### Model Not Converging

**Symptoms**: Loss not decreasing, poor accuracy

**Solutions**:
1. Check learning rate (try 1e-4 to 1e-3)
2. Verify data preprocessing is correct
3. Check for data imbalance
4. Try different encoder type:
   ```bash
   python scripts/train_crowd.py --encoder_type lstm
   ```
5. Increase training data or use augmentation

### Import Errors

**Symptoms**: `ModuleNotFoundError: No module named 'wavira'`

**Solutions**:
1. Install package in development mode:
   ```bash
   pip install -e ".[dev]"
   ```
2. Verify virtual environment is activated
3. Add project root to path:
   ```python
   import sys
   sys.path.insert(0, '/path/to/wavira')
   ```

---

## Docker Issues

### Container Won't Start

**Symptoms**: Container exits immediately

**Solutions**:
1. Check logs:
   ```bash
   docker logs <container_name>
   ```
2. Verify image was built:
   ```bash
   docker images | grep wavira
   ```
3. Check resource limits (memory, CPU)

### Volume Permission Issues

**Symptoms**: Permission denied on mounted volumes

**Solutions**:
1. Fix ownership:
   ```bash
   sudo chown -R 1000:1000 ./data
   ```
2. Use named volumes instead of bind mounts
3. Run container with correct user:
   ```yaml
   user: "${UID}:${GID}"
   ```

### Network Issues Between Containers

**Symptoms**: Containers can't communicate

**Solutions**:
1. Use service names instead of `localhost`:
   ```yaml
   MQTT_HOST: mosquitto  # Not localhost
   ```
2. Verify network is created:
   ```bash
   docker network ls
   ```
3. Check container is on correct network:
   ```bash
   docker inspect <container> | grep NetworkMode
   ```

---

## Test Issues

### Tests Failing Locally but Pass in CI

**Solutions**:
1. Check Python version matches
2. Clear pytest cache:
   ```bash
   pytest --cache-clear
   ```
3. Run tests in isolation:
   ```bash
   pytest tests/test_encoder.py -v --forked
   ```

### Async Test Timeouts

**Symptoms**: `asyncio.TimeoutError` in tests

**Solutions**:
1. Increase timeout:
   ```python
   @pytest.mark.timeout(30)
   async def test_slow_operation():
       ...
   ```
2. Use proper async fixtures
3. Check for resource leaks

---

## Common Error Messages

### `RuntimeError: CUDA error: device-side assert triggered`

Usually indicates tensor shape mismatch or invalid indices. Check:
- Input tensor shapes
- Label indices within valid range
- Data preprocessing

### `FileNotFoundError: [Errno 2] No such file or directory`

Check:
- File paths are absolute or correctly relative
- Working directory is correct
- File exists and is readable

### `ConnectionRefusedError: [Errno 111] Connection refused`

Service is not running or port is wrong. Check:
- Service status
- Port number
- Firewall rules

---

## Getting Help

1. Check existing [GitHub Issues](https://github.com/your-org/wavira/issues)
2. Review logs for specific error messages
3. Create minimal reproduction case
4. Open new issue with:
   - Environment details (OS, Python version)
   - Complete error message
   - Steps to reproduce
   - Expected vs actual behavior

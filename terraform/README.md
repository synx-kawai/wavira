# Wavira Terraform Infrastructure

AWS infrastructure for Wavira CSI monitoring system.

## Prerequisites

- Terraform >= 1.0
- AWS CLI configured with `assign-test` profile
- SSH key pair (`wavira-mqtt-key`) in AWS

## Deployment

```bash
cd terraform
terraform init
terraform apply -var-file="terraform.tfvars"
```

## Post-Deployment Tasks

### ESP32 Firmware Update (CRITICAL)

**After `terraform apply` that creates/replaces the EC2 instance, you MUST update the ESP32 firmware with the new MQTT broker IP address.**

1. Get the new IP address from Terraform output:
   ```bash
   terraform output public_ip
   ```

2. Update ESP32 firmware configuration:
   ```bash
   # Edit sdkconfig.defaults with new IP
   cd ../esp-csi/examples/get-started/csi_recv
   # Update CONFIG_WAVIRA_MQTT_BROKER_URL in sdkconfig.defaults
   ```

3. Rebuild and flash each ESP32:
   ```bash
   # For each device, update device ID and flash
   source ~/esp/esp-idf/export.sh

   # Device 1 (esp32-001)
   sed -i '' 's/CONFIG_WAVIRA_DEVICE_ID=.*/CONFIG_WAVIRA_DEVICE_ID="esp32-001"/' sdkconfig
   idf.py build
   idf.py -p /dev/cu.usbmodem21101 flash

   # Device 2 (esp32-002)
   sed -i '' 's/CONFIG_WAVIRA_DEVICE_ID=.*/CONFIG_WAVIRA_DEVICE_ID="esp32-002"/' sdkconfig
   idf.py build
   idf.py -p /dev/cu.usbmodem21201 flash
   ```

4. Verify ESP32 connection:
   ```bash
   ssh -i ~/.ssh/wavira-mqtt-key.pem ec2-user@<NEW_IP> "docker logs mosquitto 2>&1 | tail -20"
   ```

## Resources Created

- EC2 instance (t2.micro) with:
  - MQTT broker (Mosquitto)
  - CSI data processor
  - REST API
  - Nginx reverse proxy
  - Cloudflare Tunnel
- S3 bucket for dashboard assets
- IAM role for EC2 to access S3
- Security group with ports: 22, 80, 443, 1883, 9001

## Dashboard Updates

Dashboard updates (without EC2 replacement) only require:

```bash
terraform apply -var-file="terraform.tfvars"
```

The dashboard file is automatically uploaded to S3 and downloaded by EC2 during initialization.

## Outputs

| Output | Description |
|--------|-------------|
| `public_ip` | EC2 public IP address |
| `mqtt_endpoint` | MQTT broker endpoint |
| `dashboard_url` | Dashboard URL |
| `ssh_command` | SSH command to connect |
| `s3_bucket` | S3 bucket name |

## Cloudflare Tunnel

The dashboard is accessible via Cloudflare Tunnel at:
- https://wavira.takezou.com/

The tunnel is configured automatically during EC2 initialization.

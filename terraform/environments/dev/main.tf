# Wavira Test Server - Development Environment
# Deploys the CSI API server to an LXC container

terraform {
  required_version = ">= 1.0.0"
}

# Local values
locals {
  app_name    = "wavira-csi-server"
  server_addr = "${var.server_user}@${var.server_host}"
}

# Install dependencies and set up the server
resource "null_resource" "server_setup" {
  triggers = {
    # Re-run when these files change
    requirements_hash = filemd5("${path.module}/../../../tools/csi_visualizer/requirements.txt")
    server_hash       = filemd5("${path.module}/../../../tools/csi_visualizer/api_server.py")
  }

  connection {
    type        = "ssh"
    host        = var.server_host
    user        = var.server_user
    timeout     = "2m"
  }

  # Install system dependencies
  provisioner "remote-exec" {
    inline = [
      "echo '=== Installing system dependencies ==='",
      "apt-get update -qq",
      "apt-get install -y -qq python3 python3-pip python3-venv git curl sqlite3 xz-utils",
      "echo '=== System dependencies installed ==='",
    ]
  }

  # Create application user and directory
  provisioner "remote-exec" {
    inline = [
      "echo '=== Creating application user and directory ==='",
      "if ! id -u ${var.app_user} >/dev/null 2>&1; then",
      "  echo 'Creating user ${var.app_user}...'",
      "  useradd -r -s /bin/false ${var.app_user}",
      "else",
      "  echo 'User ${var.app_user} already exists.'",
      "fi",
      "mkdir -p ${var.app_dir}",
      "mkdir -p ${var.app_dir}/logs",
      "echo 'Setting permissions...'",
      "chown -R ${var.app_user}:${var.app_user} ${var.app_dir}",
      "echo '=== Application user and directory setup complete ==='",
    ]
  }
}

# Copy application files
resource "null_resource" "copy_files" {
  depends_on = [null_resource.server_setup]

  triggers = {
    api_server_hash     = filemd5("${path.module}/../../../tools/csi_visualizer/api_server.py")
    device_manager_hash = filemd5("${path.module}/../../../tools/csi_visualizer/device_manager.py")
    server_multi_hash   = filemd5("${path.module}/../../../tools/csi_visualizer/server_multi.py")
    requirements_hash   = filemd5("${path.module}/../../../tools/csi_visualizer/requirements.txt")
  }

  connection {
    type    = "ssh"
    host    = var.server_host
    user    = var.server_user
    timeout = "2m"
  }

  # Copy application files
  provisioner "file" {
    source      = "${path.module}/../../../tools/csi_visualizer/"
    destination = var.app_dir
  }
}

# Install Python dependencies and configure service
resource "null_resource" "install_deps" {
  depends_on = [null_resource.copy_files]

  triggers = {
    requirements_hash = filemd5("${path.module}/../../../tools/csi_visualizer/requirements.txt")
  }

  connection {
    type    = "ssh"
    host    = var.server_host
    user    = var.server_user
    timeout = "5m"
  }

  # Create virtual environment and install dependencies
  provisioner "remote-exec" {
    inline = [
      "echo '=== Setting up Python virtual environment ==='",
      "cd ${var.app_dir}",
      "python3 -m venv venv",
      ". venv/bin/activate",
      "pip install --upgrade pip -q",
      "pip install -r requirements.txt -q",
      "chown -R ${var.app_user}:${var.app_user} ${var.app_dir}",
      "echo '=== Python dependencies installed ==='",
    ]
  }
}

# Create systemd service
resource "null_resource" "systemd_service" {
  depends_on = [null_resource.install_deps]

  triggers = {
    service_config = sha256(local.systemd_service_content)
  }

  connection {
    type    = "ssh"
    host    = var.server_host
    user    = var.server_user
    timeout = "2m"
  }

  provisioner "file" {
    content     = local.systemd_service_content
    destination = "/etc/systemd/system/${local.app_name}.service"
  }

  provisioner "remote-exec" {
    inline = [
      "systemctl daemon-reload",
      "systemctl enable ${local.app_name}",
      "systemctl restart ${local.app_name}",
      "sleep 2",
      "systemctl status ${local.app_name} --no-pager || true",
    ]
  }
}

locals {
  systemd_service_content = <<-EOF
    [Unit]
    Description=Wavira CSI API Server
    After=network.target

    [Service]
    Type=simple
    User=${var.app_user}
    WorkingDirectory=${var.app_dir}
    Environment=PATH=${var.app_dir}/venv/bin:/usr/local/bin:/usr/bin:/bin
    ExecStart=${var.app_dir}/venv/bin/python api_server.py --port ${var.server_port}
    Restart=always
    RestartSec=5

    [Install]
    WantedBy=multi-user.target
  EOF
}

# Log rotation configuration
resource "null_resource" "logrotate" {
  depends_on = [null_resource.server_setup]

  triggers = {
    logrotate_config = sha256(local.logrotate_config)
  }

  connection {
    type    = "ssh"
    host    = var.server_host
    user    = var.server_user
    timeout = "2m"
  }

  provisioner "file" {
    content     = local.logrotate_config
    destination = "/etc/logrotate.d/wavira"
  }

  provisioner "remote-exec" {
    inline = [
      "echo '=== Logrotate configured ==='",
      "logrotate -d /etc/logrotate.d/wavira 2>&1 | head -20 || true",
    ]
  }
}

locals {
  logrotate_config = <<-EOF
    ${var.app_dir}/logs/*.log /var/log/wavira*.log {
        daily
        rotate 7
        compress
        delaycompress
        missingok
        notifempty
        create 644 root root
        postrotate
            systemctl reload ${local.app_name} > /dev/null 2>&1 || true
        endscript
    }
  EOF
}

# Database backup script and cron job
resource "null_resource" "backup_setup" {
  depends_on = [null_resource.server_setup]

  triggers = {
    backup_script = sha256(local.backup_script)
  }

  connection {
    type    = "ssh"
    host    = var.server_host
    user    = var.server_user
    timeout = "2m"
  }

  # Create backup directory
  provisioner "remote-exec" {
    inline = [
      "mkdir -p ${var.backup_dir}",
      "mkdir -p ${var.backup_dir}/daily",
    ]
  }

  # Install backup script
  provisioner "file" {
    content     = local.backup_script
    destination = "${var.app_dir}/backup.sh"
  }

  # Make executable and set up cron
  provisioner "remote-exec" {
    inline = [
      "chmod +x ${var.app_dir}/backup.sh",
      # Remove existing cron entry if present
      "crontab -l 2>/dev/null | grep -v 'wavira.*backup' > /tmp/crontab.tmp || true",
      # Add new cron entry (daily at 3 AM)
      "echo '0 3 * * * ${var.app_dir}/backup.sh >> /var/log/wavira-backup.log 2>&1' >> /tmp/crontab.tmp",
      "crontab /tmp/crontab.tmp",
      "rm /tmp/crontab.tmp",
      "echo '=== Backup cron job configured (daily at 3 AM) ==='",
      "crontab -l | grep wavira",
    ]
  }
}

locals {
  backup_script = <<-EOF
    #!/bin/bash
    # Wavira Database Backup Script
    set -e

    BACKUP_DIR="${var.backup_dir}/daily"
    APP_DIR="${var.app_dir}"
    DATE=$(date +%Y%m%d_%H%M%S)
    RETENTION_DAYS=${var.backup_retention_days}

    echo "[$(date)] Starting backup..."

    # Backup SQLite databases (using xz for high compression)
    for db in history.db devices.db; do
        if [ -f "$APP_DIR/$db" ]; then
            sqlite3 "$APP_DIR/$db" ".backup '$BACKUP_DIR/$${db%.db}_$DATE.db'"
            xz -9 "$BACKUP_DIR/$${db%.db}_$DATE.db"
            echo "[$(date)] Backed up $db -> $${db%.db}_$DATE.db.xz"
        fi
    done

    # Cleanup old backups
    find "$BACKUP_DIR" -name "*.db.xz" -mtime +$RETENTION_DAYS -delete
    echo "[$(date)] Cleaned up backups older than $RETENTION_DAYS days"

    # Show backup status
    echo "[$(date)] Current backups:"
    ls -lh "$BACKUP_DIR"/*.db.xz 2>/dev/null | tail -10 || echo "No backups found"

    echo "[$(date)] Backup complete"
  EOF
}

# Monitoring setup
resource "null_resource" "monitoring" {
  depends_on = [null_resource.systemd_service]

  triggers = {
    monitor_script  = sha256(local.monitor_script)
    monitor_service = sha256(local.monitor_timer_content)
  }

  connection {
    type    = "ssh"
    host    = var.server_host
    user    = var.server_user
    timeout = "2m"
  }

  # Install monitoring script
  provisioner "file" {
    content     = local.monitor_script
    destination = "${var.app_dir}/monitor.sh"
  }

  # Install systemd timer for periodic monitoring
  provisioner "file" {
    content     = local.monitor_service_content
    destination = "/etc/systemd/system/wavira-monitor.service"
  }

  provisioner "file" {
    content     = local.monitor_timer_content
    destination = "/etc/systemd/system/wavira-monitor.timer"
  }

  provisioner "remote-exec" {
    inline = [
      "chmod +x ${var.app_dir}/monitor.sh",
      "systemctl daemon-reload",
      "systemctl enable wavira-monitor.timer",
      "systemctl start wavira-monitor.timer",
      "echo '=== Monitoring timer configured (every 5 minutes) ==='",
      "systemctl status wavira-monitor.timer --no-pager || true",
    ]
  }
}

locals {
  monitor_script = <<-EOF
    #!/bin/bash
    # Wavira Service Monitor Script

    APP_DIR="${var.app_dir}"
    SERVICE_NAME="${local.app_name}"
    LOG_FILE="/var/log/wavira-monitor.log"
    DISK_THRESHOLD=${var.disk_warning_percent}

    log() {
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
    }

    # Check service status
    if ! systemctl is-active --quiet "$SERVICE_NAME"; then
        log "WARNING: Service $SERVICE_NAME is not running. Attempting restart..."
        systemctl restart "$SERVICE_NAME"
        sleep 2
        if systemctl is-active --quiet "$SERVICE_NAME"; then
            log "INFO: Service restarted successfully"
        else
            log "ERROR: Failed to restart service"
        fi
    fi

    # Check disk space
    DISK_USAGE=$(df "$APP_DIR" | tail -1 | awk '{print $5}' | tr -d '%')
    if [ "$DISK_USAGE" -gt "$DISK_THRESHOLD" ]; then
        log "WARNING: Disk usage is $DISK_USAGE% (threshold: $DISK_THRESHOLD%)"

        # Auto-cleanup: remove old logs
        find /var/log -name "wavira*.log.*" -mtime +7 -delete 2>/dev/null || true
        log "INFO: Cleaned up old log files"
    fi

    # Check database sizes
    for db in history.db devices.db; do
        if [ -f "$APP_DIR/$db" ]; then
            SIZE=$(du -h "$APP_DIR/$db" | cut -f1)
            log "INFO: Database $db size: $SIZE"
        fi
    done

    # Health check via API
    if curl -sf "http://localhost:${var.server_port}/api/v1/health" > /dev/null; then
        log "INFO: API health check passed"
    else
        log "WARNING: API health check failed"
    fi
  EOF

  monitor_service_content = <<-EOF
    [Unit]
    Description=Wavira Service Monitor
    After=network.target ${local.app_name}.service

    [Service]
    Type=oneshot
    ExecStart=${var.app_dir}/monitor.sh
    User=root
  EOF

  monitor_timer_content = <<-EOF
    [Unit]
    Description=Run Wavira Monitor every 5 minutes

    [Timer]
    OnBootSec=1min
    OnUnitActiveSec=5min
    AccuracySec=1min

    [Install]
    WantedBy=timers.target
  EOF
}

# Health check
resource "null_resource" "health_check" {
  depends_on = [null_resource.systemd_service, null_resource.logrotate, null_resource.backup_setup, null_resource.monitoring]

  triggers = {
    always_run = timestamp()
  }

  provisioner "local-exec" {
    command = <<-EOF
      echo "Waiting for server to start..."
      sleep 3
      curl -sf http://${var.server_host}:${var.server_port}/api/v1/health && echo " ✓ Server is healthy" || echo " ✗ Server health check failed"
    EOF
  }
}

# Wavira Test Server - Development Environment Outputs

output "server_url" {
  description = "URL of the API server"
  value       = "http://${var.server_host}:${var.server_port}"
}

output "health_endpoint" {
  description = "Health check endpoint"
  value       = "http://${var.server_host}:${var.server_port}/api/v1/health"
}

output "csi_endpoint" {
  description = "CSI data endpoint"
  value       = "http://${var.server_host}:${var.server_port}/api/v1/csi"
}

output "ssh_command" {
  description = "SSH command to connect to the server"
  value       = "ssh ${var.server_user}@${var.server_host}"
}

output "logs_command" {
  description = "Command to view server logs"
  value       = "ssh ${var.server_user}@${var.server_host} journalctl -u wavira-csi-server -f"
}

output "monitor_logs_command" {
  description = "Command to view monitoring logs"
  value       = "ssh ${var.server_user}@${var.server_host} tail -f /var/log/wavira-monitor.log"
}

output "backup_dir" {
  description = "Backup directory on server"
  value       = var.backup_dir
}

output "backup_status_command" {
  description = "Command to check backup status"
  value       = "ssh ${var.server_user}@${var.server_host} ls -lah ${var.backup_dir}/daily/"
}

output "manual_backup_command" {
  description = "Command to run manual backup"
  value       = "ssh ${var.server_user}@${var.server_host} ${var.app_dir}/backup.sh"
}

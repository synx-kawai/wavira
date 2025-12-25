output "instance_id" {
  description = "EC2 instance ID"
  value       = module.wavira_server.instance_id
}

output "public_ip" {
  description = "Public IP address of the server"
  value       = module.wavira_server.public_ip
}

output "mqtt_endpoint" {
  description = "MQTT broker endpoint"
  value       = module.wavira_server.mqtt_endpoint
}

output "websocket_endpoint" {
  description = "WebSocket endpoint"
  value       = module.wavira_server.websocket_endpoint
}

output "dashboard_url" {
  description = "Dashboard URL"
  value       = module.wavira_server.dashboard_url
}

output "ssh_command" {
  description = "SSH command to connect to the server"
  value       = module.wavira_server.ssh_command
}

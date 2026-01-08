output "instance_id" {
  description = "EC2 instance ID"
  value       = aws_instance.wavira.id
}

output "public_ip" {
  description = "Public IP address"
  value       = aws_instance.wavira.public_ip
}

output "private_ip" {
  description = "Private IP address"
  value       = aws_instance.wavira.private_ip
}

output "mqtt_endpoint" {
  description = "MQTT broker endpoint"
  value       = "mqtt://${aws_instance.wavira.public_ip}:1883"
}

output "websocket_endpoint" {
  description = "WebSocket endpoint"
  value       = "ws://${aws_instance.wavira.public_ip}:8765"
}

output "dashboard_url" {
  description = "Dashboard URL"
  value       = "http://${aws_instance.wavira.public_ip}"
}

output "ssh_command" {
  description = "SSH command"
  value       = "ssh -i ~/.ssh/${var.key_name}.pem ec2-user@${aws_instance.wavira.public_ip}"
}

output "s3_bucket" {
  description = "S3 bucket for assets"
  value       = aws_s3_bucket.assets.id
}

output "security_group_id" {
  description = "Security group ID"
  value       = aws_security_group.wavira.id
}

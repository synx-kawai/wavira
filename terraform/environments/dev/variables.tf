# Wavira Test Server - Development Environment Variables

variable "server_host" {
  description = "SSH host for the server"
  type        = string
  default     = "192.168.2.197"
}

variable "server_user" {
  description = "SSH user for the server"
  type        = string
  default     = "root"
}

variable "server_port" {
  description = "API server port"
  type        = number
  default     = 8080
}

variable "app_dir" {
  description = "Application directory on server"
  type        = string
  default     = "/opt/wavira"
}

variable "app_user" {
  description = "User to run the application"
  type        = string
  default     = "wavira"
}

# Backup configuration
variable "backup_dir" {
  description = "Directory for database backups"
  type        = string
  default     = "/var/backups/wavira"
}

variable "backup_retention_days" {
  description = "Number of days to keep backup files"
  type        = number
  default     = 365
}

# Monitoring configuration
variable "disk_warning_percent" {
  description = "Disk usage percentage to trigger warning"
  type        = number
  default     = 80
}

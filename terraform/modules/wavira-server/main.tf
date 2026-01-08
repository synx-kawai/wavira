# Data sources
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

data "aws_ami" "amazon_linux_2023" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-*-x86_64"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

data "aws_vpc" "selected" {
  id = var.vpc_id != "" ? var.vpc_id : null
  default = var.vpc_id == "" ? true : null
}

data "aws_subnets" "selected" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.selected.id]
  }

  filter {
    name   = "map-public-ip-on-launch"
    values = ["true"]
  }
}

# Security Group
resource "aws_security_group" "wavira" {
  name        = "wavira-${var.environment}-sg"
  description = "Security group for Wavira CSI server"
  vpc_id      = data.aws_vpc.selected.id

  # SSH
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ssh_cidr]
    description = "SSH access"
  }

  # MQTT
  ingress {
    from_port   = 1883
    to_port     = 1883
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "MQTT broker"
  }

  # MQTT WebSocket for browser clients
  ingress {
    from_port   = 9001
    to_port     = 9001
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "MQTT WebSocket"
  }

  # HTTPS (for WSS via nginx)
  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "HTTPS/WSS"
  }

  # HTTP (for dashboard)
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "HTTP dashboard"
  }

  # Outbound
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow all outbound"
  }

  tags = merge(var.tags, {
    Name        = "wavira-${var.environment}-sg"
    Environment = var.environment
  })
}

# S3 Bucket for assets
resource "aws_s3_bucket" "assets" {
  bucket = "wavira-${var.environment}-assets-${data.aws_caller_identity.current.account_id}"

  tags = merge(var.tags, {
    Name        = "wavira-${var.environment}-assets"
    Environment = var.environment
  })
}

resource "aws_s3_bucket_public_access_block" "assets" {
  bucket = aws_s3_bucket.assets.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Upload dashboard to S3
resource "aws_s3_object" "dashboard" {
  bucket       = aws_s3_bucket.assets.id
  key          = "dashboard/index.html"
  source       = var.dashboard_path != "" ? var.dashboard_path : "${path.module}/../../../tools/csi_visualizer/dashboard_multi.html"
  content_type = "text/html"
  etag         = filemd5(var.dashboard_path != "" ? var.dashboard_path : "${path.module}/../../../tools/csi_visualizer/dashboard_multi.html")

  tags = merge(var.tags, {
    Name = "wavira-dashboard"
  })
}

# IAM Role for EC2
resource "aws_iam_role" "wavira" {
  name = "wavira-${var.environment}-ec2-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })

  tags = merge(var.tags, {
    Name        = "wavira-${var.environment}-ec2-role"
    Environment = var.environment
  })
}

resource "aws_iam_role_policy" "s3_access" {
  name = "wavira-${var.environment}-s3-access"
  role = aws_iam_role.wavira.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.assets.arn,
          "${aws_s3_bucket.assets.arn}/*"
        ]
      }
    ]
  })
}

resource "aws_iam_instance_profile" "wavira" {
  name = "wavira-${var.environment}-instance-profile"
  role = aws_iam_role.wavira.name

  tags = merge(var.tags, {
    Name        = "wavira-${var.environment}-instance-profile"
    Environment = var.environment
  })
}

# EC2 Instance
resource "aws_instance" "wavira" {
  ami                         = data.aws_ami.amazon_linux_2023.id
  instance_type               = var.instance_type
  key_name                    = var.key_name
  vpc_security_group_ids      = [aws_security_group.wavira.id]
  subnet_id                   = data.aws_subnets.selected.ids[0]
  associate_public_ip_address = true
  iam_instance_profile        = aws_iam_instance_profile.wavira.name

  user_data = base64encode(templatefile("${path.module}/user_data.sh", {
    mqtt_broker_host = var.mqtt_broker_host
    environment      = var.environment
    s3_bucket        = aws_s3_bucket.assets.id
    aws_region       = data.aws_region.current.name
  }))

  root_block_device {
    volume_size = 30
    volume_type = "gp3"
    encrypted   = false
  }

  tags = merge(var.tags, {
    Name        = "wavira-${var.environment}"
    Environment = var.environment
  })

  lifecycle {
    create_before_destroy = true
  }
}

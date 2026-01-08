terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region  = var.aws_region
  profile = var.aws_profile
}

module "wavira_server" {
  source = "./modules/wavira-server"

  environment      = var.environment
  instance_type    = var.instance_type
  key_name         = var.key_name
  allowed_ssh_cidr = var.allowed_ssh_cidr
  mqtt_broker_host = var.mqtt_broker_host
  vpc_id           = var.vpc_id

  tags = var.tags
}

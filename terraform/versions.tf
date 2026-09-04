terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.60"
    }
  }
}

provider "aws" {
  region = var.region
  default_tags {
    tags = {
      Project   = "SentinelDesk"
      ManagedBy = "terraform"
      # Everything here is disposable. Tagged so a stray resource is identifiable
      # after the fact, which matters more than usual for a stack that is meant to be
      # destroyed the same day it is created.
      Ephemeral = "true"
    }
  }
}

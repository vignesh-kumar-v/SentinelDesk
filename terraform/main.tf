# Network. The default VPC's public subnets are used deliberately rather than building
# a private-subnet layout: private subnets need a NAT gateway to pull the image, and a
# NAT gateway is ~$32/month standing charge — several times the cost of the thing it
# would be protecting. For an ephemeral demo the honest trade is a public subnet with a
# security group locked to one CIDR, which is what happens below.
data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

data "aws_region" "current" {}
data "aws_caller_identity" "current" {}

resource "aws_security_group" "alb" {
  name        = "${var.name}-alb"
  description = "Ingress to the SentinelDesk vLLM load balancer"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description = "vLLM OpenAI API, restricted to allowed_cidr"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = [var.allowed_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "task" {
  name        = "${var.name}-task"
  description = "vLLM tasks; reachable only from the load balancer"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description     = "from the ALB only"
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  egress {
    description = "model download from the Hugging Face hub, and ECR"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

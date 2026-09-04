resource "aws_ecs_cluster" "this" {
  name = var.name

  setting {
    name  = "containerInsights"
    value = "disabled" # Container Insights bills per metric; not worth it for a demo
  }
}

resource "aws_ecs_cluster_capacity_providers" "this" {
  cluster_name       = aws_ecs_cluster.this.name
  capacity_providers = ["FARGATE", "FARGATE_SPOT"]

  default_capacity_provider_strategy {
    capacity_provider = var.use_fargate_spot ? "FARGATE_SPOT" : "FARGATE"
    weight            = 1
  }
}

resource "aws_cloudwatch_log_group" "vllm" {
  name              = "/ecs/${var.name}-vllm"
  retention_in_days = 3 # logs from an ephemeral stack have a short useful life
}

# Pulls the image and writes logs. Kept separate from the task role: this one is used
# by the ECS agent before the container starts, the task role by the process inside it.
resource "aws_iam_role" "execution" {
  name = "${var.name}-ecs-execution"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "execution" {
  role       = aws_iam_role.execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# The serving process itself needs no AWS permissions at all — it loads a model and
# answers HTTP. An empty role is deliberate: it is the difference between a compromised
# inference endpoint being an inconvenience and being an account-wide incident.
resource "aws_iam_role" "task" {
  name = "${var.name}-ecs-task"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_ecs_task_definition" "vllm" {
  family                   = "${var.name}-vllm"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.task_cpu
  memory                   = var.task_memory
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn

  runtime_platform {
    operating_system_family = "LINUX"
    # The image is built for arm64 (scripts/build_vllm_docker.sh), and Fargate ARM is
    # also about 20% cheaper per vCPU-hour than X86_64.
    cpu_architecture = "ARM64"
  }

  container_definitions = jsonencode([{
    name      = "vllm"
    image     = "${aws_ecr_repository.vllm.repository_url}:${var.image_tag}"
    essential = true
    command = [
      "--model", var.model_id,
      "--served-model-name", "sentineldesk-dpo",
      "--host", "0.0.0.0",
      "--port", "8000",
      "--max-model-len", "2048",
      "--dtype", "bfloat16",
      "--enforce-eager",
    ]
    environment = [
      { name = "VLLM_CPU_KVCACHE_SPACE", value = "4" },
      { name = "HF_HOME", value = "/tmp/hf" },
    ]
    portMappings = [{ containerPort = 8000, protocol = "tcp" }]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.vllm.name
        "awslogs-region"        = data.aws_region.current.name
        "awslogs-stream-prefix" = "vllm"
      }
    }
    healthCheck = {
      command = ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"]
      # vLLM's CPU backend loads weights and warms up before it serves; a short
      # startPeriod here makes ECS kill the task mid-boot and retry forever.
      interval    = 30
      timeout     = 10
      retries     = 5
      startPeriod = 300
    }
  }])
}

resource "aws_ecs_service" "vllm" {
  name            = "${var.name}-vllm"
  cluster         = aws_ecs_cluster.this.id
  task_definition = aws_ecs_task_definition.vllm.arn
  desired_count   = var.desired_count

  capacity_provider_strategy {
    capacity_provider = var.use_fargate_spot ? "FARGATE_SPOT" : "FARGATE"
    weight            = 1
  }

  network_configuration {
    subnets = data.aws_subnets.default.ids
    # Public IP is required to pull from ECR and the HF hub without a NAT gateway.
    # The security group, not the address, is what keeps this private.
    assign_public_ip = true
    security_groups  = [aws_security_group.task.id]
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.vllm.arn
    container_name   = "vllm"
    container_port   = 8000
  }

  # The model load takes minutes; without this the ALB drains the task before it is
  # ever ready and the service never stabilises.
  health_check_grace_period_seconds = 600

  depends_on = [aws_lb_listener.http]
}

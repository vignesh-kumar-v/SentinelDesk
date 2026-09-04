# Registry for the vLLM CPU image built by scripts/build_vllm_docker.sh.
resource "aws_ecr_repository" "vllm" {
  name                 = "${var.name}-vllm-cpu"
  image_tag_mutability = "MUTABLE"
  force_delete         = true # ephemeral stack: destroy must not block on stored images

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "vllm" {
  repository = aws_ecr_repository.vllm.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep only the 3 most recent images; a 3.5GB vLLM image is not worth storing history of"
      selection    = { tagStatus = "any", countType = "imageCountMoreThan", countNumber = 3 }
      action       = { type = "expire" }
    }]
  })
}

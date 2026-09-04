output "endpoint" {
  description = "OpenAI-compatible base URL. Set SD_RESOLUTION_BASE_URL to this."
  value       = "http://${aws_lb.this.dns_name}/v1"
}

output "ecr_repository_url" {
  description = "Push the locally built vLLM image here before scaling the service up."
  value       = aws_ecr_repository.vllm.repository_url
}

output "push_commands" {
  description = "Exact commands to publish the local image to this repository."
  value = join("\n", [
    "aws ecr get-login-password --region ${data.aws_region.current.name} | docker login --username AWS --password-stdin ${data.aws_caller_identity.current.account_id}.dkr.ecr.${data.aws_region.current.name}.amazonaws.com",
    "docker tag sentineldesk/vllm-cpu:0.11.0-pinned ${aws_ecr_repository.vllm.repository_url}:${var.image_tag}",
    "docker push ${aws_ecr_repository.vllm.repository_url}:${var.image_tag}",
  ])
}

output "log_group" {
  description = "Where the vLLM server writes; the first place to look when a task will not become healthy."
  value       = aws_cloudwatch_log_group.vllm.name
}

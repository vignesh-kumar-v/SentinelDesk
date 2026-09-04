variable "region" {
  description = "AWS region. us-east-1 is the cheapest for Fargate and has the widest instance availability."
  type        = string
  default     = "us-east-1"
}

variable "name" {
  description = "Name prefix for every resource, so a stray one is traceable to this project."
  type        = string
  default     = "sentineldesk"
}

variable "image_tag" {
  description = "Tag of the vLLM image in ECR to serve."
  type        = string
  default     = "0.11.0-cpu"
}

variable "model_id" {
  description = <<-EOT
    Model the vLLM server loads. A Hugging Face id is pulled at task start; an S3 or
    EFS path would be used for the DPO checkpoint in a real deployment. Left as the
    base model by default so a `terraform apply` does not silently depend on an
    artifact that only exists on one laptop.
  EOT
  type        = string
  default     = "Qwen/Qwen2.5-0.5B-Instruct"
}

variable "task_cpu" {
  description = "Fargate vCPU units. vLLM's CPU backend is throughput-bound on cores; 4096 = 4 vCPU."
  type        = number
  default     = 4096
}

variable "task_memory" {
  description = "Fargate memory (MiB). Must be a valid pairing with task_cpu."
  type        = number
  default     = 8192
}

variable "desired_count" {
  description = "Number of serving tasks. 0 stands the stack up without paying for compute."
  type        = number
  default     = 1
}

variable "use_fargate_spot" {
  description = <<-EOT
    Run on FARGATE_SPOT (~70% cheaper, interruptible). Correct for a demo or a batch
    evaluation; wrong for anything a customer waits on, since an interruption drops
    in-flight requests.
  EOT
  type        = bool
  default     = true
}

variable "allowed_cidr" {
  description = <<-EOT
    CIDR permitted to reach the load balancer. Defaults to nothing on purpose: an
    unauthenticated LLM endpoint open to 0.0.0.0/0 is someone else's free inference,
    billed to you. Set this to your own address to reach it.
  EOT
  type        = string
  default     = "127.0.0.1/32"
}

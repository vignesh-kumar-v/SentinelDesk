# Terraform — vLLM serving on ECS Fargate

Provisions the agent-hosting layer: the same `linux/arm64` vLLM CPU image that runs
locally (`scripts/build_vllm_docker.sh`), served on Fargate behind an ALB, exposing the
OpenAI-compatible endpoint the resolution node already speaks to.

```bash
make tf-init tf-plan     # free: validate and plan
make tf-apply            # BILLABLE — see the cost note below
make tf-destroy          # tear everything down
```

## What it creates

15 resources: an ECR repository (with a 3-image lifecycle policy), an ECS cluster and
Fargate service, a task definition pinned to ARM64, two IAM roles, an ALB with a target
group and listener, two security groups, and a CloudWatch log group.

## Decisions worth knowing about

**Default VPC, public subnets.** Private subnets would need a NAT gateway to pull the
image, and a NAT gateway is roughly $32/month standing — several times the cost of the
thing it protects. For an ephemeral stack the honest trade is a public subnet with the
security group locked to one CIDR.

**`allowed_cidr` defaults to `127.0.0.1/32`, i.e. nothing.** An unauthenticated LLM
endpoint open to `0.0.0.0/0` is free inference for whoever finds it, billed to you. Set
it to your own address explicitly.

**The task role is empty.** The serving process loads a model and answers HTTP; it needs
no AWS permissions. That is the difference between a compromised inference endpoint
being an inconvenience and being an account-wide incident.

**ARM64 + Fargate Spot.** The image is built arm64 already, ARM Fargate is ~20% cheaper
per vCPU-hour, and Spot is ~70% cheaper again. Spot is right for a demo and wrong for
anything a customer waits on, since an interruption drops in-flight requests —
`use_fargate_spot = false` for that case.

**Long health-check grace periods.** vLLM's CPU backend loads weights before it answers
`/health`. With web-app defaults the ALB drains the task before it is ever ready and the
service never stabilises, which presents as a crash loop rather than as a slow start.

**`desired_count = 0` is a supported state.** It stands the stack up without paying for
compute, which is the cheapest way to prove the plumbing applies cleanly.

## Cost

Roughly, `us-east-1`, at `desired_count = 1`:

| resource | approx |
|---|---|
| ALB | ~$16/month + LCU |
| Fargate Spot, 4 vCPU / 8 GB | ~$0.04/hour |
| ECR, CloudWatch | cents at this volume |

The ALB's standing charge dominates, and it accrues whether or not a task is running.
**Run `make tf-destroy` when finished.** Every resource is tagged `Ephemeral=true` so a
stray one is identifiable later.

## Serving the DPO checkpoint rather than the base model

`model_id` defaults to the base model on purpose, so `terraform apply` does not silently
depend on an artifact that exists on one laptop. To serve the tuned checkpoint, push it
to S3 or EFS, mount it into the task, and point `model_id` at the mount. The local
`make vllm-serve` path already serves the checkpoint directly and is the faster loop for
evaluation work.

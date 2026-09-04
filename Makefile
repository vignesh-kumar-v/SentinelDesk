# SentinelDesk — one target per build phase. Every target is idempotent.
PY      := $(HOME)/Developer/venv/bin/python
VLLM_PY := .venv-vllm/bin/python
SD      := $(PY) -m sentineldesk.cli
VLLM_IMAGE := sentineldesk/vllm-cpu:0.11.0-pinned

.DEFAULT_GOAL := help
.PHONY: help setup doctor test lint fmt clean \
        smoke tickets pairs spotcheck spotcheck-human \
        sweep train curves graph-check \
        vllm-build vllm-build-native vllm-serve vllm-stop bench \
        arena arena-aa report all-core

help: ## show this help
	@grep -hE '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# ------------------------------------------------------------------ environment
setup: ## install the package into the dev venv
	$(PY) -m pip install -q -e ".[dev]"
	@echo "ready"

doctor: ## check torch, device and judge reachability
	$(SD) doctor

test: ## run the test suite
	$(PY) -m pytest tests -q

lint: ## ruff
	$(PY) -m ruff check src tests scripts

fmt: ## ruff format + autofix
	$(PY) -m ruff check --fix src tests
	$(PY) -m ruff format src tests

# ----------------------------------------------------------------- phase 0
smoke: ## PHASE 0 gate: DPO loop end-to-end on a toy batch
	$(SD) dpo-smoke

# ----------------------------------------------------------------- phase 1
tickets: ## PHASE 1a: generate synthetic tickets grounded in the policy KB
	$(SD) gen-tickets --n 600 --concurrency 10

pairs: ## PHASE 1b/c: candidates + both-order judging -> preference pairs
	$(SD) build-pairs --batch-size 32 --concurrency 8

spotcheck: ## PHASE 1 gate (automated): second frontier judge cross-check + kappa
	$(SD) spotcheck --n 40

spotcheck-human: ## PHASE 1 gate (human): write a blind worksheet to label yourself
	$(SD) spotcheck-human --n 20

# ----------------------------------------------------------------- phase 2
sweep: ## PHASE 2: sweep DPO hyperparameters, select on a validation split
	$(SD) sweep-dpo

train: ## PHASE 2: a single DPO run with fixed hyperparameters
	$(SD) train-dpo

curves: ## plot loss / rewards / margin / KL from the training history
	$(SD) plot-curves

# ----------------------------------------------------------------- phase 3
graph-check: ## PHASE 3 gate: route sample tickets through the full graph
	$(SD) graph-check --n 10

# ----------------------------------------------------------------- phase 4
vllm-build: ## build the vLLM CPU image (linux/arm64) — the path that serves
	MAX_JOBS=3 ./scripts/build_vllm_docker.sh
	docker build -f docker/Dockerfile.vllm-pins -t $(VLLM_IMAGE) .

vllm-build-native: ## build vLLM from source on macOS (compiles, then hangs — see docs)
	./scripts/build_vllm_macos.sh

vllm-serve: ## serve the DPO checkpoint via vLLM
	$(SD) serve --model artifacts/dpo/checkpoint

vllm-stop: ## stop any running vLLM server
	-docker rm -f sentineldesk-vllm 2>/dev/null
	-pkill -f "vllm.entrypoints.openai.api_server"

bench: ## PHASE 4: tokens/s and latency, vLLM vs transformers-MPS vs MLX
	$(SD) bench

# ----------------------------------------------------------------- phase 5
arena-aa: ## PHASE 5 sanity: A/A null test — same model in both arms, expect 50%
	$(SD) arena-aa --n 30

arena: ## PHASE 5: blind judge-scored win-rate on the held-out set
	$(SD) arena

report: ## regenerate reports/RESULTS.md from the phase reports
	$(SD) report

all-core: smoke tickets pairs spotcheck sweep curves graph-check bench arena-aa arena report ## the whole core scope

clean: ## remove generated data and artifacts (keeps the policy KB)
	rm -rf data/processed data/prefs artifacts reports/*.json reports/*.png

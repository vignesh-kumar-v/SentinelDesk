#!/usr/bin/env bash
# Build vLLM's CPU backend as a linux/arm64 image.
#
# Why a container at all: vLLM's native macOS CPU build compiles fine but deadlocks
# at 0% CPU in cpu_worker device init (see docs/decisions.md, F1). Running the same
# CPU backend under linux/arm64 sidesteps the macOS OpenMP runtime that hang points
# at, while still being a real vLLM server on this machine.
#
# Why the Dockerfile gets patched: vLLM's docker/Dockerfile.cpu hardcodes the build
# to all available cores. Docker Desktop here exposes 15 CPUs but only ~8 GB, and 15
# parallel C++ compiles of vLLM's kernels exhaust it - the build dies with
# "cannot allocate memory", which reads like a disk or daemon problem and is not one.
# setup.py honours MAX_JOBS, but the upstream Dockerfile declares no ARG for it, so
# the one line below adds one. The patch is applied to a copy; the vendored source is
# left untouched.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/third_party/vllm"
TAG="${TAG:-sentineldesk/vllm-cpu:0.11.0}"
MAX_JOBS="${MAX_JOBS:-4}"
DOCKERFILE="$SRC/docker/Dockerfile.cpu.patched"

[ -d "$SRC" ] || { echo "vLLM source missing; run scripts/build_vllm_macos.sh first"; exit 1; }
docker info >/dev/null 2>&1 || { echo "docker daemon is not running"; exit 1; }

# Insert MAX_JOBS into the build stage, immediately after its ARG TARGETARCH-free
# marker line, so the wheel build sees it.
python3 - "$SRC/docker/Dockerfile.cpu" "$DOCKERFILE" "$MAX_JOBS" <<'PY'
import sys
src, dst, jobs = sys.argv[1], sys.argv[2], sys.argv[3]
text = open(src).read()
needle = "FROM base AS vllm-build\n"
if needle not in text:
    sys.exit("upstream Dockerfile.cpu no longer has the 'FROM base AS vllm-build' stage; "
             "re-check the patch before trusting this build")
text = text.replace(needle, needle + f"ENV MAX_JOBS={jobs}\n", 1)
open(dst, "w").write(text)
print(f"patched Dockerfile written with MAX_JOBS={jobs}")
PY

echo "==> building $TAG for linux/arm64 (MAX_JOBS=$MAX_JOBS)"
cd "$SRC"
docker buildx build \
  --platform=linux/arm64 \
  --target vllm-openai \
  -f "$DOCKERFILE" \
  -t "$TAG" \
  --load .

echo "==> built $TAG"
docker images --filter=reference="$TAG"

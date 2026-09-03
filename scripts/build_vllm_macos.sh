#!/usr/bin/env bash
# Build vLLM's CPU backend from source on Apple silicon.
#
# vLLM publishes no macOS wheel - the PyPI wheels are Linux-only, and there is no
# Metal/MPS backend - so serving the tuned checkpoint through vLLM on this machine
# means compiling the CPU backend. vLLM documents this path and pins torch 2.8.0 for
# Darwin, which is why this uses its own virtualenv: installing that pin into the
# shared dev environment would downgrade torch underneath every other project here.
#
# Usage: scripts/build_vllm_macos.sh [vllm-tag]
set -euo pipefail

TAG="${1:-v0.11.0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/third_party/vllm"
VENV="$ROOT/.venv-vllm"
PY="${SD_PYTHON311:-$HOME/.pyenv/versions/3.11.9/bin/python3.11}"

command -v cmake >/dev/null || { echo "cmake missing: brew install cmake ninja"; exit 1; }
command -v ninja >/dev/null || { echo "ninja missing: brew install ninja"; exit 1; }
[ -x "$PY" ] || { echo "no python3.11 at $PY (set SD_PYTHON311)"; exit 1; }

CLT_VER="$(pkgutil --pkg-info=com.apple.pkg.CLTools_Executables | awk -F': ' '/version/{print $2}')"
echo "Xcode Command Line Tools: ${CLT_VER}  (vLLM requires >= 15.4)"

[ -d "$SRC" ] || git clone --depth 1 --branch "$TAG" https://github.com/vllm-project/vllm.git "$SRC"
[ -d "$VENV" ] || "$PY" -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip setuptools wheel

echo "==> installing CPU requirements (pulls torch 2.8.0 for Darwin)"
"$VENV/bin/pip" install -r "$SRC/requirements/cpu.txt"

# --no-build-isolation below means the build backend runs in THIS environment, so
# vLLM's build-time deps (setuptools-scm, ninja, cmake's python shim) have to be
# present here first. Without this the metadata step dies on a missing
# setuptools_scm, which reads like a vLLM packaging bug and is not one.
echo "==> installing build backend requirements"
"$VENV/bin/pip" install -r "$SRC/requirements/cpu-build.txt"

# cpu.txt leaves torchaudio unpinned, so pip resolves the newest release - which is
# built against a newer torch than the 2.8.0 pinned two lines above. The mismatch
# does not surface at install time: it surfaces when transformers imports torchaudio
# inside the running server, as a dlopen "Symbol not found: _torch_library_impl".
echo "==> pinning torchaudio to the torch it was built against"
"$VENV/bin/pip" install "torchaudio==2.8.0"

echo "==> compiling vLLM CPU backend (this takes a while)"
cd "$SRC"
# --no-build-isolation so the build links against the torch just installed rather
# than pulling a second copy into an isolated build env.
VLLM_TARGET_DEVICE=cpu "$VENV/bin/pip" install -e . --no-build-isolation

echo "==> verifying"
"$VENV/bin/python" -c "import vllm; print('vllm', vllm.__version__)"

#!/usr/bin/env bash
# Run the released Qwen3.8-27B NariQ model on Apple Silicon from a source checkout.
set -euo pipefail

usage() {
    echo 'Usage: bash nariq/run-mac.sh [2.0|2.5|3.0] [--model FILE] [--download-only] [-- llama-server options]'
}

die() {
    echo "run-mac.sh: $*" >&2
    exit 1
}

budget=2.0 model= download_only=0
while [ $# -gt 0 ]; do
    case "$1" in
        2.0|2.5|3.0) budget="$1" ;;
        --model)
            [ $# -ge 2 ] && [ -n "$2" ] || die '--model requires a file'
            model="$2"
            shift ;;
        --download-only) download_only=1 ;;
        -h|--help) usage; exit 0 ;;
        --) shift; break ;;
        *) usage >&2; die "unknown option: $1" ;;
    esac
    shift
done

[ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ] || die 'use a native arm64 terminal on Apple Silicon'

if [ "$download_only" = 0 ]; then
    for tool in cmake python3; do
        command -v "$tool" >/dev/null || die "$tool is required to build from source"
    done
    xcode-select -p >/dev/null 2>&1 || die 'install the Xcode command line tools with xcode-select --install'
fi

case "$budget" in
    2.0) sha=3a9e4955329dba8ba0be5509a9552220e5ccdae2f66bcc72c21c4580cd49bbb2 ;;
    2.5) sha=2445372e6de20620b940fbfab53aa718af9ba741f202da30cd2f02b5df78d05c ;;
    3.0) sha=1eaad934aa05579b8eba47ed9b26450392f9875fc2354efe8cfa8b29be8f4419 ;;
esac

repo=nari-labs/Qwen3.8-27B-NariQ-GGUF
file="Qwen3.8-27B-NariQ-${budget}bpw.gguf"
models="${TRELLIS_HOME:-$HOME/.cache/nari-llama.cpp}/models"

verify() {
    local digest
    [ -f "$1" ] || return 1
    echo "Checking SHA-256: $1" >&2
    digest=$(/usr/bin/openssl dgst -sha256 -r < "$1") || return 1
    [ "${digest%% *}" = "$sha" ]
}

if [ -n "$model" ]; then
    verify "$model" || die "$model does not match the released $budget bpw model"
else
    for candidate in "$models/$file" "$models/Qwen3.8-27B-NariQ-${budget%.0}bit.gguf"; do
        if verify "$candidate"; then
            model="$candidate"
            break
        fi
    done
    if [ -z "$model" ]; then
        model="$models/$file"
        [ ! -e "$model" ] || die "$model failed verification. Move it out of the cache or pass a verified --model file."
        command -v hf >/dev/null || die 'install the Hugging Face CLI (hf) to download the model, or pass --model FILE'
        echo "Downloading $file"
        hf download "$repo" "$file" --local-dir "$models" || die 'download failed. Check the error above and run hf auth login if the repository requires access.'
        verify "$model" || die "$model does not match the release checksum"
    fi
fi

[ "$download_only" = 0 ] || exit 0

root=$(cd "$(dirname "$0")/.." && pwd)
build="$root/build-metal"
mkdir -p "$build"
lock="$build/nariq-run.lock"
shlock -p "$$" -f "$lock" || die "another launcher is preparing $build"
trap 'rm -f "$lock"' EXIT

echo 'Preparing the pinned browser UI'
python3 "$root/nariq/prepare_ui.py"
echo 'Building llama-server with Metal'
cmake -S "$root" -B "$build" -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DGGML_CUDA=OFF
cmake --build "$build" --config Release --target llama-server -j 4

rm -f "$lock"
trap - EXIT
echo "Loading $model. Open the address printed when the server is ready. Stop with Ctrl-C."
exec "$build/bin/llama-server" -m "$model" --alias "${file%.gguf}" --device MTL0 --fit on \
    -np 1 -b 32 -ub 32 --jinja --min-p 0 --host 127.0.0.1 --port 8080 --cors-origins localhost "$@"

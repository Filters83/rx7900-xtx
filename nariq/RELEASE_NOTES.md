llama.cpp b12431 with NariQ trellis-coded weights: the runtime for the
[Qwen3.8-27B NariQ GGUFs](https://huggingface.co/nari-labs/Qwen3.8-27B-NariQ-GGUF) at 3.0, 2.5 and 2.0 bits per weight.
Stock llama.cpp, Ollama and LM Studio cannot load those files yet.

## Quick start (Linux, NVIDIA GPU)

```bash
curl -fsSLO https://github.com/nari-labs/nari-llama.cpp/releases/download/b12431-nariv0/run.sh
bash run.sh 3.0   # or 2.5, 2.0
```

`run.sh` downloads the binaries and the model once, checks every sha256, fits the context to free GPU memory (up to
262,144 tokens) and serves a chat page and an OpenAI-compatible API at http://127.0.0.1:8080. Options: `--no-vision`,
`--port N`, `--docker`, `--download-only`; arguments after `--` go to `llama-server`.

Manual setup:

```bash
mkdir nari-llama.cpp
tar -xzf nari-llama.cpp-b12431-nariv0-bin-linux-cuda-12.8-x64.tar.gz --strip-components=1 -C nari-llama.cpp
tar -xzf cudart-nari-llama.cpp-b12431-nariv0-linux-cuda-12.8-x64.tar.gz --strip-components=1 -C nari-llama.cpp
hf download nari-labs/Qwen3.8-27B-NariQ-GGUF \
    Qwen3.8-27B-NariQ-3.0bpw.gguf Qwen3.8-27B-NariQ-mmproj-BF16.gguf --local-dir .
./nari-llama.cpp/llama-server -m Qwen3.8-27B-NariQ-3.0bpw.gguf --mmproj Qwen3.8-27B-NariQ-mmproj-BF16.gguf \
    -ngl 99 -c 131072 --jinja --port 8080 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0
```

Keep `--port 8080` (this version otherwise listens on 9931). `-c 131072` fits 24 GB; use `-c 262144` on 32 GB or more.
Docker: `bash run.sh 3.0 --docker` runs `ghcr.io/nari-labs/nari-llama.cpp:b12431-nariv0`.

Models, scores and memory use are on the [model card](https://huggingface.co/nari-labs/Qwen3.8-27B-NariQ-GGUF).

## What's in it

- NariQ tensor types TRL1-TRL6, the model loader and a converter ([format](https://github.com/nari-labs/nari-llama.cpp/blob/b12431-nariv0/nariq/FORMAT.md)).
- CUDA kernels (vendored libtrellis) for compute capability 8.0 and newer, tuned for single-token decode, plus a CPU
  path for every coded operation.
- Metal, experimental and build-from-source only ([notes](https://github.com/nari-labs/nari-llama.cpp/blob/b12431-nariv0/nariq/README-metal.md)).

Decode on one RTX PRO 6000, one session, 4K / 64K input tokens: 88.9 / 70.7 (3.0bpw), 99.2 / 77.5 (2.5bpw) and
111.7 / 85.0 (2.0bpw) tokens/s. Eight sessions: 249-275 tokens/s in total.

## Platforms

| Platform | Download | Status |
| --- | --- | --- |
| Linux x86-64, NVIDIA, CUDA 12.8 | `...-bin-linux-cuda-12.8-x64.tar.gz` + `cudart-...` | Tested on RTX PRO 6000, H100, L4 and A10G |
<!-- windows -->
| Windows x86-64, NVIDIA, CUDA 12.8 | `...-bin-win-cuda-12.8-x64.zip` + `cudart-...` | **Experimental**: run on one GeForce RTX 5060 Ti, not qualified |
<!-- /windows -->
<!-- linux-only -->
| Windows x86-64 | none yet | Not part of this release yet |
<!-- /linux-only -->
| macOS, Apple Silicon | build from source | **Experimental** |

Linux needs an NVIDIA driver of 570 or newer, glibc 2.34 or newer, libstdc++ with `GLIBCXX_3.4.30` and libgomp
(Ubuntu 22.04, Debian 12); on RHEL 9 or Amazon Linux 2023 use `bash run.sh 3.0 --docker`.
<!-- windows -->
Windows needs driver 572 or newer. The Windows binaries are not code-signed, so Smart App Control blocks them when it is
on; there, `run.ps1 -Wsl` runs the Linux build in WSL2.
<!-- /windows -->
One GeForce card has been run so far (RTX 5060 Ti 16 GB, Windows 11, native and WSL2), not qualified. GPUs below compute capability 8.0, and AMD and Intel GPUs, run on the CPU.

## Known limits

- The 2.0bpw file is weaker than the 2.5 and 3.0bpw files, most on math and agent tasks.
- On Ampere, Ada, Hopper and Apple GPUs, NariQ files decode slower than comparable stock GGUFs for now.
- TRL1-TRL6 use GGUF type ids 44-49, which upstream has not assigned; NariQ files carry a `trellis.version` key so the
  fork can remap them if upstream ever takes those ids.
- The vision tower stays BF16; the MTP module is not included; LoRA on NariQ weights is rejected.

Qualification record (parity, token checks, serving tests):
[`nariq/cuda-qualification.md`](https://github.com/nari-labs/nari-llama.cpp/blob/b12431-nariv0/nariq/cuda-qualification.md).
The binaries were built from `9fc1bfa` (`30f794c` in `build-info.json`, the same tree before a commit-message rewording).
Check downloads with `sha256sum -c SHA256SUMS --ignore-missing`.

## License

llama.cpp and the NariQ additions: MIT. Vendored libtrellis: Apache-2.0. The coded layout follows exllamav3 (MIT);
credits in `NOTICE-NariQ`. The model files inherit the base model's Apache-2.0 license.

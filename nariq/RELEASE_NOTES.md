llama.cpp b12431 (`a946eb5`) with NariQ trellis-coded weights. This is the runtime for the
[Qwen3.8-27B NariQ GGUFs](https://huggingface.co/nari-labs/Qwen3.8-27B-NariQ-GGUF) at 3.0, 2.5 and 2.0 bits per weight.
Stock llama.cpp, Ollama and LM Studio cannot load those files.

The binaries of `b12431-nariv0` were built from commit `9fc1bfa` with CUDA 12.8 (`build-info.json` names it `30f794c`:
the same source tree, before the commit messages were reworded). Later commits up to the tag change only documentation,
CI and the license files.

## Quick start (Linux, NVIDIA GPU)

```bash
curl -fsSLO https://github.com/nari-labs/nari-llama.cpp/releases/download/b12431-nariv0/run.sh
bash run.sh 3.0   # or 2.5, 2.0
```

`run.sh` downloads this release's two Linux archives and the model files once, checks every file's sha256, fits the
context to free GPU memory (up to 262,144 tokens) and serves a chat page and an OpenAI-compatible API at
http://127.0.0.1:8080. Options: `--no-vision`, `--port N`, `--host ADDR`, `--docker`, `--download-only`; arguments
after `--` go to `llama-server` unchanged.

To set it up yourself:

```bash
mkdir nari-llama.cpp
tar -xzf nari-llama.cpp-b12431-nariv0-bin-linux-cuda-12.8-x64.tar.gz --strip-components=1 -C nari-llama.cpp
tar -xzf cudart-nari-llama.cpp-b12431-nariv0-linux-cuda-12.8-x64.tar.gz --strip-components=1 -C nari-llama.cpp
hf download nari-labs/Qwen3.8-27B-NariQ-GGUF \
    Qwen3.8-27B-NariQ-3.0bpw.gguf Qwen3.8-27B-NariQ-mmproj-BF16.gguf --local-dir .
./nari-llama.cpp/llama-server -m Qwen3.8-27B-NariQ-3.0bpw.gguf --mmproj Qwen3.8-27B-NariQ-mmproj-BF16.gguf \
    -ngl 99 -c 131072 --jinja --port 8080 \
    --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0
```

Pass `--port 8080` explicitly: without it, this llama.cpp version listens on port 9931. `-c 131072` fits a 24 GB GPU;
use `-c 262144` on 32 GB or more, or leave `-c` out to fit the context to free memory.

Docker (needs the NVIDIA Container Toolkit): `bash run.sh 3.0 --docker` runs
`ghcr.io/nari-labs/nari-llama.cpp:b12431-nariv0`, whose entry point is `llama-server`.

## Models

| File | Bits per weight | Size |
| --- | ---: | ---: |
| [`Qwen3.8-27B-NariQ-3.0bpw.gguf`](https://huggingface.co/nari-labs/Qwen3.8-27B-NariQ-GGUF/blob/main/Qwen3.8-27B-NariQ-3.0bpw.gguf) | 3.00 | 10.12 GB |
| [`Qwen3.8-27B-NariQ-2.5bpw.gguf`](https://huggingface.co/nari-labs/Qwen3.8-27B-NariQ-GGUF/blob/main/Qwen3.8-27B-NariQ-2.5bpw.gguf) | 2.50 | 8.44 GB |
| [`Qwen3.8-27B-NariQ-2.0bpw.gguf`](https://huggingface.co/nari-labs/Qwen3.8-27B-NariQ-GGUF/blob/main/Qwen3.8-27B-NariQ-2.0bpw.gguf) | 2.00 | 6.75 GB |
| [`Qwen3.8-27B-NariQ-mmproj-BF16.gguf`](https://huggingface.co/nari-labs/Qwen3.8-27B-NariQ-GGUF/blob/main/Qwen3.8-27B-NariQ-mmproj-BF16.gguf) (vision projector) | 16 | 0.93 GB |

Scores, sampling settings and memory use are on the
[model card](https://huggingface.co/nari-labs/Qwen3.8-27B-NariQ-GGUF). `run.sh` and `run.ps1` pin each file's sha256.

## What's in this release

- **NariQ tensor types TRL1-TRL6** (1-6-bit trellis codes with scales, permutations and Hadamard transforms), the
  model loader and a converter for existing NariQ checkpoints. Format: [`nariq/FORMAT.md`](https://github.com/nari-labs/nari-llama.cpp/blob/b12431-nariv0/nariq/FORMAT.md).
- **CUDA kernels** from the vendored libtrellis for NVIDIA GPUs of compute capability 8.0 or newer. Coded embedding
  lookups run on the CPU.
- **Single-token decode tuning (Nov 18):** fused rotations and gathers around the coded GEMV, an ADD + SOFTPLUS + MUL
  fusion for the delta-net gate, in-place recurrent and convolution state for single-sequence decode, and early launch
  of the kernel after a coded linear on SM90 and newer. On the RTX PRO 6000 this build decodes 4-6% faster than the
  previous build, with identical greedy output tokens.
- **Hopper tuning:** flash attention reads the BF16 KV cache directly for single-token decode on SM90, and the archive
  carries sm_90 code, so an H100 no longer compiles PTX on its first start. On an H100, a development build with
  these changes raised 3.0bpw decode from 69.9 to 74.0 tokens/s at 4K context, 65.6 to 71.7 at 16K and 53.5 to 65.5
  at 64K, with identical greedy tokens (same flags as the table below).
- **CPU path:** every coded operation also runs on the CPU. The Linux archive carries all x86-64 CPU variants and
  picks the best one at start.
- **Metal (experimental, build from source):** native coded products on Apple GPUs; see
  [`nariq/README-metal.md`](https://github.com/nari-labs/nari-llama.cpp/blob/b12431-nariv0/nariq/README-metal.md).
- The llama-server chat page, pinned by sha256.

## Platforms

| Platform | Download | Status |
| --- | --- | --- |
| Linux x86-64, NVIDIA, CUDA 12.8 | `nari-llama.cpp-b12431-nariv0-bin-linux-cuda-12.8-x64.tar.gz` + `cudart-...-linux-cuda-12.8-x64.tar.gz` | Tested with this build on RTX PRO 6000, H100, L4 and A10, all three files, through `run.sh` and the manual command. B200, A100 and L40S ran an earlier build of this release line. |
<!-- windows -->
| Windows x86-64, NVIDIA, CUDA 12.8 | `nari-llama.cpp-b12431-nariv0-bin-win-cuda-12.8-x64.zip` + `cudart-...-win-cuda-12.8-x64.zip` | **Experimental.** Built and CPU-checked on GitHub's Windows runner; never run on a Windows GPU. |
<!-- /windows -->
<!-- linux-only -->
| Windows x86-64 | none yet | Windows archives are not part of this release yet. |
<!-- /linux-only -->
| macOS, Apple Silicon | none (build from source) | **Experimental.** Only an M3 Pro (18 GB) has run it: about 3.8, 4.0 and 4.9 tokens/s decode for the 3.0, 2.5 and 2.0bpw files. |

- No GeForce card has been tested yet. The tested GPUs are datacenter and workstation cards.
- The Linux archive has native code for compute capability 8.6, 8.9, 9.0 and 12.0 (RTX 30xx, RTX 40xx / L4 / L40S,
  H100 / H200, RTX 50xx / RTX PRO 6000). Other cards of compute capability 8.0 or newer (A100, B200) compile PTX on
  the first start, which takes about 50 s.
- GPUs older than compute capability 8.0 fall back to the CPU, which is slow for a 27B model. AMD and Intel GPUs
  (HIP, Vulkan, SYCL) have no NariQ kernels.

## Requirements

- Linux: an NVIDIA driver of 570 or newer, glibc 2.34 or newer, libstdc++ with `GLIBCXX_3.4.30`, and libgomp
  (updated Ubuntu 22.04 or Debian 12 meet these). `run.sh` also needs `curl`, `tar` and `sha256sum`.
<!-- windows -->
- Windows: Windows 10 1803 or newer and an NVIDIA driver of 572 or newer.
<!-- /windows -->
- The tests used drivers 610 and 620; the minimum driver versions themselves were not tested.

## Decode speed

One NVIDIA RTX PRO 6000 Blackwell Server Edition (driver 610.57.04), this release's Linux archive, one session, 512
greedy output tokens, tokens/s (median of three):

| Input tokens | 3.0bpw | 2.5bpw | 2.0bpw |
| ---: | ---: | ---: | ---: |
| 4,096 | 88.9 | 99.2 | 111.7 |
| 16,384 | 84.5 | 93.8 | 105.1 |
| 65,536 | 70.7 | 77.5 | 85.0 |

Flags: `llama-server -np 1 -ngl all -fa on -ctk bf16 -ctv bf16 -t 1 -tb 16 --cache-ram 0 --backend-sampling
--samplers temperature`, with `-c` 512 above the input length (greedy sampling on the GPU, BF16 KV cache). With the
manual command above (`-ngl 99 -c 131072 --jinja`, llama-server's other defaults), 3.0bpw decodes at 59.8 tokens/s
at 4,096 input tokens. With eight concurrent sessions (`-np 8 -c 262144`), total decode was 254, 264 and 275 tokens/s
for the 3.0, 2.5 and 2.0bpw files.

## Known limits

- The GGUFs need this fork. Stock llama.cpp, Ollama and LM Studio cannot load the NariQ tensor types.
- TRL1-TRL6 use GGUF type ids 44-49, which upstream llama.cpp has not assigned. A future upstream version that
  assigns them may misread these files instead of refusing them. NariQ files carry a required `trellis.version` key;
  if upstream takes these ids, the fork will map NariQ files by that key.
- The 2.0bpw file is weaker than the 2.5 and 3.0bpw files, most on math and agent tasks (model card).
- The vision tower stays in BF16; only the language model is quantized. The MTP module is not included.
- Kernel tuning has focused on SM120 (RTX PRO 6000), with the Hopper changes above. On Ampere, Ada, Hopper and Apple
  GPUs, NariQ files are slower than comparable stock GGUFs for now.
- Coded embedding lookups run on the CPU, also with a GPU.
- LoRA adapters on NariQ weights are rejected.
<!-- windows -->
- Windows archives are experimental: never run on a Windows GPU.
<!-- /windows -->
- macOS is experimental and build-from-source only. GeForce cards are untested.
- Same-weight parity: against a dense F32 decode of the same coded weights, these binaries pass mean KL <= 0.001
  and top-1 agreement >= 98% for all three files. A BF16 reference exceeds the KL limit
  because of BF16 rounding; both results are in
  [`nariq/cuda-qualification.md`](https://github.com/nari-labs/nari-llama.cpp/blob/b12431-nariv0/nariq/cuda-qualification.md).

## Checksums

`SHA256SUMS` lists the sha256 of every file attached to this release. After downloading, run
`sha256sum -c SHA256SUMS --ignore-missing`. `run.sh` reads the same file and checks each archive before extracting it.

## License

The llama.cpp code and the NariQ additions are under the MIT license (`LICENSE`). The vendored libtrellis is under
Apache-2.0 (`LICENSE-libtrellis` in the archives). The coded weight layout follows exllamav3 (MIT); credits are in
`nariq/NOTICE` (`NOTICE-NariQ`) and `licenses/LICENSE-exllamav3` (`LICENSE-exllamav3`). The model files inherit the
base model's Apache-2.0 license.

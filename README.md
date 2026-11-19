# nari-llama.cpp

The Nari Labs fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) that runs the NariQ trellis-coded
[Qwen3.8-27B GGUFs](https://huggingface.co/nari-labs/Qwen3.8-27B-NariQ-GGUF) (3.0, 2.5 and 2.0 bits per weight). It adds
the TRL1-TRL6 tensor types, the model loader, checkpoint conversion, CPU operations, CUDA kernels (vendored libtrellis)
and experimental Metal kernels. Stock llama.cpp, Ollama and LM Studio cannot load these files.

This branch is based on upstream b12431 (`a946eb5`). The current release is
[`b12431-nariv0`](https://github.com/nari-labs/nari-llama.cpp/releases/tag/b12431-nariv0).

## Quick start (Linux x86-64, NVIDIA GPU)

```sh
curl -fsSLO https://github.com/nari-labs/nari-llama.cpp/releases/download/b12431-nariv0/run.sh
bash run.sh 3.0   # or 2.5, 2.0
```

`run.sh` downloads the release's binaries and the model files, checks their sha256, fits the context to free GPU memory
and serves a chat page and an OpenAI-compatible API at http://127.0.0.1:8080. `bash run.sh --help` lists its options.
It needs an NVIDIA driver of 570 or newer, glibc 2.34 or newer and libstdc++ with `GLIBCXX_3.4.30` (updated Ubuntu
22.04 or Debian 12).

## Prebuilt binaries

The [release page](https://github.com/nari-labs/nari-llama.cpp/releases/tag/b12431-nariv0) has the Linux x86-64 CUDA 12.8
archives (programs and CUDA runtime), `run.sh` and a `SHA256SUMS` over all assets. Extract both archives into one
folder, then:

```sh
./llama-server -m Qwen3.8-27B-NariQ-3.0bpw.gguf --mmproj Qwen3.8-27B-NariQ-mmproj-BF16.gguf \
    -ngl 99 -c 131072 --jinja --port 8080 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0
```

Pass `--port 8080` explicitly: this llama.cpp version's default port is 9931. The release also has Windows x86-64 CUDA
archives and `run.ps1`. They are experimental: built and checked on the CPU, never run on a Windows GPU.

## Docker

```sh
docker run --rm --gpus all -p 127.0.0.1:8080:8080 -v "$PWD:/models:ro" ghcr.io/nari-labs/nari-llama.cpp:b12431-nariv0 \
    -m /models/Qwen3.8-27B-NariQ-3.0bpw.gguf --mmproj /models/Qwen3.8-27B-NariQ-mmproj-BF16.gguf \
    --fit-target 2560 --host 0.0.0.0 --port 8080
```

This needs the NVIDIA Container Toolkit. `bash run.sh 3.0 --docker` does the same with the files it downloads.

## Supported hardware

| Hardware | Status |
| --- | --- |
| NVIDIA, compute capability 8.0 or newer, Linux | Tested on RTX PRO 6000, H100, L4 and A10 with all three files; B200, A100 and L40S ran an earlier build. Native code for 8.6, 8.9, 9.0 and 12.0; other cards compile PTX on the first start (about 50 s). |
| NVIDIA GeForce | Not tested yet. |
| NVIDIA, Windows | Experimental; GPU execution untested. |
| Apple Silicon (Metal) | Experimental, build from source; only an M3 Pro has run it, at about 4-5 tokens/s. See [README-metal.md](nariq/README-metal.md). |
| CPU | Every coded operation runs on the CPU, slowly for a 27B model; GPUs older than compute capability 8.0 fall back to it. x86-64 is packaged; ARM64 has been checked on an M3 Pro only. |
| AMD, Intel GPUs | No NariQ kernels (HIP, Vulkan, SYCL). |

Coded embedding lookups run on the CPU, also with a GPU. LoRA adapters on NariQ weights are rejected. Kernel tuning has
focused on SM120 (RTX PRO 6000); on Ampere, Ada, Hopper and Apple GPUs NariQ files are slower than comparable stock
GGUFs for now.

## Build from source

Linux with the NVIDIA CUDA toolkit:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build build -j 4
./build/bin/llama-server -m /path/to/model.gguf --mmproj /path/to/projector.gguf --jinja --port 8080
```

CPU only:

```sh
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_METAL=OFF
cmake --build build-cpu -j 4
./build-cpu/bin/llama-server -m /path/to/model.gguf -ngl 0 --port 8080
```

Apple Silicon with Metal: see [README-metal.md](nariq/README-metal.md). Release packaging, the Docker image, conversion
and the launchers: [nariq/README.md](nariq/README.md).

## Guides

- [Builds, packages, conversion and launchers](nariq/README.md)
- [NariQ tensor format](nariq/FORMAT.md)
- [CUDA qualification record](nariq/cuda-qualification.md)
- [Metal (experimental)](nariq/README-metal.md)

## Upstream and licensing

General llama.cpp documentation is in [docs](docs/) and the [upstream project](https://github.com/ggml-org/llama.cpp).
Upstream installers and distributions do not include the NariQ tensor types, and upstream llama.cpp does not endorse
this fork.

The NariQ additions are under the same [MIT license](LICENSE) as llama.cpp. The vendored libtrellis keeps its own
[Apache-2.0 license](ggml/src/ggml-cuda/libtrellis/LICENSE). The coded weight layout follows
[exllamav3](https://github.com/turboderp-org/exllamav3) (MIT); see [NOTICE](nariq/NOTICE).

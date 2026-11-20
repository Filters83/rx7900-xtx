# nari-llama.cpp: Windows x86-64, CUDA 12.8

Extract both the program ZIP and the matching CUDA runtime ZIP into one directory. The package requires Windows 10 1803 or newer and NVIDIA driver 572 or newer for CUDA execution. A GPU with compute capability 8.0 or newer is required for NariQ CUDA kernels.

```powershell
.\llama-server.exe -m Qwen3.8-27B-NariQ-3.0bpw.gguf --mmproj Qwen3.8-27B-NariQ-mmproj-BF16.gguf --port 8080
```

The server opens a chat page and an OpenAI-compatible API at http://127.0.0.1:8080 (without `--port`, llama-server's default is 9931). Without `-c`, context is fitted to available memory. Without a working GPU, the CPU backend runs slowly. `run.ps1` on the release page automates the downloads, checksums and setup: `powershell -ExecutionPolicy Bypass -File run.ps1 -Budget 3.0`.

Native compilation and CPU checks do not establish Windows GPU qualification. See the release notes for the tested hardware and known limitations.

`build-info.json` records the source revision, hashes and build settings. llama.cpp and the NariQ additions are MIT (`LICENSE`); libtrellis is Apache-2.0 (`LICENSE-libtrellis`); credits, including exllamav3, are in `NOTICE-NariQ` and `LICENSE-exllamav3`.

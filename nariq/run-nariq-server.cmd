@echo off
rem NariQ 3.0 bpw on Vulkan with the DFlash2 draft, with the settings tested on an RX 9070 16 GB (112K context, q4_0
rem KV cache). The release zip has this script next to llama-server.exe. Put the GGUF files in a "models" folder next
rem to it, or set MODEL, DRAFT, MMPROJ (vision projector) or CTX before running it. The draft and the projector are
rem optional: without them the server runs without speculative decoding or without vision. Extra arguments go to
rem llama-server (for example --port 8081). Chat page and OpenAI-compatible API: http://127.0.0.1:8080
setlocal
cd /d "%~dp0"
if not defined MODEL set "MODEL=models\Qwen3.8-27B-NariQ-3.0bpw.gguf"
if not defined DRAFT set "DRAFT=models\Qwen3.8-27B-DFlash2-Q4_K_M.gguf"
if not defined MMPROJ set "MMPROJ=models\Qwen3.8-27B-NariQ-mmproj-BF16.gguf"
if not defined CTX set "CTX=114688"
if not exist "%MODEL%" (
    echo Model not found: %MODEL%
    echo Download it from https://huggingface.co/nari-labs/Qwen3.8-27B-NariQ-GGUF or set MODEL to its path.
    exit /b 1
)
set "VISION="
if exist "%MMPROJ%" set VISION=--mmproj "%MMPROJ%" --no-mmproj-offload
set "SPEC="
if exist "%DRAFT%" set SPEC=--spec-type draft-dflash --spec-draft-model "%DRAFT%" --spec-draft-device Vulkan0 --spec-draft-ngl 99 --spec-draft-type-k q4_0 --spec-draft-type-v q4_0 --spec-draft-n-max 6 --spec-draft-temp 0.8 --spec-accept rejection -bs --spec-draft-vocab 65536
llama-server.exe --model "%MODEL%" %VISION% --device Vulkan0 --n-gpu-layers 99 --ctx-size %CTX% --parallel 1 ^
    --flash-attn on --cache-type-k q4_0 --cache-type-v q4_0 --batch-size 512 --ubatch-size 512 ^
    --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --repeat-penalty 1 --jinja --reasoning-effort medium ^
    --cache-ram 4096 --ctx-checkpoints 12 --fit off %SPEC% --host 127.0.0.1 --port 8080 %*

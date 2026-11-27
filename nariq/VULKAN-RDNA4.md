# NariQ on Vulkan: Windows, AMD driver and RDNA4

This branch (`nariq-vulkan-rdna4`) runs Nari Labs' NariQ models (Trellis-coded Qwen3.8-27B, 3.0 bpw) on the Vulkan
backend, on Windows with the AMD proprietary driver. It is tested on a single machine with an RX 9070; all numbers
below come from there. The full experiment log, with measurements and rejected attempts, is in
[`NARIQ_PIANO.md`](../NARIQ_PIANO.md) (Italian).

## Test system

| | |
|---|---|
| GPU | AMD Radeon RX 9070, 16 GB (RDNA4) |
| Driver | AMD proprietary, Adrenalin 26.11.2 (Vulkan, LLPC) |
| Host | Ryzen 7 5800X, 32 GB RAM, Windows 11 |
| Compiler | MSVC 19.51 (Visual Studio 2026), Vulkan SDK 1.4.363 |

## Build

```
scripts\build-vulkan-windows.cmd
```

Requires Visual Studio with the C++ toolchain and the Vulkan SDK (see [`docs/rdna3/WINDOWS.md`](../docs/rdna3/WINDOWS.md)).
The script configures with Ninja, `GGML_VULKAN=ON`, `GGML_NATIVE=OFF` and AVX2/FMA, and builds the server, benchmarks
and tests.

## Models

- Model: `Qwen3.8-27B-NariQ-3.0bpw.gguf` and vision projector `Qwen3.8-27B-NariQ-mmproj-BF16.gguf`
  ([nari-labs/Qwen3.8-27B-NariQ-GGUF](https://huggingface.co/nari-labs/Qwen3.8-27B-NariQ-GGUF)).
- Draft for speculative decoding: `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` (DFlash2). With a NariQ target the draft shares
  the target's embedding and Trellis output head.

The same binary also runs regular GGUF models (e.g. Qwen3.8-27B Unsloth UD-Q2_K_XL).

## Recommended launch

Daily settings with 16 GB of VRAM (112K context, q4_0 KV cache, vision projector on the CPU):

```
llama-server --model Qwen3.8-27B-NariQ-3.0bpw.gguf --mmproj Qwen3.8-27B-NariQ-mmproj-BF16.gguf --no-mmproj-offload ^
  --device Vulkan0 --n-gpu-layers 99 --ctx-size 114688 --parallel 1 --flash-attn on ^
  --cache-type-k q4_0 --cache-type-v q4_0 --batch-size 512 --ubatch-size 512 ^
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --repeat-penalty 1 --jinja --reasoning-effort medium ^
  --cache-ram 4096 --ctx-checkpoints 12 --fit off ^
  --spec-type draft-dflash --spec-draft-model Qwen3.8-27B-DFlash2-Q4_K_M.gguf --spec-draft-device Vulkan0 ^
  --spec-draft-ngl 99 --spec-draft-type-k q4_0 --spec-draft-type-v q4_0 ^
  --spec-draft-n-max 6 --spec-draft-temp 0.8 --spec-accept rejection -bs --spec-draft-vocab 65536
```

Notes:

- **Draft of 6 tokens, draft temperature 0.8.** With the verify optimizations, 6 tokens beat 3-5; 7-8 give similar
  but less stable results and use more VRAM.
- **`--spec-accept rejection`**: rejection-sampling verification, same distribution as the target alone, more
  accepted tokens.
- **VRAM at the limit.** The server uses ~14.5 GB dedicated. With other programs using the GPU (e.g. a browser with
  heavy tabs) the driver moves part of the server into shared memory and everything slows down: as a check, the
  shared memory of the `llama-server` process should stay around 100-150 MB.

## Performance

RX 9070, KV cache q4_0, flash attention, batch and micro-batch 512, build `8d94a9c7f`. Unsloth = Qwen3.8-27B
UD-Q2_K_XL on the same binary.

**llama-bench** (no draft):

| | NariQ 3.0 bpw | Unsloth UD-Q2_K_XL |
|---|---|---|
| Prefill pp512 | 1213 t/s | 1160 t/s |
| Prefill pp512 at 32K context | 822 t/s | 796 t/s |
| Generation tg64 | 30.1 t/s | 39.4 t/s |
| 7-token verify (pp7, draft of 6) | 169 t/s | |

**llama-server with the DFlash2 draft** (NariQ: 6 tokens at 0.8; Unsloth: 3 tokens at 1.0), 112K context:

| | NariQ | Unsloth |
|---|---|---|
| Generation, chat | 68-73 t/s | 107-109 t/s |
| Generation, 32K prompt | 66 t/s | 106 t/s |
| Prefill, 32K prompt | 896 t/s (35.9 s) | 898 t/s (35.8 s) |
| Short prompt (153 tokens) | ~0.65 s | ~1.2 s |

NariQ generation is compute-bound, not memory-bound: every weight is rebuilt from its 16-bit state for each token.
That is why the draft matters so much: verifying 7 tokens costs only ~1.25× a single token.

## What the branch contains

- **Trellis ops on Vulkan** (`TRELLIS`, `TRELLIS_GET_ROWS`) with dedicated products:
  - 1 token: int8 product with `dot4`;
  - 2-16 tokens (draft verification): int8 products on cooperative matrices, int32 accumulation over 32-input blocks
    and prefetch of the next tile;
  - prompt: FP16 product, 128 outputs × 256 tokens per workgroup.
- **State hash with 24-bit multiplies.** The AMD compiler fused the two halves back into a quarter-rate 32-bit
  multiply; the constants now come from a value the compiler cannot see.
- **Prefill:** fused MLP chain (gate and up output rotations, SwiGLU and down input rotation in one kernel), and
  128-block rotations done in registers, fused with the input and output stages.
- **Two 256-token chunks per prompt-product launch.** The 128 × 256 product fits one workgroup per CU (180 VGPRs),
  so 136 workgroups for gate and up left the last of 3 rounds under half full; with a 512-token micro-batch both
  chunks now share one launch (5 rounds instead of 6, no extra memory).
- **DFlash2 draft:** shares embedding and Trellis head with the target; the prompt features go to the draft directly
  on the GPU, even when the draft runs in smaller chunks than the target batch.
- **Faster recurrent-state checkpoints** (Qwen3.8 delta-net) with the draft: rebuilding the replay registers uses
  AVX2 + FMA under MSVC too, and is no longer done twice (675 → 90 ms per checkpoint).

## Environment variables

All optimizations are on by default; `=0` turns one off, for comparisons or to isolate a problem.

| Variable | What it disables |
|---|---|
| `GGML_VK_TRELLIS_Q8` | int8 products for 1-16 tokens (falls back to the float product) |
| `GGML_VK_TRELLIS_Q8_PACKED` | packed int8 activations for verification |
| `GGML_VK_TRELLIS_VERIFY_CM` | verification on cooperative matrices |
| `GGML_VK_TRELLIS_VERIFY_I32` | int32 accumulation over 32-blocks in verification |
| `GGML_VK_TRELLIS_COOPMAT` | cooperative-matrix products |
| `GGML_VK_TRELLIS_MM`, `GGML_VK_TRELLIS_MM_WIDE`, `GGML_VK_TRELLIS_MM_BIG` | prefill products |
| `GGML_VK_TRELLIS_BLOCK_SHUFFLE` | block rotations with subgroup shuffles |
| `GGML_VK_TRELLIS_FUSE` | fused input/output stages up to 15 tokens |
| `GGML_VK_TRELLIS_CHAIN` | fused MLP chain in prefill |
| `GGML_VK_TRELLIS_PREP_IN`, `GGML_VK_TRELLIS_OUT128` | 128-block rotations in registers in prefill |
| `GGML_VK_TRELLIS_MM_PAIR` | two 256-token chunks per prompt-product launch |
| `LLAMA_DFLASH_DEV_INJECT` | direct GPU hand-off of features to the draft |

Diagnostics: `GGML_VK_TRELLIS_PROFILE=1` (with `GGML_VK_PERF_LOGGER=1`) prints the GPU time of each Trellis stage;
`GGML_VK_TRELLIS_DEBUG=1` prints the rotation shapes.

## Tests

`test-backend-ops -o TRELLIS` (380/380), `ctest -R trellis` (4/4), logits compared with the CPU
(`test-trellis-model` and `nariq/compare_logits.py`, max KL < 1e-3, same top token) and KL on WikiText-2 with
`llama-perplexity --kl-divergence`.

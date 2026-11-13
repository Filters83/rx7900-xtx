# Windows and the AMD proprietary driver (RDNA4 report)

The fork was developed on Linux with Mesa RADV. This page covers what it takes to build and run it on Windows with the
AMD proprietary Vulkan driver, and what was measured on one RDNA4 card. Everything here comes from a single machine.

## Test system

| | |
|---|---|
| GPU | AMD Radeon RX 9070, 16 GB (RDNA4, detected by this fork as RDNA3) |
| Driver | AMD proprietary, Adrenalin 26.10.1 (Vulkan driver 2.0.416, LLPC) |
| Host | Ryzen 7 5800X, 32 GB RAM, Windows 11 |
| Compiler | MSVC 19.51 (Visual Studio 2026), Vulkan SDK 1.4.363 |

## Build

```
scripts\build-vulkan-windows.cmd
```

It needs Visual Studio with the C++ toolchain and the Vulkan SDK. Two portability fixes were needed for MSVC
(`clock_gettime` in the speculative trace and `__builtin_ctz` in the graph-replay key).

## What had to change for the proprietary driver

- **Grouped-query flash attention shader (`flash_attn_cm1_gqa.comp`).** The driver's shader compiler crashed with an
  access violation in `amdvlk64.dll` during `createComputePipeline`, so the server died at the first decode. Bisecting the
  shader showed that a store to an element of an array of cooperative matrices inside the KV loop is enough to trigger it.
  The S, P and O matrices are now separate variables. `FLASH_ATTN_EXT` then passes 3622 cases on this card.
- **Large matmul tile.** Port of upstream llama.cpp #31587: the large coopmat tile is enabled on the proprietary driver
  from 2.0.416 on. `GGML_VK_AMD_LARGE_TILE=0` keeps the old gate.
- **Split-K.** Ports of upstream llama.cpp #30280 and #30146. `GGML_VK_SPLIT_K_LARGE=0` and `GGML_VK_SPLIT_K_SMALLN=0`
  switch them off. No effect of #30146 on MTP decode was measured here, in either direction.
- **q4_0 KV cache.** The grouped-query shader had a fast load path for q8_0 K/V only; q4_0 went through the generic
  per-element decode and was slower than not using the shader at all (349 vs 502 tokens/s prefill at 32K). q4_0 now
  takes the same octet loads as q8_0 (18-byte blocks, low and high nibbles): 615 tokens/s.
- **bf16 K/V.** The driver also crashes compiling the coopmat1 flash attention shader with bf16 K/V; that case falls
  back to the scalar shader on the proprietary driver.

`GGML_VK_AMD_PROP_XL=1` opts in to the fork's RADV-only matmul work (128x256 tile, double-buffered tiles, f32
accumulators) on the proprietary driver. No effect was measured with `-ub 256`.

`GGML_VK_TRACE_PIPELINES=1` logs every pipeline before and after the driver compiles it, which is how the crashing
shader was found.

## Known problem

`test-backend-ops -o FLASH_ATTN_EXT` still stops with a driver crash while compiling the non-GQA coopmat1 shader
(`flash_attn_f32_f16_cm1`) for head size 128. It was not hit by Qwen3.8-27B in any run. Not fixed.

## Results

Qwen3.8-27B Unsloth UD-Q2_K_XL (mostly IQ2 / IQ3 tensors), `-c 114688`, q4_0 KV cache, `-b 512 -ub 256`, flash attention,
vision projector on the CPU, temperature 1.0. Synthetic prompt, one pass per row, 256-512 generated tokens with
`ignore_eos`. Upstream is llama.cpp b12233 (`5f490d7`, 2026-11-09), official Windows Vulkan build.

| Decode, tokens/s | short | 32K | 65K |
|---|---:|---:|---:|
| upstream, no draft | 35.3 | 32.5 | 30.2 |
| upstream + MTP 1 | 50.1 | 45.3 | 41.8 |
| this fork + MTP 1 | 68.4 | 62.9 | 57.6 |
| this fork + MTP 3, `--spec-accept rejection -bs --spec-draft-vocab 65536` | 79.2 | 67.9 | 53.1 |
| this fork + DFlash2 4, same flags | 74.1 | 66.0 | 47.0 |

| Prefill, tokens/s | 0-32K | 32K-65K |
|---|---:|---:|
| upstream, no draft | 755 | 482 |
| upstream + MTP 1 | 704 | 456 |
| this fork + MTP 1 | 554 | 300 |
| this fork + MTP 3 | 551 | 292 |
| this fork + DFlash2 4 | 603 | 344 |

Decode is 27-58% faster than upstream with MTP. Prefill is still 14-36% slower, and the gap grows with depth, so it is
in the attention path rather than the matmuls: the fixed cost per token is the same in both builds. Peak VRAM is the
same as upstream with MTP (13.3 GiB against 13.5) and 0.6 GiB higher with the DFlash2 draft.

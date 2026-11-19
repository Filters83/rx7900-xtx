# Metal (experimental)

This fork runs NariQ coded linear layers (`GGML_OP_TRELLIS`) on Apple GPUs. TRL1-TRL6 weights stay compressed: the GPU applies the input gathers and scales, block or whole-slice Hadamard transforms, the coded products, the inverse output transforms and the output gathers, for token decode and batched prefill, including banded output heads. Coded embedding lookups (`GGML_OP_TRELLIS_GET_ROWS`) run on the CPU backend; the scheduler moves activations between the two.

There is no macOS package or launcher. Build from source.

## Build and run

Apple Silicon, macOS with the Xcode command line tools. The coded products need Apple GPU family 7 or newer; other devices and unsupported layouts fall back to the CPU.

```sh
cmake -S . -B build-metal -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=ON -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=ON
cmake --build build-metal -j 4
./build-metal/bin/llama-server -m /path/to/model.gguf -ngl 99 -c 8192 --jinja --port 8080
```

Check that unified memory holds the model before loading it; use fewer GPU layers (`-ngl`) if it does not. Add `--mmproj /path/to/projector.gguf` for vision. The default build embeds the shader source and compiles it at start; `-DGGML_METAL_EMBED_LIBRARY=OFF` builds a `default.metallib` that must stay next to the executables.

## Checks

```sh
ctest --test-dir build-metal --output-on-failure -R trellis
MTL_SHADER_VALIDATION=1 ./build-metal/bin/test-trellis MTL0
GGML_METAL_SHARED_BUFFERS_DISABLE=1 ./build-metal/bin/test-trellis MTL0
```

`test-trellis MTL0` uploads coded weights, side tensors and activations to Metal, runs the graphs and compares the results with independently decoded CPU products (tolerance `2e-4 * (1 + |reference|)`). It covers all six bit widths, identity, block and whole-slice transforms, fused output parts and shards, permutations, non-contiguous and four-dimensional inputs, several token counts, and mixed CPU/Metal graphs with CPU embeddings feeding Metal output heads. A requested Metal device that is missing fails the test.

The `NariQ CPU and Metal checks` workflow compiles the Metal shaders on GitHub's hosted macOS runners and runs the CPU reference there. Hosted runners have no usable Metal GPU, so that job is a build check, not GPU evidence.

## Tested configuration and limits

Only one machine has run this: an M3 Pro with 18 GB of unified memory, macOS 14.6, Xcode 16.2.

- The device test above passes, with shader validation and with private buffers.
- For all three budgets, short full-model captures on CPU and Metal agree: identical greedy choices at every captured position, maximum full-vocabulary KL below 1e-8.
- Decode runs at about 4-5 tokens/s and prompt processing at about 27 tokens/s. The M3 Pro's 150 GB/s memory bandwidth caps decode near 20-25 tokens/s for these files. Kernel redesigns tried so far (register matrix products, split-K, on-chip codebooks) were slower.
- Not qualified: vision, long context, concurrent requests and other Macs.

This speed is too slow for long coding-agent sessions: a two-task DeepSWE subset run on the M3 Pro with the 2.0bpw file timed out on both tasks.

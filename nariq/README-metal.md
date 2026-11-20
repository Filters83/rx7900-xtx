# Metal (experimental)

NariQ coded linear layers (`GGML_OP_TRELLIS`) run on Apple GPUs with the weights kept compressed, for decode and
prefill, including banded output heads. Coded embedding lookups run on the CPU. There is no macOS package; build from
source.

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

`test-trellis MTL0` compares Metal results with independently decoded CPU products (tolerance
`2e-4 * (1 + |reference|)`) across all bit widths, transforms, permutations, input shapes and mixed CPU/Metal graphs.
CI compiles the shaders on hosted macOS runners, which have no usable GPU: a build check only.

## Tested configuration and limits

Only one machine has run this: an M3 Pro with 18 GB of unified memory, macOS 14.6, Xcode 16.2.

- The device test above passes, with shader validation and with private buffers.
- For all three budgets, short full-model captures on CPU and Metal agree: identical greedy choices at every captured position, maximum full-vocabulary KL below 1e-8.
- Decode runs at about 3.8 (3.0bpw) to 4.9 (2.0bpw) tokens/s (prompt about 27 tokens/s), well below the 15-22
  tokens/s that the M3 Pro's 150 GB/s memory bandwidth allows for these files; too slow for long agent sessions.
- Not qualified: vision, long context, concurrent requests and other Macs.

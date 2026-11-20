## Overview

<!-- What this PR changes and why. Changes that do not touch NariQ belong upstream: https://github.com/ggml-org/llama.cpp -->

## Hardware tested

<!-- GPU (or CPU / Apple chip), driver, operating system. -->

## Checks

- [ ] NariQ tests pass (`python3 -m unittest discover -s nariq/tests` and the `ctest` subset in [nariq/README.md](https://github.com/nari-labs/nari-llama.cpp/blob/nariq/b12431/nariq/README.md))
- [ ] Kernel or numeric changes: F32 runtime parity still passes for the 3.0, 2.5 and 2.0bpw files (mean KL <= 0.001, top-1 >= 98%; see [nariq/cuda-qualification.md](https://github.com/nari-labs/nari-llama.cpp/blob/nariq/b12431/nariq/cuda-qualification.md)), or greedy tokens are identical to the previous build
- [ ] Speed changes: `llama-bench` decode and prefill tokens/s before and after, on the same machine

<!-- Paste results here, or write "not applicable" next to a check. -->

## AI usage disclosure

<!-- YES / NO. If yes, say how AI was used, and add an `Assisted-by:` trailer to the commits (policy: https://github.com/nari-labs/nari-llama.cpp/blob/nariq/b12431/CONTRIBUTING.md; trailer format: https://github.com/nari-labs/nari-llama.cpp/blob/nariq/b12431/AGENTS.md). -->

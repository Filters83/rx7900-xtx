# CUDA qualification: b12431-nariv0

The release binaries are built from commit `9fc1bfa`, named `30f794c` in `build-info.json` and below: the same source tree, before the commit messages were reworded. The checks below ran on these binaries with the launcher's pinned model files. The GPU checks used the archives under the release's earlier name, `b12431-nari2`; the Linux archives were then repacked under the new name with every member unchanged (contents, modes, owners, times), and the Windows zips, which carry no name inside, were only renamed. These are engineering checks on Modal; none of them published anything. After the checks, the license files in the two `bin` archives were replaced when the NariQ additions moved to the MIT license (`LICENSE-NariQ` removed; `NOTICE-NariQ` and `README.md` updated; `LICENSE-exllamav3` added). Every other member, including every binary and `build-info.json`, is byte-identical to the checked archives, so the checks below still apply; the two `bin` archives have new hashes.

## Build

Linux: a `git archive` of `30f794c` (no Git metadata) built like `nariq/Dockerfile`'s `build` stage, on a 64-core Modal CPU container: `nvidia/cuda:12.8.2-devel-ubuntu22.04` (amd64 `sha256:b212f94014ff078bf5aa96bc1ee43cace0ba95d0542ea8b52dbcf42264971054`), the Dockerfile's apt packages and `cmake==3.31.6`, only the base image's environment, `NARIQ_SOURCE_REVISION=30f794c`, then `bash nariq/build_linux.sh /dist` (CUDA 12.8.93, GCC 11.4.0; real code for 8.6, 8.9, 9.0 and 12.0a, PTX for 5.0-9.0). `build-info.json` records revision `30f794c`, and its 3,531 source hashes equal the commit's files (the other 70 entries are the pinned UI).

Windows: the `NariQ Windows artifacts` workflow at `30f794c` ([run 44435062044](https://github.com/nari-labs/nari-llama.cpp/actions/runs/44435062044)): MSVC 2022 with CUDA 12.8 for the CUDA backend, clang for the rest. The runner extracted both zips, ran `llama-server --version` and loaded the packaged CPU backend. No Windows GPU has run these files.

| File | Bytes | SHA256 |
| --- | ---: | --- |
| `nari-llama.cpp-b12431-nariv0-bin-linux-cuda-12.8-x64.tar.gz` | 257,432,658 | `1bf65e866d46323594ffacaf0de95120677e954cdd80a91a4f1b9c91a7e5836a` |
| `cudart-nari-llama.cpp-b12431-nariv0-linux-cuda-12.8-x64.tar.gz` | 594,266,099 | `ec215628695711b8b80e232403165f0eb1d140cf7a28d804fe085b1eae5cddcc` |
| `nari-llama.cpp-b12431-nariv0-bin-win-cuda-12.8-x64.zip` | 224,585,591 | `9e52858b7f067c635da54a42981affbf237abeefec93602c2b825fb7136001c1` |
| `cudart-nari-llama.cpp-b12431-nariv0-win-cuda-12.8-x64.zip` | 563,452,046 | `fe20366827d357c00797eebb58244dddab7fd9a348d70090c3871004c320f38d` |
| `run.sh` | 7,917 | `9eebff519f522f07cccc06fa8c780ca28b449e5478e489b7aa0cefd5f69dbc30` |
| `run.ps1` | 6,676 | `fce57fe42d2e3361ed18ce4c4e11f7339d40be25e2655d17d8b75915fe60a8d8` |

## Results

| Check | GPUs | What it covers | Result |
| --- | --- | --- | --- |
| F32 runtime parity (`20261119T012811Z`) | B200, one per budget | Coded prefill (`-ub 512`) and coded decode (`-ub 1`) against an F32 decode of the same coded GGUF; 510 scored positions; gate fixed in advance: mean full-vocabulary KL <= 0.001, top-1 >= 98% | Pass, all three files (table below) |
| Token parity and speed (`fork-qual-20261119T012742Z`) | RTX PRO 6000, H100, L4 | Greedy token ids against the previous build of this release (program archive `95de72b0…`) on the same GPU: all three files, 4K-64K input, 51 requests; single-session decode speed; serving checks with the model-card command and `run.sh` | All 51 requests token-identical; speed within 0.5% (table below); serving checks pass |
| Serve matrix (`serve-matrix-20261119T012915Z`) | RTX PRO 6000, H100, L4, A10G | All three files and the vision projector through `run.sh` (fitted context) and the model-card command (`-c 131072`): chat with and without thinking, a tool call and its result, two images, a needle at 248,694 tokens (80/96 GB) or 59,658 tokens (24 GB, capped); eight sessions on the RTX PRO 6000 (255.4 / 268.2 / 290.7 tokens/s in total for 3.0 / 2.5 / 2.0) | Pass on every GPU and file |
| Launch rehearsal (`rehearsal-20261119T033256Z`) | RTX PRO 6000 | The staged release folder served locally; in plain `ubuntu:22.04` an unprivileged user fetches `run.sh` with curl and runs `bash run.sh 3.0`, `2.5` and `2.0` with no flags; every download hashed again; serving checks and a 65,160-token needle | Pass for all three files |
| Container image (`image-check-20261119T023705Z`) | RTX PRO 6000 | The serving image as the `NariQ release image` workflow builds it ([run 44445105781](https://github.com/nari-labs/nari-llama.cpp/actions/runs/44445105781), not pushed), here from the GitHub-built Linux archives of the same code (`dde5de3`, [run 44432326235](https://github.com/nari-labs/nari-llama.cpp/actions/runs/44432326235)), since the release files are not published yet: its entry point with the arguments of `run.sh --docker` (3.0 and the projector), its health check, the serving checks and a 32,583-token needle. `/app` equals those archives file for file. On launch day the same workflow builds the image from the release's own archives and checks them against `SHA256SUMS` | Pass |

F32 runtime parity (mean KL / top-1):

| File | Coded prefill | Coded decode |
| --- | ---: | ---: |
| 3.0bpw (`1eaad934…`) | 0.000168 / 100.0% | 0.000056 / 100.0% |
| 2.5bpw (`2445372e…`) | 0.000122 / 99.8% | 0.000101 / 99.8% |
| 2.0bpw (`3a9e4955…`) | 0.000237 / 99.6% | 0.000607 / 99.4% |

All nine logit files (F32 reference, coded prefill, coded decode per file) are byte-identical to those of the previous build (`95de72b0…`), and the F32 references match their earlier hashes. The F32 references are produced by `nariq/decode_reference.py` from the coded GGUFs; the logits come from `llama-perplexity` changed only to write uncompressed F32 logits, compared with a float64 log-softmax. This is a same-weight runtime check on a short sample, not a model-quality measurement.

Single-session decode on the RTX PRO 6000, tokens/s (512 greedy tokens, median of three, `-np 1 -ngl all -fa on -ctk bf16 -ctv bf16`; the previous build in parentheses):

| File | 4,096 | 16,384 | 65,536 input tokens |
| --- | ---: | ---: | ---: |
| 3.0bpw | 89.50 (89.29) | 85.21 (85.01) | 71.78 (71.61) |
| 2.5bpw | 99.98 (99.86) | 94.72 (94.50) | 78.35 (78.27) |
| 2.0bpw | 112.91 (112.72) | 106.23 (105.67) | 85.64 (85.81) |

## BF16 references

A BF16 decode of the same coded weights does not meet the same KL limit, so it is not the release gate. Against once-rounded BF16 references, coded prefill gives mean KL 0.0016 (3.0), 0.0044 (2.5) and 0.0216 (2.0), and the BF16 reference already differs from itself between prefill and decode by 0.0014, 0.0016 and 0.0076. The difference comes from BF16 weight and activation rounding, not from the coded kernels: the F32 comparison above isolates the kernels. Earlier BF16 references built from assembled checkpoints were rounded twice and are not used.

## Not covered

- No Windows GPU has run the Windows archives; they are experimental.
- No GeForce card has been tested.
- Driver minimums (Linux 570, Windows 572) are documented, not tested; the checks used drivers 610 and 620.
- Metal is covered in [README-metal.md](README-metal.md).

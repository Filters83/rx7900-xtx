# CUDA candidate qualification

Status as of 2026-11-18: the release build is the Linux build of `642881a` below. It is the code of the earlier `19da27f` build (now `3991d37` after the commit-author rewrite), rebuilt so that the binaries report their build number and commit, and every gate was run again on it. Release decisions (2026-11-18): the direct-GGUF F32 runtime-parity check is the numerical release gate; the BF16 cross-precision results below stay in this record, are not marked passed, and are disclosed. The Windows archives ship as experimental (GPU execution untested), and GeForce cards are disclosed as untested. Publication still needs explicit approval ([PUBLISH.md](PUBLISH.md)).

## Frozen artifacts (final release build 642881a, 2026-11-18)

Source revision `642881a93e2d360df38d756cea1a3173e9cc76fc`. Before this build every NariQ commit was rewritten to a new author identity with byte-identical trees, so the earlier build revision `19da27f` is now `3991d37`. `git diff 3991d37 642881a -- ggml src tools common include` is empty. The commits in between change documentation, launchers, CI files and `nariq/build_linux.sh`, which now passes `-DLLAMA_BUILD_NUMBER` (from the tag) and `-DLLAMA_BUILD_COMMIT` when the source has no Git metadata. `llama-server --version` prints `version: 1.0.0-dev (build 12431, commit 642881a)`; the `19da27f` build printed `build 0, commit unknown`.

Built on a Modal CPU container (64 cores) the way `nariq/Dockerfile`'s `build` stage builds: `nvidia/cuda:12.8.2-devel-ubuntu22.04` (amd64 `sha256:b212f94014ff078bf5aa96bc1ee43cace0ba95d0542ea8b52dbcf42264971054`), the Dockerfile's apt packages and `cmake==3.31.6`, the source a `git archive` of `642881a` (no Git metadata, no local build folders), the base image's environment, `NARIQ_SOURCE_REVISION=642881a`, then `bash nariq/build_linux.sh /dist` (CUDA 12.8.93, GCC 11.4.0, all release architectures). `build-info.json` records `642881a` with `dirty: null` and 3,603 source hashes: 3,533 equal the commit's files and 70 are the pinned UI. The CMake options and compiler versions equal those of the `19da27f` build, and the program archive holds the same 67 file names. A first Modal build let Modal's container variables into the build: `SOURCE_DATE_EPOCH` turned off `GGML_NATIVE_DEFAULT`, and with it the `GGML_AVX/AVX2/FMA/F16C/BMI2/SSE42` cache options, and `CFLAGS=-g0` was set. That build was set aside unused, and the build was repeated with only the base image's environment.

| File | Size (bytes) | SHA256 |
| --- | ---: | --- |
| Linux program archive `nari-llama.cpp-b12431-nari2-bin-linux-cuda-12.8-x64.tar.gz` | 257,836,448 | `95de72b0fa6d3c758b5a0ce5be951fc06dac2b44c526ab649087377da84dccb6` |
| Linux CUDA runtime archive `cudart-nari-llama.cpp-b12431-nari2-linux-cuda-12.8-x64.tar.gz` | 594,376,385 | `1e3d255da4a8af0f06aa0a98682797bcf9e3c41600041d7d31ba9e17ff736f8a` |
| `build-info-linux.json` | 439,176 | `88393d4a4cb1cfc4314f251169df8283e3210985430b0a094ef54da9eaca0cd9` |
| Linux launcher `run.sh` (staged, committed) | 7,916 | `0f3de7ab0090c98c436c83507ad0047cb9106ab204776ddaecc5173a8947f115` |
| Serving image `nari-llama.cpp:b12431-nari2` (local, not pushed) | 891,983,781 | image id `da0d7a8e7555a8bc73fb2a8139c33dbbf63fe9cbd0f24e14708207c6dbcb60ba`; `docker save` `46d372dcf547a38b0d45a7e2dae544136320a3bd1609b511de5502f48afce144` |
| Windows archives | | pending the Windows workflow on the release branch |

Checks on these exact archives, with the launcher's pinned model files. All are unscored engineering checks; none published anything.

| Run | GPUs | Scope | Result |
| --- | --- | --- | --- |
| `fork-qual-20261118T151039Z` | RTX PRO 6000 (driver 610.57.04), H100, L4 | Greedy token ids against the `19da27f` build (program archive `7e1a6937…`) on the same GPU: all three files, 4K-64K input, 51 requests; single-session decode speed; serving checks through the model-card command (RTX PRO 6000), `run.sh 3.0` (H100, L4) and `run.sh 2.0 --no-vision` (L4) | All 51 identical; decode speed in the table below; serving checks pass |
| `serve-matrix-20261118T151212Z` | RTX PRO 6000, H100, L4, A10G | All three files and the projector through `run.sh` (fitted context) and the model-card command (`-c 131072`): thinking and non-thinking chat, tool call and result, two images, a needle at 248,694 tokens (80/96 GB) or 59,658 tokens (24 GB, capped); eight sessions at `-np 8 -c 262144` on the RTX PRO 6000 (251.1 / 259.3 / 277.2 tokens/s in total for 3.0 / 2.5 / 2.0) | Pass on every GPU and file |
| `rehearsal-20261118T151138Z` | RTX PRO 6000 | A new user's first runs: the staged release folder served on localhost, plain `ubuntu:22.04`, an unprivileged user fetches `run.sh` with curl and runs `bash run.sh <budget>` for 3.0, 2.5 and 2.0 with no flags, the model files from the private staging model repository; every downloaded file hashed again; serving checks and a 65,160-token needle | Pass for all three files |
| `image-check-20261118T151750Z` | RTX PRO 6000 | The serving image's own filesystem as the container image, its entry point with the arguments of `run.sh --docker` (3.0 and the projector), the image's health check, the serving checks and a 32,583-token needle; `--version` in the image prints build 12431, commit 642881a | Pass. `/app` equals the archives file for file, including the library symlinks, in the exported image; in the Modal container the ten library symlinks arrive as regular files with the same content |

Single-session decode on the RTX PRO 6000 (same flags as the `19da27f` record below; 512 greedy tokens, median of three, tokens/s), `642881a` build against the `19da27f` build in the same container:

| Budget | 4,096 | 16,384 | 65,536 input tokens |
| --- | ---: | ---: | ---: |
| 3.0bpw | 89.08 (88.85) | 84.80 (84.84) | 71.43 (71.48) |
| 2.5bpw | 99.63 (99.48) | 94.11 (94.07) | 75.87 (77.94) |
| 2.0bpw | 112.12 (112.09) | 105.22 (105.05) | 85.55 (85.54) |

Nine of the ten cells are within 0.3% of the `19da27f` build, and within 1% of the values on the model card (from run `fork-qual-20261118T112257Z`). The exception is 2.5bpw at 65,536 tokens: 2.7% below the `19da27f` build in the same container and 2.1% below the card's 77.45. The three timed requests of each server agree within 0.2%. The RTX PRO 6000 runs identical GPU code in both builds: the sm_120a SASS of the two `libggml-cuda.so` files is byte-identical (`cuobjdump -sass -arch sm_120a`, 1,812,264,566 bytes each), though the files themselves differ in packaging. The gap therefore comes from the server instance, not the build. With the model-card command (`-ngl 99 -c 131072 --jinja`), 3.0bpw at 4,096 tokens: 82.81 (83.07 for the `19da27f` build). In the earlier run the same `19da27f` archives measured 59.82 in this cell, so it varies with the container, not with the build. The tokens were identical in all 51 requests.

## Final release build 642881a: F32 runtime parity

Run 2026-11-18 on the exact archives above (program `95de72b0…`, CUDA runtime `1e3d255d…`, both hashed in the container; `build-info.json` revision `642881a`). The harness, inputs and limits are those of the `19da27f` run in the next section, unchanged (every harness file has the same SHA256): one NVIDIA B200 per budget (183,359 MiB, driver 610.57.04), the same corpus and positions, and mean full-vocabulary KL <= 0.001 and top-1 >= 98% for coded prefill (`-ub 512`) and coded decode (`-ub 1`) against the direct-GGUF F32 reference.

| Budget (coded GGUF) | Run | Coded prefill (`-ub 512`) KL / top-1 | Coded decode (`-ub 1`) KL / top-1 | Native KL / same top-1, prefill / decode | Gate |
| --- | --- | ---: | ---: | ---: | --- |
| 3.0 (`1eaad934…`) | `20261118T151153Z-f32-parity-642881a-3.0-3412f4` | 0.000167679 / 100.000% | 0.000055761 / 100.000% | 0.000170 / 100.000%, 0.000057 / 100.000% | Passed |
| 2.5 (`2445372e…`) | `20261118T151153Z-f32-parity-642881a-2.5-c5c14e` | 0.000122308 / 99.804% | 0.000100768 / 99.804% | 0.000122 / 99.804%, 0.000100 / 99.804% | Passed |
| 2.0 (`3a9e4955…`) | `20261118T151153Z-f32-parity-642881a-2.0-ac1907` | 0.000237184 / 99.608% | 0.000607337 / 99.412% | 0.000236 / 99.608%, 0.000606 / 99.412% | Passed |

All six gate comparisons pass. Every regenerated F32 reference matches its 2026-11-17 hash, and all nine uncompressed logit files (F32 reference, coded prefill, coded decode per budget) are byte-identical to those of the `19da27f` run, so every value equals that run's. Each container completed on its first launch, and nothing was retried. This is a same-weight runtime-parity check on a short sample; it says nothing about model quality.

## Earlier build 19da27f (2026-11-18, before the author rewrite)

Revision `19da27f` is the commit before the NariQ commits were rewritten to a new author identity; its tree equals `3991d37`, whose `ggml`, `src`, `tools`, `common` and `include` equal `642881a`'s. The runs in this section and the next were made on the `19da27f` build and are kept as they were. Its archives are superseded by the final release build above.

Source revision `19da27f9ccc78eee66efa15dd1a225c11595f544`, built with `nariq/Dockerfile` (CUDA 12.8.2, Ubuntu 22.04, all release architectures). The build context had no Git metadata, so `build-info.json` records `19da27f` with `dirty: null`; its 3,530 source hashes equal the commit's files, plus the pinned UI under `tools/ui/dist` (checked by `assemble_page.py`). The tag is `b12431-nari2`; it is not published.

| File | SHA256 |
| --- | --- |
| Linux program archive `nari-llama.cpp-b12431-nari2-bin-linux-cuda-12.8-x64.tar.gz` | `7e1a6937f16cb5afb72d7ca1b0645af8dac22c2712b115ab830f9a378c4fb14f` |
| Linux CUDA runtime archive `cudart-nari-llama.cpp-b12431-nari2-linux-cuda-12.8-x64.tar.gz` | `63b6e85172ed68f0d2cca1852008879822c93c0d1bc00cc3fa194218e4b6808a` |
| Tested Linux launcher `run.sh` | `904d3194a4038c3d7904da41af1be0141ab60901a83d5ecde2336e32d5a28ba5` |
| Windows archives | pending the Windows workflow at this revision |

Checks on these exact Linux archives, with the launcher's pinned model files:

| Run | GPUs | Scope | Result |
| --- | --- | --- | --- |
| `fork-qual-20261118T112257Z` | RTX PRO 6000 (driver 610.57.04), H100, L4 | Greedy token ids against the previous candidate build (source `bce471c`; program archive `d4bb0032768625bbb18e8de4670f725c825b7372ee692e1c3ef510aa54046037`) on the same GPU: all three files, 4K-64K input, 51 requests; single-session decode speed; `run.sh` and model-card command serving checks | All 51 identical; tuned decode 1.04-1.06x the previous build; serving checks pass |
| `serve-matrix-20261118T112345Z` | RTX PRO 6000, H100, L4, A10G | All three files and the projector through `run.sh` (fitted context) and the model-card command (`-c 131072`): thinking and non-thinking chat, tool call and result, two images, a needle at 248,694 tokens (80/96 GB) or 59,658 tokens (24 GB, capped); eight sessions at `-np 8 -c 262144` on the RTX PRO 6000 | Pass on every GPU and file |

Single-session decode on the RTX PRO 6000 with `-np 1 -ngl all -fa on -ctk bf16 -ctv bf16 -t 1 -tb 16 --cache-ram 0 --backend-sampling --samplers temperature` (512 greedy tokens, median of three, tokens/s): 3.0bpw 88.87 / 84.48 / 70.68, 2.5bpw 99.21 / 93.76 / 77.45, 2.0bpw 111.74 / 105.10 / 85.03 at 4,096 / 16,384 / 65,536 input tokens. With the model-card command (`-ngl 99 -c 131072 --jinja`), 3.0bpw at 4,096 tokens: 59.82. Eight sessions: 254.5 / 263.9 / 275.2 tokens/s in total. Both runs are unscored engineering checks; they published nothing.

Not repeated on this build: the BF16 reference comparisons, explicit partial offload, and the B200, A100 and L40S startup checks below, which used the superseded 2026-11-17 candidate. Between that candidate and `19da27f` the CUDA kernels, fusions and the qwen35 decode graph changed; the commits that measured their output record identical greedy tokens on short samples, and the run above shows identical tokens against `bce471c`. The F32 runtime-parity gate was repeated on `19da27f` (next section).

## Earlier build 19da27f: F32 runtime parity

Run 2026-11-18 on the exact Linux archives above (program `7e1a6937f16cb5afb72d7ca1b0645af8dac22c2712b115ab830f9a378c4fb14f`, CUDA runtime `63b6e85172ed68f0d2cca1852008879822c93c0d1bc00cc3fa194218e4b6808a`; both hashes checked in the container; `build-info.json` revision `19da27f`). One NVIDIA B200 per budget (183,359 MiB, driver 610.57.04), the GPU of the 2026-11-17 F32 record; the 108 GB F32 reference does not fit an 80 or 96 GB card. Unscored engineering check; nothing was published.

Fixed before execution, identical to the 2026-11-17 corrected-reference check: input SHA256 `b46edf2f31842f098af0f1fdd221564bfc42ebb1aa200caeca66a388332d4474`; `-c 512 --chunks 2 -b 512 -t 8 -ngl 99 -fa on --no-warmup` (two 512-token chunks, 510 scored positions); the F32 reference decoded from each coded GGUF by `decode_reference.py` with the 2026-11-17 decoder library (`4cc83c49…`) and run at `-ub 512`; coded prefill at `-ub 512` and coded decode at `-ub 1`, each compared with the F32 reference. Gate: mean full-vocabulary KL <= 0.001 (float64 log-softmax over uncompressed F32 logits) and top-1 agreement >= 98%, for both coded paths. Uncompressed logits come from the archive's libraries driven by `tools/perplexity/perplexity.cpp` at `19da27f`, changed only to write F32 logits and the vocabulary (the same change as on 2026-11-17). The archive's own `llama-perplexity` native KL, which compresses reference probabilities, is reported alongside; it is not the gate.

Each regenerated F32 reference is byte-identical to its 2026-11-17 file (source and output hashes below), and its logits equal the 2026-11-17 reference logits (KL 0, identical file hashes). All coded logit files are also byte-identical to the 2026-11-17 candidate's, so every value below equals the 2026-11-17 F32 result.

| Budget (coded GGUF) | Run | F32 reference SHA256 | Coded prefill (`-ub 512`) KL / top-1 | Coded decode (`-ub 1`) KL / top-1 | Native KL / same top-1, prefill / decode | Gate |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 3.0 (`1eaad934…`) | `20261118T125436Z-f32-parity-19da27f-3.0-dbe9e5` | `36399a2f…` | 0.000167679 / 100.000% | 0.000055761 / 100.000% | 0.000170 / 100.000%, 0.000057 / 100.000% | Passed |
| 2.5 (`2445372e…`) | `20261118T125436Z-f32-parity-19da27f-2.5-e6d3ad` | `f771e1cd…` | 0.000122308 / 99.804% | 0.000100768 / 99.804% | 0.000122 / 99.804%, 0.000100 / 99.804% | Passed |
| 2.0 (`3a9e4955…`) | `20261118T125436Z-f32-parity-19da27f-2.0-0016e7` | `121fa4d0…` | 0.000237184 / 99.608% | 0.000607337 / 99.412% | 0.000236 / 99.608%, 0.000606 / 99.412% | Passed |

Largest single-position KL: 3.0 0.058 / 0.013, 2.5 0.045 / 0.020, 2.0 0.024 / 0.156 (prefill / decode). Coded prefill against coded decode: 0.000119534, 0.000071453 and 0.000163917 mean KL. All six gate comparisons pass the unchanged limits. This is a same-weight runtime-parity check on a short sample; it does not establish bitwise equality with the F32 reference or model quality. `test-backend-ops` is not in the release archive and was not run.

Attempts: a first launch (`20261118T125402Z`) failed at container start-up from a harness packaging error before any command ran; it measured nothing and is kept. The run above then completed in all three containers. Its controller lost network access while polling, so the results were collected from the run's output storage afterwards; the containers' own records are complete. Nothing was retried. Estimated compute for both attempts: about $4.8.

## Superseded: frozen artifacts of the 2026-11-17 candidate

Superseded by the 2026-11-18 builds above. This section and the results below it are the record of the earlier candidate and stay unchanged.

Status of that candidate as of 2026-11-17: the BF16 failure investigation is complete; publication remains on hold. A reference-construction bug is fixed, and all three budgets pass corrected F32 runtime-parity checks at the unchanged limits. BF16 cross-precision checks remain above the KL limit and are not marked passed. Native Windows GPU execution and a real 16/24 GB GeForce test are also outstanding.

These checks use the Linux Actions artifact from run `43847930005`, paired with the Windows artifact from run `43847930030`. Both record clean source revision `459d5976a2fc7cef569b09d29fe60bb4a722efe4`, the PR merge revision containing runtime commit `70b6c0a02eda62828a7764faaf2c006321e55b4c`. The reserved tag is `b12431-nari2`; it is not published. Later documentation edits are not a claim that newer binaries were tested.

| File | SHA256 |
| --- | --- |
| Linux program archive | `53b4db31ccacac89a657c658a7419cb92fcde9536c77449dddc5d797196b9cf0` |
| Linux CUDA runtime archive | `aa8c79b02243f9bef16f6501b6265ad55a608308b4e79e4c8a8e351be91dbff3` |
| Windows program archive | `c5c48f7ed6a8812bb1648c8cceff8f153e61fd1d939ee3f10d3462e0188f3c65` |
| Windows CUDA runtime archive | `fe20366827d357c00797eebb58244dddab7fd9a348d70090c3871004c320f38d` |
| Tested Linux launcher | `96d5b21664daa3b5a193bfcafd9149a67b655d44cf1c3c9d01063ad2b0aa6fd3` |
| Paired Windows launcher | `4211e59c25e13c2cfefd4eb42c05d3786d13c7af252121f7ad4231613d6df8f6` |

The assembler accepted the matching clean program/runtime pairs. The Linux launcher downloaded the real archives and all four pinned model files from an isolated HTTP fixture, checked their hashes, extracted them, and started the packaged server without an injected library search path. The web UI returned valid HTML. A subsequent comment-only launcher correction adds the already-documented `GLIBCXX_3.4.30` requirement and removes an unsupported blanket RHEL 9 compatibility claim; executable lines are unchanged. No runtime or model was published.

The model files retain the launcher's pinned hashes. BF16 reference hashes are:

| Budget | Original assembled BF16 reference SHA256 |
| --- | --- |
| 3.0 | `bd05d3d2b4625244363cb5a4211ee3f04048a8d27792476cb61c9210e44443f5` |
| 2.5 | `d9894c56fa8fc037f1ff3deb47ac5c8b37f5eb972b230936a8f6e932829a2b82` |
| 2.0 | `f1094c6aaa4b44eb793ff7569ae36bdc6db565b7d3f21e375bb69355383be7a1` |

## Functional checks

| Hardware | Scope | Result |
| --- | --- | --- |
| H100 80 GB | All three coded budgets and their BF16 references: thinking and non-thinking chat, tool call/result, a fixed tool loop, 26K-token needle, two vision cases | Passed |
| H100 80 GB | Eight concurrent requests for each budget; explicit 16/65-layer offload for 3-bit | Passed |
| L4 24 GB | All three budgets: the same serving suite; startup/generation with 16 GiB available | Passed |
| L4 with 16 GiB available | 3-bit, 128K context, automatic partial offload, 123,323-token needle prompt | Passed; 49/65 layers on GPU |
| A10, A100 40 GB, L40S, B200, RTX PRO 6000 | 2-bit initial/warm startup, GPU placement, UI and generation | Passed |
| Native Windows | Artifact build and CPU checks | Passed; GPU execution untested |
| Real 16/24 GB GeForce | Exact-package execution | Not available |

Logs confirm 65/65 layers offloaded in full-GPU cases and 16/65 in the explicit partial-offload case. Smaller-memory tests use a separate CUDA allocation to leave 16 GiB available; they do not establish GeForce qualification. Initial/warm startup was 46/6 seconds on A100 and 42/6 seconds on B200. These are individual functional observations, not performance benchmarks.

Linux test containers used Ubuntu 22.04 with driver 610.57.04, except A100 with 620.70.06. The documented minimum drivers remain Linux 570 and Windows 572; those minimum versions were not tested here. The seven launcher/packaging unit tests passed. Apple CPU runner capacity and Metal qualification are separate work.

## Numerical check

The original reference files came from assembled BF16 checkpoints. Follow-up investigation found intermediate rounding in their construction; they are not exact once-rounded decodes of the final coded GGUFs. See [VALIDATION.md](VALIDATION.md) for corrected reference generation. The fixed input SHA256 is `b46edf2f31842f098af0f1fdd221564bfc42ebb1aa200caeca66a388332d4474`. `llama-perplexity` evaluates two 512-token chunks, scoring 510 positions with its native compressed KL format. That format truncates small reference probabilities, so these original values are approximate KL, not a full-vocabulary sum. This short corpus is a runtime diagnostic, not a model-quality benchmark.

Before execution, the gate was fixed at mean full-vocabulary KL <= 0.001 and top-1 agreement >= 98%. The reference uses microbatch 512; coded paths use 512 and 1. A BF16 microbatch-1 comparison records batch-path sensitivity. Results and failures are retained without selecting successful retries.

| Budget | BF16 batch-path KL | Coded prefill KL / top-1 | Coded decode KL / top-1 | Gate |
| --- | ---: | ---: | ---: | --- |
| 3.0 | 0.010398 | 0.012491 / 98.824% | 0.016661 / 98.824% | Failed |
| 2.5 | 0.001867 | 0.004462 / 99.020% | 0.003841 / 99.020% | Failed |
| 2.0 | 0.012758 | 0.034907 / 97.843% | 0.029563 / 97.843% | Failed |

The 2-bit coded paths also fall below the top-1 threshold. The BF16 batch-path comparison itself exceeds the KL limit for every budget. This shows sensitivity to numerical execution paths, but does not make the coded-model gate pass.

The follow-up checks retain the same corpus and numerical limits. Identical-path BF16 self-comparisons give top-1 agreement of 100% and reported mean KL of -0.000001 for all three budgets, consistent with reference-format rounding. Comparing coded decode with a BF16 reference also generated at microbatch 1 reduces the discrepancy, but every budget still fails the KL limit.

| Budget | Matched decode KL | Top-1 | Gate |
| --- | ---: | ---: | --- |
| 3.0 | 0.002783 | 99.608% | Failed |
| 2.5 | 0.002250 | 99.020% | Failed |
| 2.0 | 0.021945 | 99.020% | Failed |

A separate regression check compares the frozen candidate directly with the previous `b12375-nari1` runtime, using identical coded weights, inputs and microbatch sizes. All six comparisons pass the unchanged numerical limits:

| Budget | Prefill KL / top-1 | Decode KL / top-1 |
| --- | ---: | ---: |
| 3.0 | -0.000001 / 100.000% | 0.000000 / 100.000% |
| 2.5 | 0.000000 / 100.000% | -0.000001 / 100.000% |
| 2.0 | 0.000000 / 100.000% | 0.000000 / 100.000% |

The previous program archive SHA256 is `076b757a333d73188ae058c888af65d6153cf6b38ead6b584e14aa14fdb5fc49`; its CUDA runtime archive SHA256 is `90d13cf14dd778e41f10ee829f4b996c7b66711d6f64f746fea0bb3b6158b704`. Both were verified before use.

No numerical migration regression is observed on this sample. This is evidence that the BF16 discrepancy predates the migration, not a pass for the BF16 gate. The reference-generation investigation below distinguishes this failure from coded-kernel arithmetic. These original failures remain in the record; thresholds and corpus have not been changed.

Reference generation:

```sh
./llama-perplexity -m bf16-decode.gguf -f validation.txt \
  -c 512 --chunks 2 -b 512 -ub 512 -t 8 -ngl 99 -fa on \
  --no-warmup --kl-divergence-base reference.kld
```

Comparison uses the same settings with `-m coded.gguf --kl-divergence`; repeat with `-ub 1` for the decode path. The tool stores reference probabilities in its native compressed format, which can produce tiny negative KL estimates from rounding. The threshold is applied to the reported mean without changing it after observing results.

## Reference investigation (2026-11-17)

The reference-selection bug is reproducible at the weight level. On the 2-bit `blk.0.ffn_down` matrix (5120 by 17408), an independent float64 decoder and the runtime's F32 CPU decoder differ by only `8.54e-8` relative RMS. The old BF16 reference differs from that mathematical decode by `0.002342` relative RMS. Only 74.945% of its entries equal the independently decoded matrix rounded once to BF16. Its bytes instead exactly match the assembled BF16 checkpoint, whose construction rounds before applying tuned scales and then rounds again. Emulating that intermediate rounding reproduces 99.988% of the assembled entries. It is a valid assembled checkpoint, but an incorrect exact-decode reference for kernel parity.

`decode_reference.py` now builds the reference from the actual coded GGUF, applies its transforms and gathers, restores vocabulary order, and preserves uncoded tensors. It records source, decoder and output hashes. It provides both once-rounded BF16 and F32 output, so BF16 weight rounding can be measured separately from coded-kernel arithmetic. Synthetic known-value tests exercise both output types, permutations, vocabulary bands and halfway rounding; the CPU workflow includes those tests.

The original 2-bit failure was also repeated using uncompressed F32 logits and a float64 full-vocabulary KL reduction on the H100. Token IDs, vocabulary order, shapes and finite values were checked before comparison. This independently confirms that compression was not responsible for the large original failure:

| Same frozen candidate, original references | Full-vocabulary mean KL | Top-1 |
| --- | ---: | ---: |
| Old BF16 prefill vs coded prefill | 0.034912947 | 97.843% |
| Old BF16 prefill vs old BF16 decode | 0.012769477 | 98.039% |
| Old BF16 decode vs coded decode | 0.021939589 | 99.020% |
| Coded prefill vs coded decode | 0.000257244 | 99.608% |
| Coded decode vs ungrouped coded decode | 0.000000000 | 100.000% |

Ten positions contribute 95.92% of the first row's KL; the worst is token index 460 with KL 11.4406. These outliers remain in the evidence. Eight sampled CUDA linears, covering attention projections, the FFN and output bands, have relative RMS error of about `2.04e-4` to `2.12e-4` against float64 products using CPU-decoded weights. An additional requested `blk.0.attn_output` probe reported a missing tensor and is excluded from that coverage. Grouping produces bit-identical decode logits on this sample. No serving-kernel change was made on the basis of the reference mismatch.

Corrected full-model diagnostics use a B200 (the F32 model requires more than 80 GB), the same frozen Linux runtime, the same corpus and all 510 scored positions. The original numerical limits are unchanged: mean KL <= 0.001 and top-1 >= 98%. Both a prefill reference compared with decode and matched-microbatch comparisons are retained. These are separate corrected-reference diagnostics, not replacements for the recorded original failures.

### Corrected-reference results

All nine F32 comparisons below pass the unchanged mean-KL and top-1 limits. This is a same-weight runtime-parity check: F32 avoids the additional BF16 weight rounding and BF16 matrix-multiply input rounding. It does not establish bitwise equality or model quality beyond this short sample.

| Budget | F32 prefill vs coded prefill KL / top-1 | F32 prefill vs coded decode KL / top-1 | Matched F32 decode KL / top-1 |
| --- | ---: | ---: | ---: |
| 3.0 | 0.000167679 / 100.000% | 0.000055761 / 100.000% | 0.000018325 / 100.000% |
| 2.5 | 0.000122308 / 99.804% | 0.000100768 / 99.804% | 0.000116562 / 99.804% |
| 2.0 | 0.000237184 / 99.608% | 0.000607337 / 99.412% | 0.000083811 / 100.000% |

The corrected BF16 references still fail the original prefill-reference gate. The remaining difference includes BF16 weight rounding and execution-path arithmetic; the BF16 references themselves exceed the KL limit between prefill and decode. No tolerance was relaxed, and these results are not marked passed:

| Budget | BF16 prefill vs coded prefill KL | BF16 prefill vs coded decode KL | Matched BF16 decode KL | BF16 prefill vs BF16 decode KL |
| --- | ---: | ---: | ---: | ---: |
| 3.0 | 0.001608518 | 0.001546879 | 0.000866830 | 0.001363188 |
| 2.5 | 0.004374524 | 0.005208110 | 0.001791077 | 0.001614137 |
| 2.0 | 0.021630961 | 0.019642403 | 0.004356498 | 0.007609271 |

The evidence supports a reference-construction bug plus a cross-precision comparison problem, rather than a CUDA migration regression. The fix is the direct-GGUF reference generator and explicit separation of kernel parity from BF16 rounding diagnostics. Serving kernels, model weights, corpus and numerical thresholds are unchanged. The original BF16 acceptance gate remains open; F32 parity passing does not silently redefine it.

The reference decoder is the frozen package's `libggml-base.so`, SHA256 `4cc83c49635ec41f56eedb768bbab14254ef972e71b98a3e0370961ada27d8dd`. B200 checks use driver 610.57.04. Generated reference hashes are:

| Budget | Once-rounded BF16 SHA256 | F32 SHA256 |
| --- | --- | --- |
| 3.0 | `4e6f3acb90e27f37fb05fe0069a25d45445206683eeebbdaeeba47d165e0bbaf` | `36399a2f72cfb3979c51c9d9aac6cc10417e2cf27c37edeffe2ab49a786d9293` |
| 2.5 | `d8903338ab873f6985684a3a7bbc85fc0cdd15a58cc3428dc6230292fb873d62` | `f771e1cd87cfd8bf6181f3103621436561e1efd4a035120c26f90a797dff2a29` |
| 2.0 | `f218b6264c9796d43f1db5f91c84c4c4d6179463474e2b41dbfab172705aeffc` | `121fa4d052ce5d08dbd7bd1f281f904b885fef831977a3dc203defc96b0751d5` |

Both reference types preserve uncoded tensors; their sizes are 53,808,282,368 and 107,594,988,288 bytes respectively. Ten local Linux reference/packaging tests pass. The new CPU workflow wiring has not been executed on macOS in this investigation.

## Evidence and scope

The run records retain input and artifact hashes, command lines, server logs, completion state and cost estimates. The fixed tool-loop outputs are trimmed functional records, not complete agent traces. No scored benchmark was started, resumed or retried.

Three initial attempts were stopped after the UI probe omitted `Accept-Encoding: gzip` and received the upstream server's expected HTTP 415 response. Their incomplete records remain available. The corrected probe decompresses and validates the UI response; it does not alter the runtime, weights, prompts or sampling. GPU-placement logging was also enabled.

The six investigation attempts include four completed diagnostics and two incomplete infrastructure attempts (a result-bookkeeping typo and a missing Python dependency). Both failures are retained. The initial 13 attempts are also retained. All investigation and earlier validation GPU containers are stopped. Raw logits and manifests have been downloaded and hash-verified; generated dense models were temporary diagnostic files, not published artifacts.

## Remaining release gates

As of 2026-11-18 the open items are tracked in [release-readiness.md](release-readiness.md). The gates listed for the 2026-11-17 candidate were resolved as follows:

- BF16 cross-precision acceptance: decided 2026-11-18. The F32 runtime-parity check is the release gate; the BF16 results stay in this record, not marked passed, and are disclosed in the release notes.
- Windows GPU execution: still untested. The Windows archives ship marked experimental.
- 16/24 GB GeForce card: still untested, and disclosed as untested on the model card and the release page.
- Model-card commands and platform claims: the model-card command ran on the 2026-11-18 candidate on four GPUs (above). Runtime requirements are Linux driver 570+, Windows driver 572+, and the documented system libraries.
- Publication approval: still needed. No release, tag, container image or model upload has been published.

Apple Silicon/Metal qualification is separate. Vulkan remains deferred.

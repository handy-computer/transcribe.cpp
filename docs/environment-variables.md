# Environment Variables

A single reference for the environment variables transcribe.cpp recognizes,
split by audience. The library follows one parse convention for boolean
toggles: a flag is **on** when it is set, non-empty, and its first character is
not `0` (so `FOO=1`, `FOO=on`, `FOO=yes` are on; `FOO=0`, `FOO=` and unset are
off). Path-valued vars are off when unset or empty.

## Tier 1 — runtime configuration

Compiled into every build, including release. These are operational knobs, not
build-gated. Note that `TRANSCRIBE_DUMP_DIR` — while it ships — is still a
tensor-dumping debug/validation hook: enabling it does per-tensor device→host
copies and writes raw activations to disk, so it carries I/O cost and a
data-leak surface. Leave it unset in production unless you are deliberately
collecting dumps.

The two `TRANSCRIBE_TEST_*_THROW` fault-injection hooks also ship in release
binaries on purpose: wheel clean-install CI proves exception containment
against the exact artifact users install (see `AGENTS.md`, "C ABI Exception
Discipline"). Unlike the boolean convention above, **any** non-empty value
arms them (including `0`); unset or empty is inert. Leave them unset outside
of tests.

| Variable | Effect |
| --- | --- |
| `TRANSCRIBE_NO_FLASH` | Disable flash attention on encoder and decoder (forces the manual F32 path). |
| `TRANSCRIBE_FORCE_FLASH` | Force flash attention on. Wins over `TRANSCRIBE_NO_FLASH` if both are set. |
| `TRANSCRIBE_CONV_DIRECT_DW` / `TRANSCRIBE_CONV_NO_DIRECT_DW` | Force the depthwise-conv dispatch to the direct `conv_2d_dw` path / the im2col path, overriding the per-family backend default. |
| `TRANSCRIBE_CONV_DIRECT_PW` / `TRANSCRIBE_CONV_NO_DIRECT_PW` | Force the pointwise-conv dispatch to direct `mul_mat` / im2col, overriding the backend default. |
| `TRANSCRIBE_DUMP_DIR=<dir>` | Enable the per-stage tensor dumper; writes `<name>.f32` + `<name>.json` per dumped tensor into `<dir>`. The basis for the numerical-comparison harness (`scripts/compare_tensors.py`). |
| `TRANSCRIBE_PERF_DEBUG` | Print a per-stage timing breakdown to stderr (DEBUG log) on the families that profile (`cohere`, `granite`, `canary`, `canary_qwen`, `moonshine`, `moonshine_streaming`, `qwen3_asr`, `whisper`). For whisper, a value containing `cpu` or `all` additionally prints the CPU sub-section breakdown. |
| `TRANSCRIBE_VOXTRAL_REALTIME_STREAM_TIMING` | Print a per-component streaming wall-time breakdown at stream finalize (voxtral_realtime). |
| `TRANSCRIBE_TEST_DEV_INIT_THROW=<match>` | Fault injection: backend device init (`ggml_backend_dev_init`) throws for devices whose name contains `<match>` (`*` matches every device). Exercises throw → skip → CPU-fallback in backend probing; an explicit backend request fails with `TRANSCRIBE_ERR_BACKEND`. Used by `backend_init_throw_unit` and `scripts/ci/vulkan_degradation_check.py`. |
| `TRANSCRIBE_TEST_TEARDOWN_THROW` | Fault injection: any non-empty value injects a throw after each real free inside the `transcribe::safe_*` teardown wrappers, proving containment without leaking the handle. Used by `teardown_safety_unit`. |

## Tier 2 — validation hooks

Numerical-parity hooks for porting and validation. **Compiled in only when the
library is built with `-DTRANSCRIBE_ENABLE_VALIDATION_HOOKS=ON`** (see the CMake
option of the same name, or the `validation` preset). Release/wheel builds leave
them out entirely — the env-var reads, ref-mel loaders, and dump scaffolding are
all behind the compile guard, so these names do not even appear in a release
binary. In a build *without* the flag, setting any of these has no effect;
`scripts/validate.py` hard-fails `--mel-from-ref` against such a build rather
than silently falling back.

| Variable | Effect |
| --- | --- |
| `TRANSCRIBE_MEL_FROM_REF=<dir>` | Inject a reference log-mel from `<dir>` instead of computing it, so encoder drift can be isolated from frontend drift. Each family reads its own dump filename/layout from the directory (`enc.mel.in.f32` for whisper / voxtral_realtime, `frontend.mel.out.f32` for gigaam, `mel.in.f32` for medasr). Driven by `scripts/validate.py --mel-from-ref`. |
| `TRANSCRIBE_DUMP_ALL_BLOCKS` | Dump every encoder block output (not just mid/last) for a layer-by-layer divergence bisect (parakeet, gigaam). Requires `TRANSCRIBE_DUMP_DIR`. |
| `TRANSCRIBE_DUMP_SUB_BLOCKS=<csv>` | Dump intermediate sub-layer activations (ff1/attn/conv/ff2) for the listed block indices, e.g. `0,12,23` (parakeet). Requires `TRANSCRIBE_DUMP_DIR`. |

Build a validation-capable `build/` with:

```bash
cmake --preset validation        # configures build/ with the hooks ON
cmake --build build --target transcribe-cli
```

When **toggling** `TRANSCRIBE_ENABLE_VALIDATION_HOOKS` on an existing `build/`,
do a clean build of the target — an incremental build can leave a stale object
that did not pick up the changed compile definition:

```bash
cmake --build build --clean-first --target transcribe-cli
```

(`scripts/validate.py` guards against the OFF case: it hard-fails
`--mel-from-ref` when `build/CMakeCache.txt` has the flag off. Longer term, a
dedicated validation build directory would avoid the toggle entirely.)

## Test & tooling

Not part of the library runtime.

**Real-model test model paths.** Built only with
`-DTRANSCRIBE_BUILD_REAL_MODEL_TESTS=ON`; each test skips (exit code 77) when
its var is unset. Convention: `TRANSCRIBE_<FAMILY>_GGUF`.

| Variable | Test(s) |
| --- | --- |
| `TRANSCRIBE_PARAKEET_GGUF` | `parakeet_real_smoke`, `decoder_smoke` |
| `TRANSCRIBE_COHERE_GGUF` | `cohere_real_smoke`, `cohere_e2e_smoke` |
| `TRANSCRIBE_WHISPER_GGUF` | `whisper_e2e_smoke`, `whisper_tokenize_parity` |
| `TRANSCRIBE_QWEN3_ASR_GGUF` (+ `_0_6B_GGUF` / `_1_7B_GGUF`) | qwen3_asr smokes / parity |
| `TRANSCRIBE_MOONSHINE_STREAMING_TINY_GGUF` | moonshine_streaming smokes |
| `TRANSCRIBE_VOXTRAL_REALTIME_GGUF` | `voxtral_realtime_real_smoke` |
| `TRANSCRIBE_WHISPER_BIN_*` | whisper.cpp `.bin` parser/e2e fixtures |

Other test/tooling vars:

- `TRANSCRIBE_TEST_AUDIO` — override the `jfk.wav` path for ad-hoc e2e runs.
- `TRANSCRIBE_TEST_SAMPLES_DIR` / `TRANSCRIBE_TEST_FIXTURES_DIR` — sample and
  fixture directories, passed at configure time as compile definitions (not
  runtime env vars).
- Python tooling: `TRANSCRIBE_MODELS_DIR`, `TRANSCRIBE_DUMP_DIR` (shared with
  the library), `HF_TOKEN`, and the binding-level `TRANSCRIBE_BACKEND` /
  `TRANSCRIBE_LIBRARY`. See [`tools/conversion.md`](tools/conversion.md) and
  [`tools/validate.md`](tools/validate.md).

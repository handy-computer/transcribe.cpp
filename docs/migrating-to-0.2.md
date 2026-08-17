# Migrating to transcribe.cpp 0.2

Version 0.2 is a deliberate pre-1.0 API and ABI break. Rebuild native
consumers against the 0.2 headers and upgrade each language package and native
provider together. The version and ABI-hash checks in the official bindings
reject mixed 0.1/0.2 installations.

## Device selection

The integer `gpu_device` selector has been replaced by an opaque,
process-local device handle. This removes the old collision where `0` meant
"automatic" and therefore could not select registry device 0.

The new rules are:

- Leave `device` unset (`NULL`, `None`, `nil`, or omitted) for the backend's
  automatic policy.
- Pass a device returned by the current process's enumeration API to select
  that exact primary device. Exact selection never falls back to another
  primary.
- An explicit `backend` and exact `device` must match. `AUTO` accepts a CPU,
  GPU, or integrated GPU. ACCEL devices such as BLAS/AMX cannot be primary;
  select the CPU device with `CPU_ACCEL` to layer them onto exact CPU.
- Handles and registry indices are not persistent identifiers. Persist
  `device_id` when the backend provides one, then enumerate and resolve a fresh
  handle after backend initialization in each process.
- Treat dynamic backend registration as startup-only: every
  `transcribe_init_backends*()` call must finish before any thread enumerates
  devices, queries backend availability, or loads a model. The native registry
  does not support racing registration against those operations.

### C API replacements

| 0.1 API | 0.2 API |
| --- | --- |
| `transcribe_backend_device_count()` | `transcribe_device_count()` |
| `transcribe_get_backend_device(index, &info)` | `transcribe_device_get(index)`, then `transcribe_device_get_info(device, &info)` |
| `struct transcribe_backend_device` | `struct transcribe_device_info` |
| `transcribe_backend_device_init()` | `transcribe_device_info_init()` |
| `transcribe_model_get_device(model, &info)` | `transcribe_model_device(model)`, then `transcribe_device_get_info(device, &info)` |
| `TRANSCRIBE_ABI_BACKEND_DEVICE` | `TRANSCRIBE_ABI_DEVICE_INFO` |
| `transcribe_model_load_params::gpu_device` | `transcribe_model_load_params::device` |

For automatic selection, initialize the params and leave `device == NULL`:

```c
struct transcribe_model_load_params params;
transcribe_model_load_params_init(&params);
params.backend = TRANSCRIBE_BACKEND_AUTO;
```

For exact selection, enumerate after registering dynamic backends and retain the
handle whose metadata matches the application's saved `device_id`. Never assign
an unchecked `transcribe_device_get()` result to model params: an out-of-range
index returns `NULL`, and `NULL` requests automatic selection.

```c
#include <string.h>

transcribe_device_t selected = NULL;
for (int i = 0; i < transcribe_device_count(); ++i) {
    transcribe_device_t device = transcribe_device_get(i);
    struct transcribe_device_info info;
    transcribe_device_info_init(&info);
    if (transcribe_device_get_info(device, &info) != TRANSCRIBE_OK) {
        continue;
    }
    if (info.device_type != TRANSCRIBE_DEVICE_TYPE_ACCEL &&
        info.device_id != NULL && strcmp(info.device_id, saved_device_id) == 0) {
        selected = device;
        break;
    }
}

if (selected != NULL) {
    struct transcribe_model_load_params params;
    transcribe_model_load_params_init(&params);
    params.device = selected;
    /* Load with params: exact selection is now guaranteed. */
} else {
    /* Report the unavailable device and do not load unless auto is intended. */
}
```

`struct transcribe_model_load_params` changed layout (16 to 24 bytes on the
supported 64-bit ABIs), and the old device symbols were removed. A 0.1 binary
must not load a 0.2 library without being rebuilt.

## Official bindings

All official bindings now pass an enumerated device object rather than a
registry integer.

### Python

```python
device = next(d for d in transcribe_cpp.backends() if d.device_type == "cpu")
model = transcribe_cpp.Model("model.gguf", device=device)
```

Replace `Model(..., gpu_device=index)` and the one-shot helper's `gpu_device=`
with `device=`. Omit `device` for automatic selection.

### Rust

```rust
let device = transcribe_cpp::devices()
    .into_iter()
    .find(|device| device.kind == "cpu")
    .unwrap();
let model = transcribe_cpp::Model::load_with(
    "model.gguf",
    &transcribe_cpp::ModelOptions {
        device: Some(device),
        ..Default::default()
    },
)?;
```

Replace `ModelOptions::gpu_device` with `ModelOptions::device`. Use `None` for
automatic selection.

### Swift

```swift
let device = Transcribe.devices().first { $0.deviceType == .cpu }!
let model = try Model(path: "model.gguf", options: ModelOptions(device: device))
```

Replace `ModelOptions(gpuDevice:)` with `ModelOptions(device:)`. Use `nil` for
automatic selection.

### TypeScript / JavaScript

```ts
const device = getAvailableBackends().find((device) => device.deviceType === "cpu");
if (!device) throw new Error("CPU device is not registered");
const model = await TranscribeModel.load("model.gguf", { device });
```

Replace `gpuDevice` with `device`. Version 0.2 rejects the removed
`gpuDevice` property at runtime so JavaScript callers cannot silently fall back
to automatic selection. The object must come from `getAvailableBackends()` or
`model.device`; copying its visible fields does not copy its opaque native
identity. Omit `device` for automatic selection.

## Command-line tools

`transcribe-cli` and `transcribe-bench` now interpret every non-negative
`--device N` as an exact index from `transcribe-cli --list-devices`.

In particular, **`--device 0` changed meaning**:

- 0.1: automatic selection
- 0.2: exact registry device 0

To retain automatic selection, remove the `--device` option. Device indices are
process-local and should be resolved again rather than stored in configuration.

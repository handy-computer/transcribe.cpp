# transcribe-cpp-native-cu12

CUDA 12 native provider for
[`transcribe-cpp`](https://pypi.org/project/transcribe-cpp/), the Python
bindings for [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp).

Install via the extra — it is **additive**, alongside the default provider:

```sh
pip install "transcribe-cpp[cu12]"
```

At runtime the best installed provider wins (CUDA outranks Vulkan/CPU); force
one with `TRANSCRIBE_NATIVE_PROVIDER=cu12` or per-model
`Model(..., backend="cuda")`.

What's inside: `libtranscribe` + ggml backend modules (conservative CPU +
CUDA, fatbins for sm 75/80/86/89/90). The CUDA toolkit is not vendored —
cudart/cublas come from the `nvidia-*-cu12` runtime wheels this package
depends on, and `libcuda.so.1` is your system driver. On a machine without
an NVIDIA driver the CUDA module simply doesn't load and CPU keeps working.

Linux x86_64 wheels only. Building from source with CUDA goes through the
`transcribe-cpp-native` sdist:

```sh
CMAKE_ARGS="-DTRANSCRIBE_CUDA=ON -DTRANSCRIBE_GGML_BACKEND_DL=ON" \
    pip install --no-binary :all: transcribe-cpp-native
```

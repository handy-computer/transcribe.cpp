# transcribe-cpp-native

Prebuilt native library for [`transcribe-cpp`](https://pypi.org/project/transcribe-cpp/),
the Python bindings for [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp)
(local speech-to-text on ggml).

You normally don't install this directly — `pip install transcribe-cpp`
depends on it. It carries no Python API; it bundles the `libtranscribe`
shared library plus its ggml backend libraries and registers them with the
bindings via the `transcribe_cpp.native` entry point.

What each artifact ships:

| Artifact | Backends |
| --- | --- |
| Linux x86_64 wheel | CPU (conservative baseline) + Vulkan backend module |
| Windows x86_64 wheel | CPU (conservative baseline) + Vulkan backend module |
| macOS arm64 wheel | Metal (embedded shaders) + CPU |
| sdist | builds from source; machine-tuned CPU by default, backends via `CMAKE_ARGS` |

The Vulkan backend ships as a dynamic ggml module: on machines without a
Vulkan loader or driver it simply fails to load and CPU keeps working — no
crash, no import error. Specialized providers (e.g. CUDA) are separate
packages installed alongside this one via extras of `transcribe-cpp`.

Building from the sdist requires CMake ≥ 3.21, Ninja, and a C++17 compiler;
backend selection follows the repo's CMake options, e.g.:

```sh
CMAKE_ARGS="-DTRANSCRIBE_VULKAN=ON" pip install --no-binary :all: transcribe-cpp-native
```

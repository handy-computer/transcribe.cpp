/**
 * Bring-your-own CUDA runtime preload — the TypeScript twin of the Python cu12
 * provider's prepare() hook (bindings/python-native-cu12/__init__.py).
 *
 * The opt-in `@transcribe-cpp/<host>-cuda` package ships libtranscribe plus the
 * ggml cuda/vulkan/cpu modules, but NOT the CUDA runtime (cudart/cublas) — the
 * same posture the cu12 wheel takes, where the runtime comes from the
 * nvidia-*-cu12 pip wheels. Node has no equivalent wheel ecosystem, so the user
 * supplies the runtime and points TRANSCRIBE_CUDA_RUNTIME_DIR at it; we dlopen
 * the libraries with RTLD_GLOBAL so the ggml-cuda module's DT_NEEDED sonames
 * resolve against the already-loaded set when transcribe_init_backends loads it.
 *
 * Everything here is best-effort and silent: with no env var (or missing libs)
 * this is a no-op. If a system CUDA toolkit is already on the default loader
 * path the ggml-cuda module resolves cudart on its own; otherwise it quietly
 * fails to load and the bundle's Vulkan/CPU modules keep working — the same
 * degradation contract as a missing driver.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import koffi from "koffi";

// Runtime libraries the ggml-cuda module links, in dependency order (cublasLt
// before cublas). Each entry is [nvidia-pkg, subdir, soname]: the soname is the
// flat name to dlopen; the pkg/subdir mirror the nvidia-*-cu12 wheel layout so
// pointing TRANSCRIBE_CUDA_RUNTIME_DIR at a `site-packages/nvidia` tree also
// works. Linux ships versioned sonames under <pkg>/lib/; Windows ships DLLs
// under <pkg>/bin/ (the layout PyTorch relies on).
const RUNTIME_LIBS: ReadonlyArray<readonly [string, string, string]> =
  process.platform === "win32"
    ? [
        ["cuda_runtime", "bin", "cudart64_12.dll"],
        ["cublas", "bin", "cublasLt64_12.dll"],
        ["cublas", "bin", "cublas64_12.dll"],
      ]
    : [
        ["cuda_runtime", "lib", "libcudart.so.12"],
        ["cublas", "lib", "libcublasLt.so.12"],
        ["cublas", "lib", "libcublas.so.12"],
      ];

// Keep loaded handles pinned for the process lifetime so the runtime never
// unloads out from under the ggml-cuda module.
const pinned: unknown[] = [];
let done = false;

/**
 * Preload the CUDA runtime from `TRANSCRIBE_CUDA_RUNTIME_DIR` (if set) so the
 * ggml-cuda module can resolve cudart/cublas. Idempotent; safe to call when the
 * selected provider has no cuda backend (it just won't find anything to load).
 */
export function preloadCudaRuntime(): void {
  if (done) return;
  done = true;

  const dir = process.env.TRANSCRIBE_CUDA_RUNTIME_DIR;
  if (!dir) return; // rely on a system toolkit already on the default loader path

  for (const [pkg, sub, soname] of RUNTIME_LIBS) {
    const candidates = [
      path.join(dir, soname), // a flat directory of runtime libs
      path.join(dir, pkg, sub, soname), // the nvidia-*-cu12 wheel layout
    ];
    const found = candidates.find((p) => fs.existsSync(p));
    if (!found) continue; // missing lib: degrade like a missing driver
    try {
      // RTLD_GLOBAL so the module's DT_NEEDED resolves against the loaded set;
      // an absolute-path load is also the load-bearing step on Windows (the
      // import table is satisfied by the already-loaded module, matched by name).
      pinned.push(koffi.load(found, { global: true, lazy: false }));
    } catch {
      // Same posture as a missing driver: never break the CPU/Vulkan path.
    }
  }
}

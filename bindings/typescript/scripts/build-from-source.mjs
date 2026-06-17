// Build libtranscribe from source — the universal fallback when no prebuilt
// @transcribe-cpp/<platform> package exists for the host (a new arch, a custom
// backend, a self-contained library). The result installs into
// `prebuilds/<tuple>/`, which the loader probes automatically — no env var.
//
//   node scripts/build-from-source.mjs [--source <repo>] [--self-contained]
//
//   --source           Path to a transcribe.cpp checkout (default:
//                      $TRANSCRIBE_SOURCE_DIR, else walk up to find the repo).
//   --self-contained   Link ggml statically into one libtranscribe (no sibling
//                      ggml libraries). Otherwise ggml ships as sibling shared
//                      libs alongside libtranscribe.
//
// This drives CMake directly (the binding is FFI over a shared library, not an
// N-API addon — so cmake-js, which builds .node addons, is not the right tool).
// Requires cmake + a C/C++ toolchain (Ninja recommended). On macOS Metal is
// enabled and embedded; on Linux the default CPU backend builds (add your own
// -D flags via TRANSCRIBE_CMAKE_FLAGS for CUDA/Vulkan/etc.).

import { spawnSync } from "node:child_process";
import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";

const PKG_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

function arg(name) {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? (process.argv[i + 1]?.startsWith("--") ? true : process.argv[i + 1] ?? true) : undefined;
}

function tuple() {
  const { platform, arch } = process;
  const a = arch === "x64" ? "x64" : arch;
  if (platform === "darwin") return arch === "arm64" ? "darwin-arm64-metal" : "darwin-x64-cpu";
  if (platform === "linux") return `linux-${a}-cpu-vulkan`;
  if (platform === "win32") return `win32-${a}-cpu-vulkan`;
  throw new Error(`unsupported platform ${platform}/${arch}`);
}

function findSource() {
  const explicit = arg("source") || process.env.TRANSCRIBE_SOURCE_DIR;
  const start = typeof explicit === "string" ? explicit : PKG_ROOT;
  let dir = path.resolve(start);
  for (let i = 0; i < 12; i++) {
    if (
      fs.existsSync(path.join(dir, "CMakeLists.txt")) &&
      fs.existsSync(path.join(dir, "include", "transcribe.h"))
    ) {
      return dir;
    }
    const up = path.dirname(dir);
    if (up === dir) break;
    dir = up;
  }
  throw new Error(
    "could not locate a transcribe.cpp source tree. Pass --source <repo> or set " +
      "TRANSCRIBE_SOURCE_DIR (the published npm package ships no C++ sources).",
  );
}

function run(cmd, args) {
  console.log(`$ ${cmd} ${args.join(" ")}`);
  const r = spawnSync(cmd, args, { stdio: "inherit" });
  if (r.status !== 0) {
    throw new Error(`${cmd} exited with ${r.status ?? r.signal}`);
  }
}

const src = findSource();
const t = tuple();
const buildDir = path.join(PKG_ROOT, "build-from-source");
const prefix = path.join(PKG_ROOT, "prebuilds", t);
const selfContained = Boolean(arg("self-contained"));

console.log(`building libtranscribe for ${t}`);
console.log(`  source: ${src}`);
console.log(`  install prefix: ${prefix}${selfContained ? " (self-contained)" : ""}`);

const flags = ["-DTRANSCRIBE_BUILD_SHARED=ON"];
if (process.platform === "darwin") flags.push("-DGGML_METAL=ON", "-DGGML_METAL_EMBED_LIBRARY=ON");
if (selfContained) flags.push("-DBUILD_SHARED_LIBS=OFF"); // ggml static -> one libtranscribe
if (process.env.TRANSCRIBE_CMAKE_FLAGS) flags.push(...process.env.TRANSCRIBE_CMAKE_FLAGS.split(" ").filter(Boolean));

fs.rmSync(prefix, { recursive: true, force: true });
run("cmake", ["-B", buildDir, "-S", src, "-G", "Ninja", ...flags]);
run("cmake", ["--build", buildDir, "--target", "transcribe"]);
run("cmake", ["--install", buildDir, "--prefix", prefix]);

console.log(`\ndone. The loader will find prebuilds/${t}/lib/ automatically.`);

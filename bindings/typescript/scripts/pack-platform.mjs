// Build an @transcribe-cpp/<tuple> platform package from a native lib directory
// (a `transcribe-native-<tuple>` bundle, or a `cmake --install` lib/ tree). The
// package carries the shared library + its ggml siblings + contract.json +
// third-party licenses, and an os/cpu/libc-constrained package.json so npm
// installs only the one matching the host. The API package depends on all of
// these via optionalDependencies; the loader require.resolve's the matching one.
//
//   node scripts/pack-platform.mjs --lib-dir <dir> --tuple <tuple> \
//     --version <x.y.z> --out <dir> [--header-hash <hash>] [--backends a,b]
//
// Used by the release workflow (over the CI native bundles) and locally to
// validate the clean-machine install path.

import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";

const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../..");

// npm tuple (Node platform-arch convention) -> os/cpu/libc + lib name + backends.
const TUPLES = {
  "darwin-arm64-metal": { os: "darwin", cpu: "arm64", lib: "libtranscribe.dylib", backends: ["metal", "cpu"] },
  "darwin-x64-cpu": { os: "darwin", cpu: "x64", lib: "libtranscribe.dylib", backends: ["cpu"] },
  "linux-x64-cpu-vulkan": { os: "linux", cpu: "x64", libc: "glibc", lib: "libtranscribe.so", backends: ["cpu", "vulkan"] },
  "linux-arm64-cpu-vulkan": { os: "linux", cpu: "arm64", libc: "glibc", lib: "libtranscribe.so", backends: ["cpu", "vulkan"] },
  "win32-x64-cpu-vulkan": { os: "win32", cpu: "x64", lib: "transcribe.dll", backends: ["cpu", "vulkan"] },
};

function arg(name, fallback) {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && i + 1 < process.argv.length ? process.argv[i + 1] : fallback;
}

const libDir = arg("lib-dir");
const tuple = arg("tuple");
const version = arg("version");
const outRoot = arg("out");
if (!libDir || !tuple || !version || !outRoot) {
  console.error("usage: --lib-dir <dir> --tuple <tuple> --version <x.y.z> --out <dir>");
  process.exit(2);
}
const spec = TUPLES[tuple];
if (!spec) {
  console.error(`unknown tuple ${tuple}; known: ${Object.keys(TUPLES).join(", ")}`);
  process.exit(2);
}

const pkgDir = path.join(outRoot, tuple);
fs.rmSync(pkgDir, { recursive: true, force: true });
fs.mkdirSync(pkgDir, { recursive: true });

// 1) Copy the shared libraries (transcribe + ggml siblings + symlinks).
const libExt = { darwin: ".dylib", linux: ".so", win32: ".dll" }[spec.os];
const files = [];
for (const f of fs.readdirSync(libDir)) {
  if (f.includes(libExt) || f.endsWith(".so") || /\.so\.\d/.test(f)) {
    // Dereference: npm pack drops symlinks, and the install-name deps reference
    // the .0 / plain names, so every referenced name must be a real file. (The
    // CI native bundles are already delocate/auditwheel-flattened — no symlinks
    // — so this is a no-op there; only the local cmake-install tree has them.)
    fs.cpSync(path.join(libDir, f), path.join(pkgDir, f), { recursive: true, dereference: true });
    files.push(f);
  }
}
if (!files.includes(spec.lib)) {
  console.error(`expected ${spec.lib} in ${libDir}; found: ${files.join(", ") || "(none)"}`);
  process.exit(1);
}
// The link manifest helps source/compiled consumers; ship it if present.
if (fs.existsSync(path.join(libDir, "transcribe-link.json"))) {
  fs.cpSync(path.join(libDir, "transcribe-link.json"), path.join(pkgDir, "transcribe-link.json"));
  files.push("transcribe-link.json");
}

// 2) contract.json — copy the bundle's if present, else synthesize it.
const headerHash =
  arg("header-hash") || fs.readFileSync(path.join(REPO, "include", "transcribe.abihash"), "utf8").trim();
const backends = (arg("backends") || spec.backends.join(",")).split(",");
const srcContract = path.join(libDir, "contract.json");
const contract = fs.existsSync(srcContract)
  ? JSON.parse(fs.readFileSync(srcContract, "utf8"))
  : { version, header_hash: headerHash, backends, lane: tuple };
fs.writeFileSync(path.join(pkgDir, "contract.json"), JSON.stringify(contract, null, 2) + "\n");
files.push("contract.json");

// 3) Third-party license texts (ggml at minimum) — required by §5.
fs.mkdirSync(path.join(pkgDir, "licenses"), { recursive: true });
for (const [name, rel] of [
  ["LICENSE", "LICENSE"],
  ["LICENSE.ggml", "ggml/LICENSE"],
]) {
  const p = path.join(REPO, rel);
  if (fs.existsSync(p)) fs.cpSync(p, path.join(pkgDir, "licenses", name));
}
files.push("licenses");

// 4) package.json — os/cpu/libc gate so npm installs only the matching host.
const pkg = {
  name: `@transcribe-cpp/${tuple}`,
  version,
  description: `Prebuilt transcribe.cpp native library for ${tuple}.`,
  license: "MIT",
  repository: { type: "git", url: "git+https://github.com/handy-computer/transcribe.cpp.git" },
  os: [spec.os],
  cpu: [spec.cpu],
  ...(spec.libc ? { libc: [spec.libc] } : {}),
  files: [...new Set(files)].sort(),
  // The bytes are the deliverable; nothing to run on install.
  preferUnplugged: true,
};
fs.writeFileSync(path.join(pkgDir, "package.json"), JSON.stringify(pkg, null, 2) + "\n");

console.log(`wrote ${pkg.name}@${version} -> ${pkgDir}`);
console.log(`  os=${spec.os} cpu=${spec.cpu}${spec.libc ? ` libc=${spec.libc}` : ""}`);
console.log(`  backends=${contract.backends.join(",")} header_hash=${contract.header_hash}`);
console.log(`  files: ${pkg.files.join(", ")}`);

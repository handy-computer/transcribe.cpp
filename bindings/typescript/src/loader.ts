/**
 * Resolve the native transcribe.cpp library. Resolution order (mirrors the
 * Python provider choke point in `_library.py`):
 *
 *   1. TRANSCRIBE_LIBRARY  — explicit path (developer escape hatch).
 *   2. A per-platform npm package `@transcribe-cpp/<tuple>` (the prebuilt
 *      bundle wrapping the shared `transcribe-native-<tuple>` bytes). On a host
 *      with a CUDA build, the opt-in `@transcribe-cpp/<host>-cuda` package is
 *      preferred when installed (it is a strict superset — cuda + vulkan + cpu),
 *      else the default `<host>-cpu-vulkan` / `<host>-metal` package.
 *   3. A dev build tree (build-shared/src, build/src, …) found by walking up.
 *
 * Before dlopen of a platform package, its `contract.json` is validated
 * (version base-match + header_hash equals our PUBLIC_HEADER_HASH).
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import * as g from "./_generated.js";
import { OUR_VERSION, baseVersion } from "./version.js";
import { AbiError, BackendError, TranscribeError, VersionMismatch } from "./errors.js";

export interface Resolved {
  libraryPath: string;
  /** Directory holding the library and any ggml backend modules. */
  artifactDir: string;
  /** Provider package name, or null for env/dev resolution. */
  provider: string | null;
  /**
   * Backend kinds the resolved artifact advertises (from its contract.json),
   * or empty for env/dev resolution where no contract is read. Drives the
   * CUDA-runtime preload in native.ts.
   */
  backends: string[];
}

const LIB_NAME =
  process.platform === "darwin"
    ? "libtranscribe.dylib"
    : process.platform === "win32"
      ? "transcribe.dll"
      : "libtranscribe.so";

/**
 * The platform-package tuple for this host, or null if unsupported. Uses the
 * Node platform-arch convention (matches npm os/cpu fields), e.g.
 * `darwin-arm64-metal`. The release workflow maps each shared
 * `transcribe-native-<build-tuple>` bundle onto its tuple here.
 */
function platformTuple(): string | null {
  const { platform, arch } = process;
  if (platform === "darwin") {
    if (arch === "arm64") return "darwin-arm64-metal";
    if (arch === "x64") return "darwin-x64-cpu";
  } else if (platform === "linux") {
    if (arch === "x64") return "linux-x64-cpu-vulkan";
    if (arch === "arm64") return "linux-arm64-cpu-vulkan";
  } else if (platform === "win32") {
    if (arch === "x64") return "win32-x64-cpu-vulkan";
  }
  return null;
}

/**
 * Candidate platform-package tuples for this host, most-preferred first. A
 * CUDA-capable host lists its opt-in `<host>-cuda` tuple ahead of the default:
 * the CUDA package is a strict superset (cuda + vulkan + cpu modules), so when a
 * user has installed it we load it and let ggml pick CUDA at runtime — falling
 * back to its bundled Vulkan/CPU when no driver is present. Hosts without a CUDA
 * build just list their single default tuple.
 */
function preferredTuples(): string[] {
  const base = platformTuple();
  if (!base) return [];
  const { platform, arch } = process;
  if (arch === "x64" && (platform === "linux" || platform === "win32")) {
    return [`${platform}-x64-cuda`, base];
  }
  return [base];
}

interface Contract {
  version?: string;
  header_hash?: string;
  backends?: string[];
}

/** Validate a bundle's contract.json and return it. Throws on a real mismatch. */
function validateContract(dir: string, provider: string): Contract {
  const file = path.join(dir, "contract.json");
  let contract: Contract;
  try {
    contract = JSON.parse(fs.readFileSync(file, "utf8"));
  } catch (e) {
    throw new BackendError(
      `provider ${provider} is missing a readable contract.json at ${file}: ${String(e)}`,
    );
  }
  if (contract.header_hash !== g.PUBLIC_HEADER_HASH) {
    throw new AbiError(
      `provider ${provider} was built against header hash ${contract.header_hash} ` +
        `but this binding expects ${g.PUBLIC_HEADER_HASH}. Reinstall matching packages.`,
    );
  }
  if (contract.version && baseVersion(contract.version) !== OUR_VERSION) {
    throw new VersionMismatch(
      `provider ${provider} is version ${contract.version} but this binding is ${OUR_VERSION}. ` +
        `Pre-1.0 requires an exact base-version match.`,
    );
  }
  return contract;
}

function tryPlatformPackage(): Resolved | null {
  const require = createRequire(import.meta.url);
  // Most-preferred tuple first (CUDA ahead of the default on capable hosts).
  for (const tuple of preferredTuples()) {
    const provider = `@transcribe-cpp/${tuple}`;
    let dir: string;
    try {
      dir = path.dirname(require.resolve(`${provider}/package.json`));
    } catch {
      continue; // not installed — try the next candidate, then the dev tree
    }
    // A contract mismatch is fatal, not a fall-through.
    const contract = validateContract(dir, provider);
    return {
      libraryPath: path.join(dir, LIB_NAME),
      artifactDir: dir,
      provider,
      backends: contract.backends ?? [],
    };
  }
  return null;
}

/** A library produced by the source build (`npm run build:native`). */
function tryLocalPrebuild(): Resolved | null {
  const tuple = platformTuple();
  if (!tuple) return null;
  // <package>/prebuilds/<tuple>/lib/<lib> (build-from-source installs here).
  const pkgRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
  const dir = path.join(pkgRoot, "prebuilds", tuple, "lib");
  const cand = path.join(dir, LIB_NAME);
  return fs.existsSync(cand)
    ? {
        libraryPath: cand,
        artifactDir: dir,
        provider: `source-build:${tuple}`,
        backends: [],
      }
    : null;
}

/** Walk up to the repo root, then probe known build trees. */
function tryDevTree(): Resolved | null {
  let dir = path.dirname(fileURLToPath(import.meta.url));
  let root: string | null = null;
  for (let i = 0; i < 12; i++) {
    if (
      fs.existsSync(path.join(dir, "CMakeLists.txt")) &&
      fs.existsSync(path.join(dir, "include", "transcribe.h"))
    ) {
      root = dir;
      break;
    }
    const up = path.dirname(dir);
    if (up === dir) break;
    dir = up;
  }
  if (!root) return null;
  for (const rel of ["build-shared/src", "build-shared/bin", "build/src", "build/bin"]) {
    const cand = path.join(root, rel, LIB_NAME);
    if (fs.existsSync(cand)) {
      return {
        libraryPath: cand,
        artifactDir: path.dirname(cand),
        provider: null,
        backends: [],
      };
    }
  }
  return null;
}

export function resolveLibrary(): Resolved {
  const override = process.env.TRANSCRIBE_LIBRARY;
  if (override) {
    if (!fs.existsSync(override)) {
      throw new TranscribeError(`TRANSCRIBE_LIBRARY points at a missing file: ${override}`);
    }
    return {
      libraryPath: override,
      artifactDir: path.dirname(override),
      provider: null,
      backends: [],
    };
  }
  return (
    tryPlatformPackage() ??
    tryLocalPrebuild() ??
    tryDevTree() ??
    (() => {
      throw new BackendError(
        "No transcribe.cpp native library found. Set TRANSCRIBE_LIBRARY, install a " +
          "matching @transcribe-cpp/<platform> package, run `npm run build:native` to " +
          `build from source, or build the shared library (build-shared/src/${LIB_NAME}).`,
      );
    })()
  );
}

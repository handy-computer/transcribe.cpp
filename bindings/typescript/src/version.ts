/**
 * This binding's own version, and the base-version helper the load gates use.
 *
 * `OUR_VERSION` is read from the API package's `package.json` at runtime — NOT
 * from the generated FFI macros. The generators stopped emitting
 * `TRANSCRIBE_VERSION_*` so a version-only bump no longer churns generated
 * files or the abihash (notes/releasing.md §8 P0 #1). `package.json` is always
 * present in the published tarball and sits one directory above the compiled
 * `dist/` output, so a runtime `require("../package.json")` resolves it.
 *
 * A runtime `createRequire(...)` is used rather than `import "../package.json"`:
 * tsconfig sets `rootDir: "src"` with no `resolveJsonModule`, so a static JSON
 * import would break the emit layout.
 */

import { createRequire } from "node:module";

/** Leading dotted-numeric release segment, suffix stripped: "0.0.1.post3" -> "0.0.1". */
export function baseVersion(v: string): string {
  const m = /^\d+(?:\.\d+)*/.exec(v.trim());
  return m ? m[0] : v.trim();
}

function readPackageVersion(): string {
  const require = createRequire(import.meta.url);
  const pkg = require("../package.json") as { version?: string };
  if (!pkg.version) {
    throw new Error("transcribe-cpp: package.json is missing a version field");
  }
  return pkg.version;
}

/** The base `MAJOR.MINOR.PATCH` this binding was built as. */
export const OUR_VERSION = baseVersion(readPackageVersion());

/**
 * Load-time ABI verification. TypeScript has no C compiler to check struct
 * layout at build time (unlike Rust/Swift), so — like the Python ctypes
 * binding — we verify the generated layout twice before constructing anything:
 *
 *   1. self-check: koffi's computed size/align/offsets vs the layout the C
 *      compiler captured at generation time (STRUCT_LAYOUT).
 *   2. native-agreement: koffi's size/align vs the loaded library via
 *      transcribe_abi_struct_size/_align (catches a binding/library skew).
 */

import { ABI_STRUCT_IDS, STRUCT_LAYOUT } from "./_generated.js";
import * as g from "./_generated.js";
import { AbiError } from "./errors.js";
import type { Bound } from "./ffi.js";

export function verifyLayouts({ koffi, T, F }: Bound): void {
  const mismatches: string[] = [];

  for (const [name, lo] of Object.entries(STRUCT_LAYOUT)) {
    const type = T[name];
    if (!type) continue;
    const size = koffi.sizeof(type);
    const align = koffi.alignof(type);
    if (size !== lo.size) mismatches.push(`${name}: size ${size} != captured ${lo.size}`);
    if (align !== lo.align) mismatches.push(`${name}: align ${align} != captured ${lo.align}`);
    for (const [field, off] of Object.entries(lo.offsets)) {
      const actual = koffi.offsetof(type, field);
      if (actual !== off) {
        mismatches.push(`${name}.${field}: offset ${actual} != captured ${off}`);
      }
    }
  }

  for (const [name, id] of Object.entries(ABI_STRUCT_IDS)) {
    const type = T[name];
    if (!type) continue;
    const nativeSize = Number(F.abiStructSize(id));
    const nativeAlign = Number(F.abiStructAlign(id));
    if (nativeSize === 0) {
      mismatches.push(`${name}: native size 0 (loaded library is older than this binding)`);
      continue;
    }
    const size = koffi.sizeof(type);
    const align = koffi.alignof(type);
    if (size !== nativeSize) mismatches.push(`${name}: size ${size} != native ${nativeSize}`);
    if (align !== nativeAlign) mismatches.push(`${name}: align ${align} != native ${nativeAlign}`);
  }

  if (mismatches.length) {
    throw new AbiError(
      "transcribe_cpp ABI layout check failed — this is a version/build skew between the " +
        "binding and the native library:\n  " +
        mismatches.join("\n  "),
      g.TRANSCRIBE_ERR_BAD_STRUCT_SIZE,
    );
  }
}

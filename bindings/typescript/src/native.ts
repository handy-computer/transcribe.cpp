/**
 * Native bootstrap: resolve -> load -> bind -> verify ABI -> version-gate ->
 * init backends. The single choke point both the public API and tests go
 * through. Lazy and memoized: nothing loads until the first call.
 */

import { resolveLibrary } from "./loader.js";
import { abortProto, bindLibrary, type Bound, logProto } from "./ffi.js";
import { verifyLayouts } from "./abi.js";
import { BackendError, VersionMismatch } from "./errors.js";
import { OUR_VERSION, baseVersion } from "./version.js";
import * as g from "./_generated.js";

export interface Native extends Bound {
  libraryPath: string;
  provider: string | null;
  abortProto: any;
  logProto: any;
}

let cached: Native | null = null;

// ---- log routing -----------------------------------------------------------

export type LogHandler = (level: number, message: string) => void;
let logHandler: LogHandler | null = null;
let logRegistered: any = null;

function ensureLogCallback(bound: Bound): void {
  if (logRegistered) return;
  const thunk = (level: number, msg: string) => {
    const h = logHandler;
    if (!h) return;
    try {
      h(level, msg ?? "");
    } catch {
      /* a logging handler must never propagate into native code */
    }
  };
  logRegistered = bound.koffi.register(thunk, bound.koffi.pointer(logProto()));
  bound.F.logSet(logRegistered, null);
}

export function native(): Native {
  if (cached) return cached;

  const resolved = resolveLibrary();
  const bound = bindLibrary(resolved.libraryPath);
  verifyLayouts(bound);

  const nativeVersion = bound.F.version();
  if (baseVersion(nativeVersion) !== OUR_VERSION) {
    throw new VersionMismatch(
      `native library is version ${nativeVersion} but this binding is ${OUR_VERSION} ` +
        `(pre-1.0 requires an exact base-version match)`,
    );
  }

  // transcribe_log_set is only supported as a startup-time install. Register a
  // stable JS dispatch callback once, before backend init can create worker
  // threads; setLogHandler later only swaps the JS target.
  ensureLogCallback(bound);

  // Register backend modules package-local. On a compiled-in build this is a
  // near no-op; on a dynamic-backend bundle it scans the artifact dir.
  let st = bound.F.initBackends(resolved.artifactDir);
  if (st !== g.TRANSCRIBE_OK) st = bound.F.initBackendsDefault();
  if (st !== g.TRANSCRIBE_OK) {
    throw new BackendError(
      `transcribe_init_backends found no usable compute device in ${resolved.artifactDir}: ` +
        `${bound.F.statusString(st)} (status ${st})`,
    );
  }

  cached = {
    ...bound,
    libraryPath: resolved.libraryPath,
    provider: resolved.provider,
    abortProto: abortProto(),
    logProto: logProto(),
  };
  return cached;
}

/**
 * Route native (and ggml) diagnostics to `handler`, or pass null to disable.
 * The callback may fire from ggml worker threads; koffi marshals it to the
 * event-loop thread (proven by the M2 spike), so the handler runs on the main
 * thread. Exceptions thrown by the handler are swallowed (never re-enter C).
 *
 * The native callback is installed once during lazy bootstrap, before backend
 * initialization. Later calls only swap this JS handler; they never call
 * transcribe_log_set again.
 */
export function setLogHandler(handler: LogHandler | null): void {
  logHandler = handler;
}

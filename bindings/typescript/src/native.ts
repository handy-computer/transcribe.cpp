/**
 * Native bootstrap: resolve -> load -> bind -> verify ABI -> version-gate ->
 * init backends. The single choke point both the public API and tests go
 * through. Lazy and memoized: nothing loads until the first call.
 */

import { resolveLibrary } from "./loader.js";
import { abortProto, bindLibrary, type Bound, logProto } from "./ffi.js";
import { verifyLayouts } from "./abi.js";
import { BackendError, VersionMismatch } from "./errors.js";
import * as g from "./_generated.js";

export interface Native extends Bound {
  libraryPath: string;
  provider: string | null;
  abortProto: any;
  logProto: any;
}

let cached: Native | null = null;

const OUR_VERSION = `${g.TRANSCRIBE_VERSION_MAJOR}.${g.TRANSCRIBE_VERSION_MINOR}.${g.TRANSCRIBE_VERSION_PATCH}`;
const baseVersion = (v: string): string => (/^\d+(?:\.\d+)*/.exec(v.trim())?.[0] ?? v.trim());

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

  // Sever the native->JS log path at process exit, before koffi tears down the
  // callback trampoline. A dlopen'd backend module (the Linux/Windows
  // cpu-vulkan builds; macOS compiles metal in and never dlopens) can emit a
  // diagnostic during teardown that routes through the transcribe sink into the
  // freed trampoline and crashes. transcribe_shutdown() disables the sink and
  // detaches ggml's logger; if the loaded library predates that symbol, falling
  // back to logSet(null) still clears the trampoline from the sink. We do NOT
  // koffi.unregister: freeing the trampoline while a worker thread might still
  // hold it is a use-after-free — leaving it registered keeps any stale call
  // landing on live memory. Registered here, before initBackends, so a failed
  // backend init still detaches cleanly on exit.
  process.once("exit", () => {
    try {
      if (bound.F.shutdown) bound.F.shutdown();
      else bound.F.logSet(null, null);
    } catch {
      /* exiting; nothing to recover */
    }
  });
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

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

// Install the native dispatch trampoline. Called lazily from setLogHandler the
// first time a host opts into logging — deliberately NOT at bootstrap. The koffi
// callback trampoline must not coexist with ggml worker threads in a process
// that later exits: on the dlopen'd backend builds (Linux/Windows cpu-vulkan;
// macOS metal is compiled in and never dlopens) their teardown ordering
// segfaults the process at exit. A process that never sets a log handler never
// carries the trampoline, so loading and running a model stays crash-free. The
// host that does want logs accepts that risk here and gets the best-effort exit
// detach below as mitigation.
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

  // Best-effort mitigation for the opted-in case: sever the native->JS log path
  // at process exit, before koffi tears down the trampoline. transcribe_shutdown
  // disables the transcribe sink and detaches ggml's logger; a library that
  // predates the symbol falls back to logSet(null), which still clears the
  // trampoline from the sink. We do NOT koffi.unregister — freeing the
  // trampoline while a worker thread might still hold it is a use-after-free.
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

  // The log callback is installed lazily by setLogHandler, NOT here: a process
  // that never logs must not carry the koffi trampoline (see ensureLogCallback).

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
 * The native dispatch trampoline is installed lazily here, the first time a
 * non-null handler is set, and never removed; later calls only swap the JS
 * target. Installing it is opt-in by design — a process that never logs must
 * not carry the trampoline (see ensureLogCallback for why). Disabling without
 * ever having logged silences native diagnostics without installing one.
 */
export function setLogHandler(handler: LogHandler | null): void {
  logHandler = handler;
  if (handler) {
    ensureLogCallback(native());
  } else if (!logRegistered) {
    native().F.logSet(null, null);
  }
  // handler === null with a trampoline already installed: the thunk drops.
}

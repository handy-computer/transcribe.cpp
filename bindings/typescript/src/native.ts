/**
 * Native bootstrap, in two phases:
 *
 *   1. `loadLibrary()` — resolve -> (CUDA runtime preload) -> dlopen -> bind ->
 *      verify ABI -> version-gate -> install the log callback. Synchronous and
 *      GPU-free: it never touches the ggml backend registry. Memoized on
 *      success; a failure is NOT cached (nothing native was mutated, so the
 *      next call simply retries).
 *
 *   2. Backend initialization — `transcribe_init_backends`, which builds the
 *      ggml backend registry and with it every backend's device bootstrap
 *      (on Metal: compiling the embedded shader library, which can take
 *      seconds). `initialize()` runs it on a koffi async worker so the event
 *      loop keeps running; concurrent callers share one attempt. A non-OK
 *      status is cached as the terminal `failed` state, because the C library
 *      documents it as not retryable in this process.
 *
 * Synchronous entry points that touch the registry go through `nativeReady()`:
 *
 *   - ready         -> return the binding.
 *   - failed        -> rethrow the cached bootstrap error.
 *   - initializing  -> throw BackendInitializing. It must not wait: the init
 *                      worker may be blocked delivering a log message to this
 *                      (main) thread, so waiting on it deadlocks.
 *   - uninitialized -> run backend init inline on this thread (the pre-async
 *                      behavior, kept for compatibility; this is the call that
 *                      blocks a UI thread, so UI hosts should `await
 *                      initialize()` first).
 */

import { resolveLibrary } from "./loader.js";
import { preloadCudaRuntime } from "./cuda.js";
import { abortProto, bindLibrary, type Bound, logProto } from "./ffi.js";
import { verifyLayouts } from "./abi.js";
import { BackendError, BackendInitializing, TranscribeError, VersionMismatch } from "./errors.js";
import { OUR_VERSION, baseVersion } from "./version.js";
import * as g from "./_generated.js";

export interface Native extends Bound {
  libraryPath: string;
  provider: string | null;
  /** Directory scanned for backend modules by `transcribe_init_backends`. */
  artifactDir: string;
  abortProto: any;
  logProto: any;
}

/** Lifecycle of the process-wide backend initialization. */
export type BackendState = "uninitialized" | "initializing" | "ready" | "failed";

let lib: Native | null = null;
let state: BackendState = "uninitialized";
let initPromise: Promise<void> | null = null; // set while an async init is in flight
let initError: TranscribeError | null = null; // the cached bootstrap failure

// ---- log routing -----------------------------------------------------------

export type LogHandler = (level: number, message: string) => void;
let logHandler: LogHandler | null = null;
let logRegistered: any = null;

// Registered once and never unregistered or re-installed: native code (including
// an in-flight backend init on a worker) may hold the pointer at any time. Only
// the JS target (`logHandler`) ever changes.
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

// ---- phase 1: library -------------------------------------------------------

/** Load and verify the native library without initializing any backend. */
export function loadLibrary(): Native {
  if (lib) return lib;

  const resolved = resolveLibrary();
  // Bring-your-own CUDA runtime: if the selected provider advertises cuda,
  // preload cudart/cublas (the TS twin of the Python cu12 provider's prepare()
  // hook) BEFORE any dlopen, so the ggml-cuda module's DT_NEEDED sonames resolve
  // when backend init loads it. A no-op without cuda or the runtime dir. This
  // is synchronous (koffi.load has no async form) and can be slow for a large
  // cublas; it only runs for the opt-in CUDA package with the env var set.
  if (resolved.backends.includes("cuda")) preloadCudaRuntime();
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

  lib = {
    ...bound,
    libraryPath: resolved.libraryPath,
    provider: resolved.provider,
    artifactDir: resolved.artifactDir,
    abortProto: abortProto(),
    logProto: logProto(),
  };
  return lib;
}

// ---- phase 2: backends ------------------------------------------------------

function callAsync<T>(fn: any, ...args: any[]): Promise<T> {
  return new Promise((resolve, reject) =>
    fn.async(...args, (err: Error | null, res: T) => (err ? reject(err) : resolve(res))),
  );
}

function initFailure(n: Native, st: number): BackendError {
  return new BackendError(
    `transcribe_init_backends found no usable compute device in ${n.artifactDir}: ` +
      `${n.F.statusString(st)} (status ${st})`,
    st,
  );
}

/**
 * Register backend modules package-local, falling back to the library's own
 * directory. Then query the device count: that constructs the ggml backend
 * registry (and so runs every compiled-in backend's device bootstrap, e.g. the
 * Metal shader compile) even on a path that skipped the module scan, such as
 * `transcribe_init_backends_default` on a compiled-in build. After this, no
 * registry query can trigger first-time device bootstrap on the caller.
 */
async function initAsync(n: Native): Promise<number> {
  let st = await callAsync<number>(n.F.initBackends, n.artifactDir);
  if (st !== g.TRANSCRIBE_OK) st = await callAsync<number>(n.F.initBackendsDefault);
  if (st === g.TRANSCRIBE_OK) await callAsync<number>(n.F.deviceCount);
  return st;
}

function initSync(n: Native): number {
  let st = n.F.initBackends(n.artifactDir);
  if (st !== g.TRANSCRIBE_OK) st = n.F.initBackendsDefault();
  if (st === g.TRANSCRIBE_OK) n.F.deviceCount();
  return st;
}

/** The current backend initialization state. Never loads or initializes anything. */
export function backendState(): BackendState {
  return state;
}

/**
 * Load the native library and initialize compute backends without blocking the
 * event loop. Idempotent: concurrent and repeated calls share one attempt and
 * resolve together; once ready it resolves immediately. A backend bootstrap
 * failure is permanent for the process and every later call rejects with it.
 * A library-load failure (missing provider, version mismatch) is not cached.
 */
export function initialize(): Promise<void> {
  if (state === "ready") return Promise.resolve();
  if (state === "failed") return Promise.reject(initError);
  if (initPromise) return initPromise;
  if (state === "initializing") {
    // Re-entered from a log handler during a synchronous inline init.
    return Promise.reject(inProgress());
  }

  let n: Native;
  try {
    n = loadLibrary();
  } catch (e) {
    return Promise.reject(e);
  }

  state = "initializing";
  initPromise = initAsync(n).then(
    (st) => {
      initPromise = null;
      if (st === g.TRANSCRIBE_OK) {
        state = "ready";
        return;
      }
      initError = initFailure(n, st);
      state = "failed";
      throw initError;
    },
    (e) => {
      // koffi refused to dispatch the call (e.g. its async queue is full), so
      // native init never ran: not a bootstrap failure, the next call retries.
      initPromise = null;
      state = "uninitialized";
      throw e;
    },
  );
  return initPromise;
}

/** Initialize (or join the in-flight init) and return the binding. */
export async function nativeAsync(): Promise<Native> {
  await initialize();
  return lib as Native;
}

function inProgress(): BackendInitializing {
  return new BackendInitializing(
    "transcribe.cpp backend initialization is in progress; await initialize() " +
      "(or use getAvailableBackendsAsync() / backendAvailableAsync()) instead of " +
      "calling a synchronous backend query concurrently with it",
  );
}

/** The binding with backends ready, for synchronous entry points. See module docs. */
export function nativeReady(): Native {
  if (state === "ready") return lib as Native;
  if (state === "failed") throw initError;
  if (state === "initializing") throw inProgress();

  const n = loadLibrary();
  state = "initializing"; // a re-entrant call from a log handler fails fast
  let st: number;
  try {
    st = initSync(n);
  } catch (e) {
    state = "uninitialized";
    throw e;
  }
  if (st !== g.TRANSCRIBE_OK) {
    initError = initFailure(n, st);
    state = "failed";
    throw initError;
  }
  state = "ready";
  return n;
}

/**
 * Route native (and ggml) diagnostics to `handler`, or pass null to disable.
 * The callback may fire from ggml worker threads; koffi marshals it to the
 * event-loop thread, so the handler runs on the main thread. Exceptions thrown
 * by the handler are swallowed (never re-enter C).
 *
 * The native callback is installed once when the library loads, before backend
 * initialization. Later calls only swap this JS handler (safe at any time,
 * including during `initialize()`); they never call transcribe_log_set again.
 */
export function setLogHandler(handler: LogHandler | null): void {
  logHandler = handler;
}

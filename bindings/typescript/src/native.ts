/**
 * Native bootstrap, in two phases:
 *
 *   1. `loadLibrary()`: resolve -> (CUDA runtime preload) -> dlopen -> bind ->
 *      verify ABI -> version-gate -> install the log callback. Synchronous but
 *      GPU-free. Memoized on success; a failure is not cached.
 *
 *   2. Backend initialization: `transcribe_init_backends` builds the ggml
 *      backend registry, running each device's bootstrap (on Metal, the shader
 *      library compile, which can take seconds). `initialize()` runs it on a
 *      koffi async worker; concurrent callers share one attempt. A non-OK
 *      status is cached as `failed`: the C library documents it as not
 *      retryable in this process.
 *
 * Synchronous entry points that touch the registry go through `nativeReady()`.
 * It initializes inline when uninitialized (blocking the caller, the pre-async
 * behavior) and throws BackendInitializing while an async init is in flight.
 * It must not wait for that init: the worker may be blocked delivering a log
 * message to this (main) thread, so waiting would deadlock.
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

export function callAsync<T = number>(fn: any, ...args: any[]): Promise<T> {
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
 * The device count after init forces registry construction (and so every device
 * bootstrap) even on paths that skip the module scan. Zero devices is a failure
 * even after an OK init: the compiled-in default path doesn't check the count,
 * and `transcribe_device_count` reports 0 if registry construction threw.
 */
function requireDevices(st: number, count: number): number {
  return st === g.TRANSCRIBE_OK && count <= 0 ? g.TRANSCRIBE_ERR_BACKEND : st;
}

async function initAsync(n: Native): Promise<number> {
  let st = await callAsync<number>(n.F.initBackends, n.artifactDir);
  if (st !== g.TRANSCRIBE_OK) st = await callAsync<number>(n.F.initBackendsDefault);
  if (st !== g.TRANSCRIBE_OK) return st;
  return requireDevices(st, await callAsync<number>(n.F.deviceCount));
}

function initSync(n: Native): number {
  let st = n.F.initBackends(n.artifactDir);
  if (st !== g.TRANSCRIBE_OK) st = n.F.initBackendsDefault();
  if (st !== g.TRANSCRIBE_OK) return st;
  return requireDevices(st, n.F.deviceCount());
}

/** The current backend initialization state. Never loads or initializes anything. */
export function backendState(): BackendState {
  return state;
}

/**
 * Load the native library and initialize compute backends off the event loop.
 * Idempotent; concurrent calls share one attempt. A backend init failure is
 * permanent for the process; a library-load failure is not cached.
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
      // koffi failed to dispatch a call: not a native bootstrap status, so the
      // next call retries (transcribe_init_backends is idempotent).
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
 * initialization. Later calls only swap this JS handler; they never call
 * transcribe_log_set again.
 */
export function setLogHandler(handler: LogHandler | null): void {
  logHandler = handler;
}

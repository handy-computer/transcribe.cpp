/**
 * The koffi engine adapter: load the library and hand-bind the C functions
 * with explicit in/out/inout direction (which a generic generator can't infer).
 * Struct *layouts* come from the generated `defineTypes`; the drift gate covers
 * signature changes via the pinned PUBLIC_HEADER_HASH.
 *
 * This is the one runtime-specific seam. A Bun/Deno adapter would re-implement
 * this file against the same generated types and the same public shape.
 */

import koffi from "koffi";
import { defineTypes } from "./_generated.js";

export interface Bound {
  koffi: typeof koffi;
  lib: ReturnType<typeof koffi.load>;
  T: Record<string, any>;
  F: Record<string, any>;
}

export function bindLibrary(libraryPath: string): Bound {
  const lib = koffi.load(libraryPath);
  const T = defineTypes(koffi);

  const inp = (t: any) => koffi.pointer(t); // const X * (marshalled in)
  const outp = (t: any) => koffi.out(koffi.pointer(t)); // X * pure out (e.g. *_init)
  // Accessors read struct_size on input then fill the rest: copy in AND back.
  const iop = (t: any) => koffi.inout(koffi.pointer(t));
  const handleOut = koffi.out(koffi.pointer("void *")); // X ** out

  const F: Record<string, any> = {
    // identity / diagnostics
    version: lib.func("transcribe_version", "str", []),
    versionCommit: lib.func("transcribe_version_commit", "str", []),
    statusString: lib.func("transcribe_status_string", "str", ["int"]),
    abiStructSize: lib.func("transcribe_abi_struct_size", "uint64", ["int"]),
    abiStructAlign: lib.func("transcribe_abi_struct_align", "uint64", ["int"]),

    // backends
    initBackends: lib.func("transcribe_init_backends", "int", ["str"]),
    initBackendsDefault: lib.func("transcribe_init_backends_default", "int", []),
    backendDeviceCount: lib.func("transcribe_backend_device_count", "int", []),
    backendDeviceInit: lib.func("transcribe_backend_device_init", "void", [
      outp(T.transcribe_backend_device),
    ]),
    getBackendDevice: lib.func("transcribe_get_backend_device", "int", [
      "int",
      iop(T.transcribe_backend_device),
    ]),
    backendAvailable: lib.func("transcribe_backend_available", "bool", ["int"]),

    // logging (callback pointer comes from koffi.register -> passed as void*)
    logSet: lib.func("transcribe_log_set", "void", ["void *", "void *"]),

    // model
    modelLoadParamsInit: lib.func("transcribe_model_load_params_init", "void", [
      outp(T.transcribe_model_load_params),
    ]),
    modelLoadFile: lib.func("transcribe_model_load_file", "int", [
      "str",
      inp(T.transcribe_model_load_params),
      handleOut,
    ]),
    modelFree: lib.func("transcribe_model_free", "void", ["void *"]),
    modelArch: lib.func("transcribe_model_arch_string", "str", ["void *"]),
    modelVariant: lib.func("transcribe_model_variant_string", "str", ["void *"]),
    modelBackend: lib.func("transcribe_model_backend", "str", ["void *"]),
    modelSupports: lib.func("transcribe_model_supports", "bool", ["void *", "int"]),
    tokenize: lib.func("transcribe_tokenize", "int", ["void *", "str", "int32_t *", "size_t"]),
    capabilitiesInit: lib.func("transcribe_capabilities_init", "void", [
      outp(T.transcribe_capabilities),
    ]),
    modelGetCapabilities: lib.func("transcribe_model_get_capabilities", "int", [
      "void *",
      iop(T.transcribe_capabilities),
    ]),

    // session
    sessionParamsInit: lib.func("transcribe_session_params_init", "void", [
      outp(T.transcribe_session_params),
    ]),
    sessionInit: lib.func("transcribe_session_init", "int", [
      "void *",
      inp(T.transcribe_session_params),
      handleOut,
    ]),
    sessionFree: lib.func("transcribe_session_free", "void", ["void *"]),
    sessionLimitsInit: lib.func("transcribe_session_limits_init", "void", [
      outp(T.transcribe_session_limits),
    ]),
    sessionGetLimits: lib.func("transcribe_session_get_limits", "int", [
      "void *",
      iop(T.transcribe_session_limits),
    ]),
    setAbortCallback: lib.func("transcribe_set_abort_callback", "void", [
      "void *",
      "void *",
      "void *",
    ]),
    wasAborted: lib.func("transcribe_was_aborted", "bool", ["void *"]),
    wasTruncated: lib.func("transcribe_was_truncated", "bool", ["void *"]),

    // run (offline) + result accessors
    runParamsInit: lib.func("transcribe_run_params_init", "void", [
      outp(T.transcribe_run_params),
    ]),
    run: lib.func("transcribe_run", "int", [
      "void *",
      inp("float"),
      "int",
      inp(T.transcribe_run_params),
    ]),
    fullText: lib.func("transcribe_full_text", "str", ["void *"]),
    detectedLanguage: lib.func("transcribe_detected_language", "str", ["void *"]),
    returnedTimestampKind: lib.func("transcribe_returned_timestamp_kind", "int", ["void *"]),
    nSegments: lib.func("transcribe_n_segments", "int", ["void *"]),
    nWords: lib.func("transcribe_n_words", "int", ["void *"]),
    nTokens: lib.func("transcribe_n_tokens", "int", ["void *"]),
    segmentInit: lib.func("transcribe_segment_init", "void", [outp(T.transcribe_segment)]),
    wordInit: lib.func("transcribe_word_init", "void", [outp(T.transcribe_word)]),
    tokenInit: lib.func("transcribe_token_init", "void", [outp(T.transcribe_token)]),
    getSegment: lib.func("transcribe_get_segment", "int", [
      "void *",
      "int",
      iop(T.transcribe_segment),
    ]),
    getWord: lib.func("transcribe_get_word", "int", ["void *", "int", iop(T.transcribe_word)]),
    getToken: lib.func("transcribe_get_token", "int", ["void *", "int", iop(T.transcribe_token)]),
    timingsInit: lib.func("transcribe_timings_init", "void", [outp(T.transcribe_timings)]),
    getTimings: lib.func("transcribe_get_timings", "int", ["void *", iop(T.transcribe_timings)]),

    // family extensions
    modelAcceptsExtKind: lib.func("transcribe_model_accepts_ext_kind", "bool", [
      "void *",
      "int",
      "uint32",
    ]),
    whisperRunExtInit: lib.func("transcribe_whisper_run_ext_init", "void", [
      outp(T.transcribe_whisper_run_ext),
    ]),
    moonshineStreamingStreamExtInit: lib.func(
      "transcribe_moonshine_streaming_stream_ext_init",
      "void",
      [outp(T.transcribe_moonshine_streaming_stream_ext)],
    ),
    parakeetStreamExtInit: lib.func("transcribe_parakeet_stream_ext_init", "void", [
      outp(T.transcribe_parakeet_stream_ext),
    ]),
    parakeetBufferedStreamExtInit: lib.func("transcribe_parakeet_buffered_stream_ext_init", "void", [
      outp(T.transcribe_parakeet_buffered_stream_ext),
    ]),
    voxtralRealtimeStreamExtInit: lib.func("transcribe_voxtral_realtime_stream_ext_init", "void", [
      outp(T.transcribe_voxtral_realtime_stream_ext),
    ]),

    // batch (offline)
    runBatch: lib.func("transcribe_run_batch", "int", [
      "void *",
      "float **",
      "int *",
      "int",
      inp(T.transcribe_run_params),
    ]),
    batchNResults: lib.func("transcribe_batch_n_results", "int", ["void *"]),
    batchStatus: lib.func("transcribe_batch_status", "int", ["void *", "int"]),
    batchNSegments: lib.func("transcribe_batch_n_segments", "int", ["void *", "int"]),
    batchGetSegment: lib.func("transcribe_batch_get_segment", "int", [
      "void *",
      "int",
      "int",
      iop(T.transcribe_segment),
    ]),
    batchNWords: lib.func("transcribe_batch_n_words", "int", ["void *", "int"]),
    batchGetWord: lib.func("transcribe_batch_get_word", "int", [
      "void *",
      "int",
      "int",
      iop(T.transcribe_word),
    ]),
    batchNTokens: lib.func("transcribe_batch_n_tokens", "int", ["void *", "int"]),
    batchGetToken: lib.func("transcribe_batch_get_token", "int", [
      "void *",
      "int",
      "int",
      iop(T.transcribe_token),
    ]),
    batchFullText: lib.func("transcribe_batch_full_text", "str", ["void *", "int"]),
    batchDetectedLanguage: lib.func("transcribe_batch_detected_language", "str", ["void *", "int"]),
    batchReturnedTimestampKind: lib.func("transcribe_batch_returned_timestamp_kind", "int", [
      "void *",
      "int",
    ]),
    batchGetTimings: lib.func("transcribe_batch_get_timings", "int", [
      "void *",
      "int",
      iop(T.transcribe_timings),
    ]),

    // streaming
    streamParamsInit: lib.func("transcribe_stream_params_init", "void", [
      outp(T.transcribe_stream_params),
    ]),
    streamUpdateInit: lib.func("transcribe_stream_update_init", "void", [
      outp(T.transcribe_stream_update),
    ]),
    streamTextInit: lib.func("transcribe_stream_text_init", "void", [outp(T.transcribe_stream_text)]),
    streamBegin: lib.func("transcribe_stream_begin", "int", [
      "void *",
      inp(T.transcribe_run_params),
      inp(T.transcribe_stream_params),
    ]),
    streamFeed: lib.func("transcribe_stream_feed", "int", [
      "void *",
      inp("float"),
      "int",
      iop(T.transcribe_stream_update),
    ]),
    streamFinalize: lib.func("transcribe_stream_finalize", "int", [
      "void *",
      iop(T.transcribe_stream_update),
    ]),
    streamGetText: lib.func("transcribe_stream_get_text", "int", [
      "void *",
      iop(T.transcribe_stream_text),
    ]),
    streamReset: lib.func("transcribe_stream_reset", "void", ["void *"]),
    streamGetState: lib.func("transcribe_stream_get_state", "int", ["void *"]),
    streamRevision: lib.func("transcribe_stream_revision", "int", ["void *"]),
    streamLastStatus: lib.func("transcribe_stream_last_status", "int", ["void *"]),
  };

  return { koffi, lib, T, F };
}

let abortProtoType: any = null;
let logProtoType: any = null;

/** koffi proto for the abort callback: `bool (void *userdata)`. */
export function abortProto(): any {
  return (abortProtoType ??= koffi.proto("bool TranscribeAbortCb(void *udata)"));
}

/** koffi proto for the log callback: `void (int level, const char *msg, void *ud)`. */
export function logProto(): any {
  return (logProtoType ??= koffi.proto("void TranscribeLogCb(int level, const char *msg, void *udata)"));
}

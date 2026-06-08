// transcribe-log.h - internal printf-style logging facility.
//
// Routes a formatted message through the process-global log callback
// installed via transcribe_log_set(), falling back to stderr when no
// callback is installed (so dev / CLI builds still see the diagnostic).
//
// This is the supported way for library internals — including per-family
// run() drivers in src/arch/<family>/ — to surface diagnostics. Family
// code must NOT call std::fprintf(stderr, ...) directly for user-facing
// conditions: a consumer that installed a log sink would never see those
// messages. In particular the input-length contract (input-too-long
// rejection, mid-decode truncation, soft-window degradation; see
// docs/input-limits.md) is surfaced through here so it reaches the
// caller's sink at the right level.
//
// The implementation lives in transcribe.cpp alongside the callback
// globals it reads.

#pragma once

#include "transcribe.h"  // transcribe_log_level

namespace transcribe {

// Emit a printf-style formatted message at `level`. The format/args are
// rendered into a bounded stack buffer (oversize messages are truncated,
// never heap-allocated). Routed through the installed log callback; if
// none is installed the rendered line is written to stderr with a
// trailing newline, matching transcribe_print_timings' fallback.
void log_msg(transcribe_log_level level, const char * fmt, ...)
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 2, 3)))
#endif
    ;

} // namespace transcribe

// transcribe-api-guard.h - exception guards for the public C ABI entry
// points (bad_alloc -> OOM, other exceptions -> BACKEND, teardown never
// throws). MSVC builds use /EHs so exceptions thrown by ggml C-ABI frames
// remain catchable here. Not part of the public API.

#pragma once

#include "transcribe-log.h"
#include "transcribe.h"

#include <exception>
#include <new>

namespace transcribe {

template <typename Fn> transcribe_status api_guard_status(const char * fn_name, Fn && body) {
    try {
        return body();
    } catch (const std::bad_alloc &) {
        transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: out of memory", fn_name);
        return TRANSCRIBE_ERR_OOM;
    } catch (const std::exception & e) {
        transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: caught backend exception: %s", fn_name, e.what());
        return TRANSCRIBE_ERR_BACKEND;
    } catch (...) {
        transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: caught unknown backend exception", fn_name);
        return TRANSCRIBE_ERR_BACKEND;
    }
}

// Value-returning variant: `on_throw` is the entry point's documented
// hard-failure value (0 devices, false, INT_MIN, ...).
template <typename T, typename Fn> T api_guard_value(const char * fn_name, T on_throw, Fn && body) {
    try {
        return body();
    } catch (const std::exception & e) {
        transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: caught backend exception: %s", fn_name, e.what());
        return on_throw;
    } catch (...) {
        transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: caught unknown backend exception", fn_name);
        return on_throw;
    }
}

// Teardown variant: log and swallow. Frees must never fail.
template <typename Fn> void api_guard_void(const char * fn_name, Fn && body) {
    try {
        body();
    } catch (const std::exception & e) {
        transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_WARN, "%s: caught backend exception during teardown: %s", fn_name,
                            e.what());
    } catch (...) {
        transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_WARN, "%s: caught unknown backend exception during teardown", fn_name);
    }
}

}  // namespace transcribe

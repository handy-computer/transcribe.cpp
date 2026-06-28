// log_unit.cpp - white-box tests for the process-global log sink.
//
// Pins the three-state contract of transcribe_log_set:
//
//   never configured   -> log_msg falls back to stderr (not directly
//                         assertable here; covered by the callback states)
//   callback installed -> library messages AND ggml diagnostics (via the
//                         bridge transcribe_log_set installs with
//                         ggml_log_set) reach the callback
//   explicitly NULL    -> everything is dropped, including ggml messages
//
// and the ggml bridge specifics: level MAPPING (ggml's DEBUG/INFO/WARN/
// ERROR numbering differs from transcribe_log_level — a raw reinterpret
// would scramble severities) and trailing-newline normalization (ggml
// messages embed "\n"; transcribe messages do not).
//
// The log sink is process-global state; this binary owns it exclusively,
// and each test installs the state it needs.

#include "ggml.h"
#include "transcribe-log.h"
#include "transcribe.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace {

int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

// Recording sink. userdata carries the vector so the tests also pin that
// userdata is delivered alongside the callback.
using Record = std::vector<std::pair<transcribe_log_level, std::string>>;

void recording_cb(transcribe_log_level level, const char * msg, void * userdata) {
    auto * rec = static_cast<Record *>(userdata);
    rec->emplace_back(level, msg != nullptr ? msg : "<null>");
}

void test_callback_receives_log_msg() {
    Record rec;
    transcribe_log_set(recording_cb, &rec);

    transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_WARN, "answer=%d", 42);

    CHECK(rec.size() == 1);
    CHECK(rec[0].first == TRANSCRIBE_LOG_LEVEL_WARN);
    CHECK(rec[0].second == "answer=42");
}

void test_null_disables_then_reinstall_restores() {
    Record rec;
    transcribe_log_set(recording_cb, &rec);
    transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_INFO, "before");
    CHECK(rec.size() == 1);

    // Explicit NULL = silence: the message must not reach the previous
    // callback (and per the public contract it goes nowhere at all).
    transcribe_log_set(nullptr, nullptr);
    transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "dropped");
    CHECK(rec.size() == 1);

    // Reinstalling brings messages back.
    transcribe_log_set(recording_cb, &rec);
    transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_INFO, "after");
    CHECK(rec.size() == 2);
    CHECK(rec[1].second == "after");
}

void test_ggml_bridge_installed_and_maps_levels() {
    Record rec;
    transcribe_log_set(recording_cb, &rec);

    // transcribe_log_set must have routed ggml's logger to its bridge.
    ggml_log_callback bridge          = nullptr;
    void *            bridge_userdata = nullptr;
    ggml_log_get(&bridge, &bridge_userdata);
    CHECK(bridge != nullptr);
    if (bridge == nullptr) {
        return;  // cannot exercise the mapping without the bridge
    }

    // Drive the bridge exactly as ggml would. The numeric values of the
    // two enums DIFFER (ggml DEBUG=1 INFO=2 WARN=3 ERROR=4 vs transcribe
    // INFO=1 WARN=2 ERROR=3 DEBUG=4): each ggml level must arrive as the
    // SAME-MEANING transcribe level, not the same number.
    bridge(GGML_LOG_LEVEL_DEBUG, "d\n", bridge_userdata);
    bridge(GGML_LOG_LEVEL_INFO, "i\n", bridge_userdata);
    bridge(GGML_LOG_LEVEL_WARN, "w\n", bridge_userdata);
    bridge(GGML_LOG_LEVEL_ERROR, "e\n", bridge_userdata);
    bridge(GGML_LOG_LEVEL_CONT, ".", bridge_userdata);

    CHECK(rec.size() == 5);
    if (rec.size() == 5) {
        CHECK(rec[0].first == TRANSCRIBE_LOG_LEVEL_DEBUG);
        CHECK(rec[1].first == TRANSCRIBE_LOG_LEVEL_INFO);
        CHECK(rec[2].first == TRANSCRIBE_LOG_LEVEL_WARN);
        CHECK(rec[3].first == TRANSCRIBE_LOG_LEVEL_ERROR);
        CHECK(rec[4].first == TRANSCRIBE_LOG_LEVEL_CONT);
        // Exactly one trailing newline is stripped (transcribe messages
        // carry none); CONT fragments pass through unmodified.
        CHECK(rec[0].second == "d");
        CHECK(rec[1].second == "i");
        CHECK(rec[4].second == ".");
    }
}

void test_init_backends_emits_device_summary() {
    // Release ggml silences its per-module load diagnostics, so the
    // library's own post-scan summary is the one reliable "which backends
    // made it" signal a host callback gets. It must arrive on a FRESH scan
    // (idempotent repeats stay quiet) and name at least one device in this
    // static build.
    Record rec;
    transcribe_log_set(recording_cb, &rec);

    CHECK(transcribe_init_backends(".") == TRANSCRIBE_OK);
    bool saw_summary = false;
    for (const auto & entry : rec) {
        if (entry.first == TRANSCRIBE_LOG_LEVEL_INFO &&
            entry.second.find("compute device(s) registered") != std::string::npos) {
            saw_summary = true;
        }
    }
    CHECK(saw_summary);

    // Idempotent repeat: same directory, no second summary.
    const size_t n_before = rec.size();
    CHECK(transcribe_init_backends(".") == TRANSCRIBE_OK);
    CHECK(rec.size() == n_before);
}

void test_ggml_bridge_honors_disable() {
    Record rec;
    transcribe_log_set(recording_cb, &rec);

    ggml_log_callback bridge          = nullptr;
    void *            bridge_userdata = nullptr;
    ggml_log_get(&bridge, &bridge_userdata);
    CHECK(bridge != nullptr);
    if (bridge == nullptr) {
        return;
    }

    // Disabling the transcribe sink must also silence ggml's diagnostics:
    // the bridge stays installed but drops on the disabled state.
    transcribe_log_set(nullptr, nullptr);
    bridge(GGML_LOG_LEVEL_ERROR, "dropped\n", bridge_userdata);
    CHECK(rec.empty());
}

}  // namespace

int main() {
    test_callback_receives_log_msg();
    test_null_disables_then_reinstall_restores();
    test_ggml_bridge_installed_and_maps_levels();
    test_init_backends_emits_device_summary();
    test_ggml_bridge_honors_disable();
    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

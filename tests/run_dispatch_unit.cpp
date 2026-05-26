// run_dispatch_unit.cpp - dispatcher-level transcribe_run behavior tests.

#include "transcribe-session.h"
#include "transcribe.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

int g_failures = 0;

#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::fprintf(stderr, "FAIL %s:%d: %s\n",                        \
                         __FILE__, __LINE__, #cond);                        \
            ++g_failures;                                                   \
        }                                                                   \
    } while (0)

} // namespace

int main() {
    transcribe_session session;
    session.full_text = "stale";
    session.has_result = true;
    session.t_mel_us = 1000;
    session.t_encode_us = 2000;
    session.t_decode_us = 3000;

    float pcm = 0.0f;
    transcribe_run_params params = transcribe_run_default_params();
    const transcribe_status st = transcribe_run(&session, &pcm, 1, &params);

    CHECK(st == TRANSCRIBE_ERR_NOT_IMPLEMENTED);
    CHECK(!session.has_result);
    CHECK(session.full_text.empty());

    transcribe_timings t = TRANSCRIBE_TIMINGS_INIT;
    CHECK(transcribe_get_timings(&session, &t) == TRANSCRIBE_OK);
    CHECK(t.mel_ms == 0.0f);
    CHECK(t.encode_ms == 0.0f);
    CHECK(t.decode_ms == 0.0f);

    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

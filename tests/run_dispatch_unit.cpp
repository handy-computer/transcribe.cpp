// run_dispatch_unit.cpp - dispatcher-level transcribe_run behavior tests.

#include "transcribe-context.h"
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
    transcribe_context ctx;
    ctx.full_text = "stale";
    ctx.has_result = true;
    ctx.t_mel_us = 1000;
    ctx.t_encode_us = 2000;
    ctx.t_decode_us = 3000;

    float pcm = 0.0f;
    transcribe_params params = transcribe_default_params();
    const transcribe_status st = transcribe_run(&ctx, &pcm, 1, &params);

    CHECK(st == TRANSCRIBE_ERR_NOT_IMPLEMENTED);
    CHECK(!ctx.has_result);
    CHECK(ctx.full_text.empty());

    const transcribe_timings t = transcribe_get_timings(&ctx);
    CHECK(t.mel_ms == 0.0f);
    CHECK(t.encode_ms == 0.0f);
    CHECK(t.decode_ms == 0.0f);

    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

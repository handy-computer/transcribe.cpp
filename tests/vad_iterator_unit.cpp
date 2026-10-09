// Model-free live VAD policy, pinned to upstream VADIterator with a stub
// probability model. Includes allocation-failure transaction/ownership checks.
#include "transcribe/vad.h"

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

// Allocation interposition for static-library tests only (not portable across
// shared-library boundaries); never a shipped fault hook.
#ifdef TRANSCRIBE_TEST_ALLOC_INTERPOSE
static int g_allocations_before_failure = -1;

void * operator new(size_t n) {
    if (g_allocations_before_failure == 0) {
        g_allocations_before_failure = -1;
        throw std::bad_alloc();
    }
    if (g_allocations_before_failure > 0) {
        --g_allocations_before_failure;
    }
    if (void * p = std::malloc(n ? n : 1)) {
        return p;
    }
    throw std::bad_alloc();
}

void operator delete(void * p) noexcept {
    std::free(p);
}

void operator delete(void * p, size_t) noexcept {
    std::free(p);
}

#endif

namespace {
int          g_failures = 0;
const char * g_case     = "API";
#define CHECK(x)                                                                           \
    do {                                                                                   \
        if (!(x)) {                                                                        \
            std::fprintf(stderr, "FAIL %s:%d [%s]: %s\n", __FILE__, __LINE__, g_case, #x); \
            ++g_failures;                                                                  \
        }                                                                                  \
    } while (0)

struct GoldenEvent {
    transcribe_vad_event_type type;
    int64_t                   sample;
    int32_t                   detection_frame;
};

struct Golden {
    const char *             name;
    int32_t                  frame;
    double                   threshold;
    int32_t                  silence_ms;
    int32_t                  pad_ms;
    std::vector<float>       probs;
    std::vector<GoldenEvent> events;
    bool                     triggered;
};

const Golden k_vectors[] = {
#include "fixtures/vad_iterator_vectors.inc"
};

transcribe_vad_iterator_result result(transcribe_vad_iterator * it) {
    transcribe_vad_iterator_result r;
    transcribe_vad_iterator_result_init(&r);
    CHECK(transcribe_vad_iterator_get_result(it, &r) == TRANSCRIBE_OK);
    return r;
}

transcribe_vad_event event(transcribe_vad_iterator * it, int32_t i) {
    transcribe_vad_event e;
    transcribe_vad_event_init(&e);
    CHECK(transcribe_vad_iterator_get_event(it, i, &e) == TRANSCRIBE_OK);
    return e;
}

void goldens() {
    for (const Golden & g : k_vectors) {
        g_case = g.name;
        transcribe_vad_iterator_params p;
        transcribe_vad_iterator_params_init(&p);
        p.threshold                  = g.threshold;
        p.min_silence_ms             = g.silence_ms;
        p.speech_pad_ms              = g.pad_ms;
        transcribe_vad_iterator * it = nullptr;
        CHECK(transcribe_vad_iterator_init(g.frame, &p, &it) == TRANSCRIBE_OK);
        // Individual frames, odd blocks and one batch must emit the same events.
        for (int mode = 0; mode < 3; ++mode) {
            transcribe_vad_iterator_reset(it);
            CHECK(result(it).current_sample == 0);
            CHECK(result(it).n_events == 0 && !result(it).triggered);
            size_t next_event = 0;
            size_t pos        = 0;
            while (pos < g.probs.size()) {
                const size_t  block  = mode == 0 ? 1 : mode == 1 ? 7 : g.probs.size();
                const size_t  n      = block < g.probs.size() - pos ? block : g.probs.size() - pos;
                const int64_t before = result(it).current_sample;
                CHECK(transcribe_vad_iterator_feed(it, nullptr, 0) == TRANSCRIBE_OK);
                CHECK(result(it).n_events == 0 && result(it).current_sample == before);
                CHECK(transcribe_vad_iterator_feed(it, g.probs.data() + pos, static_cast<int32_t>(n)) == TRANSCRIBE_OK);
                const auto r = result(it);
                CHECK(r.current_sample == static_cast<int64_t>(pos + n) * g.frame);
                int32_t expected_in_call = 0;
                while (next_event < g.events.size() &&
                       static_cast<size_t>(g.events[next_event].detection_frame) < pos + n) {
                    const auto & expected = g.events[next_event++];
                    const auto   e        = event(it, expected_in_call++);
                    CHECK(e.type == expected.type && e.sample == expected.sample);
                }
                CHECK(r.n_events == expected_in_call);
                pos += n;
            }
            CHECK(next_event == g.events.size());
            CHECK(result(it).triggered == g.triggered);
            // EOF is just absence of further feeds: even an empty feed
            // cannot close active speech or pending silence.
            CHECK(transcribe_vad_iterator_feed(it, nullptr, 0) == TRANSCRIBE_OK);
            CHECK(result(it).triggered == g.triggered);
            CHECK(result(it).current_sample == static_cast<int64_t>(g.probs.size()) * g.frame);
            CHECK(result(it).n_events == 0);
        }
        transcribe_vad_iterator_free(it);
    }
}

void invalid_and_ownership() {
    g_case = "validation-and-failure-rollback";
    transcribe_vad_iterator_params p;
    transcribe_vad_iterator_params_init(&p);
    CHECK(p.struct_size == sizeof(p));
    CHECK(p.threshold == .5 && p.min_silence_ms == 100 && p.speech_pad_ms == 30);
    transcribe_vad_iterator_params_init(nullptr);
    transcribe_vad_event_init(nullptr);
    transcribe_vad_iterator_result_init(nullptr);
    transcribe_vad_iterator_reset(nullptr);
    transcribe_vad_iterator_free(nullptr);
    CHECK(transcribe_vad_iterator_init(512, &p, nullptr) == TRANSCRIBE_ERR_INVALID_ARG);
    for (int bad = 0; bad < 9; ++bad) {
        auto    q     = p;
        int32_t frame = 512;
        if (bad == 0) {
            frame = 0;
        }
        if (bad == 1) {
            frame = -1;
        }
        if (bad == 2) {
            q.threshold = -.01;
        }
        if (bad == 3) {
            q.threshold = 1.01;
        }
        if (bad == 4) {
            q.threshold = std::numeric_limits<double>::quiet_NaN();
        }
        if (bad == 5) {
            q.threshold = std::numeric_limits<double>::infinity();
        }
        if (bad == 6) {
            q.min_silence_ms = -1;
        }
        if (bad == 7) {
            q.speech_pad_ms = -1;
        }
        if (bad == 8) {
            q.struct_size = 0;
        }
        auto it = reinterpret_cast<transcribe_vad_iterator *>(uintptr_t{ 1 });
        CHECK(transcribe_vad_iterator_init(frame, &q, &it) ==
              (bad == 8 ? TRANSCRIBE_ERR_BAD_STRUCT_SIZE : TRANSCRIBE_ERR_INVALID_ARG));
        CHECK(it == nullptr);
    }
    auto it = reinterpret_cast<transcribe_vad_iterator *>(uintptr_t{ 1 });
#ifdef TRANSCRIBE_TEST_ALLOC_INTERPOSE
    g_allocations_before_failure = 0;
    CHECK(transcribe_vad_iterator_init(512, nullptr, &it) == TRANSCRIBE_ERR_OOM);
    CHECK(it == nullptr);
#endif
    CHECK(transcribe_vad_iterator_init(512, nullptr, &it) == TRANSCRIBE_OK);
    const float start = .5f;
    CHECK(transcribe_vad_iterator_feed(it, &start, 1) == TRANSCRIBE_OK);
    CHECK(result(it).triggered && result(it).current_sample == 512 && result(it).n_events == 1);
    const auto owned_copy = event(it, 0);
    CHECK(transcribe_vad_iterator_feed(nullptr, &start, 1) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_vad_iterator_feed(it, nullptr, 1) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_vad_iterator_feed(it, &start, -1) == TRANSCRIBE_ERR_INVALID_ARG);
    const float invalid[] = { -1, 1.1f, std::numeric_limits<float>::quiet_NaN(),
                              std::numeric_limits<float>::infinity() };
    for (float bad : invalid) {
        // If validation were interleaved with mutation the five leading
        // silence frames would already emit END and advance the clock.
        const float probs[] = { 0, 0, 0, 0, 0, bad };
        CHECK(transcribe_vad_iterator_feed(it, probs, 6) == TRANSCRIBE_ERR_INVALID_ARG);
        CHECK(result(it).triggered && result(it).current_sample == 512 && result(it).n_events == 1);
        CHECK(event(it, 0).sample == owned_copy.sample);
    }
    transcribe_vad_event e;
    transcribe_vad_event_init(&e);
    e.sample = 1234;
    CHECK(transcribe_vad_iterator_get_event(it, -1, &e) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_vad_iterator_get_event(it, 1, &e) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_vad_iterator_get_event(nullptr, 0, &e) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_vad_iterator_get_event(it, 0, nullptr) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(e.sample == 1234);
    e.struct_size = 0;
    CHECK(transcribe_vad_iterator_get_event(it, 0, &e) == TRANSCRIBE_ERR_BAD_STRUCT_SIZE);
    CHECK(e.sample == 1234);
    transcribe_vad_iterator_result r;
    transcribe_vad_iterator_result_init(&r);
    r.current_sample = 1234;
    r.struct_size    = 0;
    CHECK(transcribe_vad_iterator_get_result(it, &r) == TRANSCRIBE_ERR_BAD_STRUCT_SIZE);
    CHECK(r.current_sample == 1234);
    CHECK(transcribe_vad_iterator_get_result(nullptr, &r) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_vad_iterator_get_result(it, nullptr) == TRANSCRIBE_ERR_INVALID_ARG);
    transcribe_vad_iterator_reset(it);
    CHECK(!result(it).triggered && result(it).n_events == 0 && result(it).current_sample == 0);
    transcribe_vad_iterator_free(it);
    CHECK(owned_copy.type == TRANSCRIBE_VAD_EVENT_START && owned_copy.sample == 0);

    // Transaction rolls back after several staged START/ENDs, not just the
    // first allocation. No runtime fault hook in the library is needed.
    p.min_silence_ms = 0;
    p.speech_pad_ms  = 0;
    CHECK(transcribe_vad_iterator_init(512, &p, &it) == TRANSCRIBE_OK);
    CHECK(transcribe_vad_iterator_feed(it, &start, 1) == TRANSCRIBE_OK);
    const float many[] = { 0, .5f, 0, .5f, 0, .5f };
#ifdef TRANSCRIBE_TEST_ALLOC_INTERPOSE
    g_allocations_before_failure = 2;
    CHECK(transcribe_vad_iterator_feed(it, many, 6) == TRANSCRIBE_ERR_OOM);
    g_allocations_before_failure = -1;
    CHECK(result(it).triggered && result(it).current_sample == 512 && result(it).n_events == 1);
    CHECK(event(it, 0).type == TRANSCRIBE_VAD_EVENT_START);
#endif
    CHECK(transcribe_vad_iterator_feed(it, many, 6) == TRANSCRIBE_OK);
    CHECK(result(it).n_events == 6 && result(it).current_sample == 3584);
    transcribe_vad_iterator_free(it);
}

void struct_sizes() {
    g_case = "ABI-prefix-suffix-and-large-clock";
    CHECK(transcribe_abi_struct_size(TRANSCRIBE_ABI_VAD_ITERATOR_PARAMS) == sizeof(transcribe_vad_iterator_params));
    CHECK(transcribe_abi_struct_align(TRANSCRIBE_ABI_VAD_ITERATOR_PARAMS) == alignof(transcribe_vad_iterator_params));
    CHECK(transcribe_abi_struct_size(TRANSCRIBE_ABI_VAD_EVENT) == sizeof(transcribe_vad_event));
    CHECK(transcribe_abi_struct_align(TRANSCRIBE_ABI_VAD_EVENT) == alignof(transcribe_vad_event));
    CHECK(transcribe_abi_struct_size(TRANSCRIBE_ABI_VAD_ITERATOR_RESULT) == sizeof(transcribe_vad_iterator_result));
    CHECK(transcribe_abi_struct_align(TRANSCRIBE_ABI_VAD_ITERATOR_RESULT) == alignof(transcribe_vad_iterator_result));
    transcribe_vad_iterator_params p;
    transcribe_vad_iterator_params_init(&p);
    // The required prefix, not sizeof including tail padding, is the gate.
    const size_t              p_min = offsetof(transcribe_vad_iterator_params, speech_pad_ms) + sizeof(p.speech_pad_ms);
    transcribe_vad_iterator * it    = nullptr;
    p.struct_size                   = p_min - 1;
    CHECK(transcribe_vad_iterator_init(512, &p, &it) == TRANSCRIBE_ERR_BAD_STRUCT_SIZE && it == nullptr);
    p.struct_size = p_min;
    CHECK(transcribe_vad_iterator_init(512, &p, &it) == TRANSCRIBE_OK);
    const float start = 1;
    CHECK(transcribe_vad_iterator_feed(it, &start, 1) == TRANSCRIBE_OK);

    struct ExtendedResult {
        transcribe_vad_iterator_result r;
        uint64_t                       tail;
    } x;

    transcribe_vad_iterator_result_init(&x.r);
    x.r.struct_size = sizeof(x);
    x.tail          = 0x12345678;
    CHECK(transcribe_vad_iterator_get_result(it, &x.r) == TRANSCRIBE_OK);
    CHECK(x.r.struct_size == sizeof(x) && x.tail == 0x12345678);
    const size_t r_min = offsetof(transcribe_vad_iterator_result, triggered) + sizeof(x.r.triggered);
    x.r.struct_size    = r_min - 1;
    CHECK(transcribe_vad_iterator_get_result(it, &x.r) == TRANSCRIBE_ERR_BAD_STRUCT_SIZE);
    x.r.struct_size = r_min;
    // Fill tail padding with a sentinel; copy-out must not write beyond r_min.
    std::memset(reinterpret_cast<char *>(&x.r) + r_min, 0x5a, sizeof(x.r) - r_min);
    CHECK(transcribe_vad_iterator_get_result(it, &x.r) == TRANSCRIBE_OK);
    for (size_t i = r_min; i < sizeof(x.r); ++i) {
        CHECK(reinterpret_cast<unsigned char *>(&x.r)[i] == 0x5a);
    }

    struct ExtendedEvent {
        transcribe_vad_event e;
        uint64_t             tail;
    } y;

    transcribe_vad_event_init(&y.e);
    CHECK(y.e.struct_size == sizeof(y.e));
    y.e.struct_size = sizeof(y);
    y.tail          = 0x12345678;
    CHECK(transcribe_vad_iterator_get_event(it, 0, &y.e) == TRANSCRIBE_OK);
    CHECK(y.e.struct_size == sizeof(y) && y.tail == 0x12345678);
    y.e.struct_size = offsetof(transcribe_vad_event, sample) + sizeof(y.e.sample) - 1;
    CHECK(transcribe_vad_iterator_get_event(it, 0, &y.e) == TRANSCRIBE_ERR_BAD_STRUCT_SIZE);
    transcribe_vad_iterator_free(it);

    // Large int32 parameters/frame sizes must not overflow 32-bit sample
    // arithmetic. Live padding deliberately may extend beyond the input.
    transcribe_vad_iterator_params_init(&p);
    p.min_silence_ms    = 0;
    p.speech_pad_ms     = std::numeric_limits<int32_t>::max();
    const int32_t frame = std::numeric_limits<int32_t>::max();
    CHECK(transcribe_vad_iterator_init(frame, &p, &it) == TRANSCRIBE_OK);
    const float large[] = { 1, 0, 1 };
    CHECK(transcribe_vad_iterator_feed(it, large, 3) == TRANSCRIBE_OK);
    CHECK(result(it).current_sample == int64_t{ 3 } * frame && result(it).n_events == 3);
    CHECK(event(it, 1).sample == int64_t{ 17 } * frame);
    CHECK(event(it, 2).sample == 0);  // overlapping, not sorted by timestamp
    transcribe_vad_iterator_free(it);

    // Forward-compatible input suffix is ignored.
    struct ExtendedParams {
        transcribe_vad_iterator_params p;
        uint64_t                       tail;
    } z = { p, 0x12345678 };

    z.p.struct_size = sizeof(z);
    CHECK(transcribe_vad_iterator_init(512, &z.p, &it) == TRANSCRIBE_OK);
    transcribe_vad_iterator_free(it);
}
}  // namespace

int main(int argc, char ** argv) {
    // Optional parity runner for real-model probability dumps. No model or
    // reference dependencies enter the default unit test.
    if (argc == 3 && std::strcmp(argv[1], "--probs") == 0) {
        FILE * f = std::fopen(argv[2], "rb");
        if (f == nullptr) {
            return 2;
        }
        transcribe_vad_iterator * it = nullptr;
        if (transcribe_vad_iterator_init(512, nullptr, &it) != TRANSCRIBE_OK) {
            std::fclose(f);
            return 2;
        }
        float  probs[7];
        size_t n;
        while ((n = std::fread(probs, sizeof(float), 7, f)) > 0) {
            CHECK(transcribe_vad_iterator_feed(it, probs, static_cast<int32_t>(n)) == TRANSCRIBE_OK);
            const auto r = result(it);
            for (int32_t i = 0; i < r.n_events; ++i) {
                const auto e = event(it, i);
                std::printf("%s %lld\n", e.type == TRANSCRIBE_VAD_EVENT_START ? "start" : "end",
                            static_cast<long long>(e.sample));
            }
        }
        CHECK(!std::ferror(f));
        std::fclose(f);
        transcribe_vad_iterator_free(it);
        return g_failures ? 1 : 0;
    }
    if (argc != 1) {
        return 2;
    }
    goldens();
    invalid_and_ownership();
    struct_sizes();
    if (!g_failures) {
        std::puts("VAD iterator: 15 named upstream cases, grouping/reset/EOF, validation and ABI PASS");
    }
    return g_failures ? 1 : 0;
}

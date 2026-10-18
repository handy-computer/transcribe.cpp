// langid_dispatch_unit.cpp - LANGID role dispatcher (transcribe-langid.cpp)
// against a fake arch: label table validation, role checks, softmax /
// ranking, the allowed set, crop and minimum length, non-finite input and
// logits, abort.

#include "transcribe-arch.h"
#include "transcribe-langid.h"
#include "transcribe-model.h"
#include "transcribe.h"
#include "transcribe/langid.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
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

bool near(double a, double b, double tol = 1e-6) {
    return std::fabs(a - b) <= tol;
}

transcribe::LangidLabels g_labels;  // aa..ee, alias xx=aa

int                g_run_calls   = 0;
int                g_last_n      = 0;
const float *      g_last_pcm    = nullptr;
std::vector<float> g_logits      = { 1.0f, 3.0f, 2.0f, 0.0f, -1.0f };
bool               g_check_abort = false;

const transcribe::LangidLabels & fake_labels(const transcribe_model *) {
    return g_labels;
}

transcribe_langid_session * fake_new_session() {
    return new transcribe_langid_session();
}

transcribe_status fake_run(transcribe_langid_session * s, const float * pcm, int n, std::vector<float> & logits) {
    ++g_run_calls;
    g_last_n   = n;
    g_last_pcm = pcm;
    if (g_check_abort && s->poll_abort()) {
        return TRANSCRIBE_ERR_ABORTED;
    }
    logits = g_logits;
    return TRANSCRIBE_OK;
}

const transcribe::LangidOps k_ops = { fake_labels, fake_new_session, fake_run };

const transcribe::Arch k_arch = {
    /* .name             = */ "fake-langid",
    /* .load             = */ nullptr,
    /* .init_context     = */ nullptr,
    /* .run              = */ nullptr,
    /* .run_batch        = */ nullptr,
    /* .stream_validate  = */ nullptr,
    /* .stream_begin     = */ nullptr,
    /* .stream_feed      = */ nullptr,
    /* .stream_finalize  = */ nullptr,
    /* .stream_reset     = */ nullptr,
    /* .accepts_ext_kind = */ nullptr,
    /* .run_validate     = */ nullptr,
    /* .diarize          = */ nullptr,
    /* .langid           = */ &k_ops,
};

struct Fixture {
    transcribe_model            model;
    transcribe_langid_session * session = nullptr;
    std::vector<float>          pcm     = std::vector<float>(16000, 0.0f);  // 1 s

    Fixture() {
        g_run_calls   = 0;
        g_logits      = { 1.0f, 3.0f, 2.0f, 0.0f, -1.0f };
        g_check_abort = false;
        model.arch    = &k_arch;
        model.roles   = TRANSCRIBE_ROLE_LANGID;
        transcribe_langid_session_params sp;
        transcribe_langid_session_params_init(&sp);
        CHECK(transcribe_langid_session_init(&model, &sp, &session) == TRANSCRIBE_OK);
    }

    ~Fixture() { transcribe_langid_session_free(session); }

    transcribe_status run(const transcribe_langid_params * p = nullptr) {
        return transcribe_langid_run(session, pcm.data(), static_cast<int>(pcm.size()), p);
    }

    transcribe_langid_result result() const {
        transcribe_langid_result r;
        transcribe_langid_result_init(&r);
        CHECK(transcribe_langid_get_result(session, &r) == TRANSCRIBE_OK);
        return r;
    }

    transcribe_langid_candidate candidate(int i) const {
        transcribe_langid_candidate c;
        transcribe_langid_candidate_init(&c);
        CHECK(transcribe_langid_get_candidate(session, i, &c) == TRANSCRIBE_OK);
        return c;
    }
};

// Malformed label metadata fails the load instead of being last-wins.
void test_label_table() {
    const std::vector<std::string> names = { "A", "B" };
    transcribe::LangidLabels       out;
    CHECK(transcribe::build_langid_labels({ "aa", "bb" }, names, { "xx=aa" }, "test", out) == TRANSCRIBE_OK);
    CHECK(out.codes.size() == 2 && out.index.size() == 3 && out.index.at("xx") == 0);
    CHECK(transcribe::build_langid_labels({ "aa", "aa" }, names, {}, "test", out) == TRANSCRIBE_ERR_GGUF);
    CHECK(transcribe::build_langid_labels({ "aa", "bb" }, names, { "xx=zz" }, "test", out) == TRANSCRIBE_ERR_GGUF);
}

void test_role_checks_and_labels() {
    transcribe_model model;
    model.arch  = &k_arch;
    model.roles = TRANSCRIBE_ROLE_ASR;  // no LANGID bit

    transcribe_langid_session * s = reinterpret_cast<transcribe_langid_session *>(0x1);
    CHECK(transcribe_langid_session_init(&model, nullptr, &s) == TRANSCRIBE_ERR_UNSUPPORTED_ROLE);
    CHECK(s == nullptr);
    transcribe_langid_info info;
    transcribe_langid_info_init(&info);
    CHECK(transcribe_langid_get_info(&model, &info) == TRANSCRIBE_ERR_UNSUPPORTED_ROLE);

    model.roles = TRANSCRIBE_ROLE_LANGID;
    CHECK(transcribe_langid_get_info(&model, &info) == TRANSCRIBE_OK);
    CHECK(info.sample_rate == 16000 && info.n_labels == 5 && info.min_audio_ms == 500 && info.max_audio_ms == 30000);
    CHECK(std::strcmp(transcribe_langid_label_code(&model, 1), "bb") == 0);
    CHECK(std::strcmp(transcribe_langid_label_name(&model, 4), "Ee") == 0);
    CHECK(transcribe_langid_label_index(&model, "cc") == 2);
    CHECK(transcribe_langid_label_index(&model, "xx") == 0);
    CHECK(transcribe_langid_label_index(&model, "zz") == -1);

    // A negative thread count is rejected up front.
    transcribe_langid_session_params sp;
    transcribe_langid_session_params_init(&sp);
    sp.n_threads = -1;
    CHECK(transcribe_langid_session_init(&model, &sp, &s) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(s == nullptr);
}

void test_ranking() {
    Fixture f;
    CHECK(f.run() == TRANSCRIBE_OK);
    CHECK(g_run_calls == 1 && g_last_n == 16000 && g_last_pcm == f.pcm.data());

    const transcribe_langid_result r = f.result();
    CHECK(r.n_candidates == 5 && r.allowed_mass == 1.0f);

    // Logits {1, 3, 2, 0, -1} rank bb, cc, aa, dd, ee.
    const int    want[] = { 1, 2, 0, 3, 4 };
    double       sum    = 0.0;
    const double denom  = std::exp(1.0) + std::exp(3.0) + std::exp(2.0) + std::exp(0.0) + std::exp(-1.0);
    for (int i = 0; i < 5; ++i) {
        const transcribe_langid_candidate c = f.candidate(i);
        CHECK(c.index == want[i]);
        CHECK(c.code != nullptr && std::strcmp(c.code, g_labels.codes[want[i]].c_str()) == 0);
        CHECK(c.logit == g_logits[want[i]]);
        CHECK(near(c.p, std::exp(static_cast<double>(g_logits[want[i]])) / denom));
        sum += c.p;
    }
    CHECK(near(sum, 1.0));
    CHECK(f.candidate(5).code == nullptr && f.candidate(5).index == 0);  // out of range: zeroed row

    // Tied p keeps label order.
    g_logits = { 0.0f, 2.0f, 0.0f, 2.0f, 0.0f };
    CHECK(f.run() == TRANSCRIBE_OK);
    const int tied[] = { 1, 3, 0, 2, 4 };
    for (int i = 0; i < 5; ++i) {
        CHECK(f.candidate(i).index == tied[i]);
    }
}

// The allowed set renormalizes over its members; malformed or unknown lists
// are rejected before the family runs and keep the last result.
void test_allowed_set() {
    Fixture                  f;
    transcribe_langid_params p;
    transcribe_langid_params_init(&p);

    const char * dd_bb[] = { "dd", "bb" };
    p.allowed            = dd_bb;
    p.n_allowed          = 2;
    CHECK(f.run(&p) == TRANSCRIBE_OK);
    const transcribe_langid_result r = f.result();
    CHECK(r.n_candidates == 2);
    const double e_bb = std::exp(3.0);
    const double e_dd = std::exp(0.0);
    const double all  = std::exp(1.0) + e_bb + std::exp(2.0) + e_dd + std::exp(-1.0);
    CHECK(near(r.allowed_mass, (e_bb + e_dd) / all));
    const transcribe_langid_candidate c0 = f.candidate(0);
    const transcribe_langid_candidate c1 = f.candidate(1);
    CHECK(c0.index == 1 && c1.index == 3);
    CHECK(near(c0.p, e_bb / (e_bb + e_dd)) && near(c1.p, e_dd / (e_bb + e_dd)));

    // n_candidates is the allowed-set size after duplicates (an alias names
    // the same label as its code) count once.
    const char * dups[] = { "bb", "bb", "xx", "aa" };
    p.allowed           = dups;
    p.n_allowed         = 4;
    CHECK(f.run(&p) == TRANSCRIBE_OK);
    CHECK(f.result().n_candidates == 2);
    CHECK(f.candidate(0).index == 1 && f.candidate(1).index == 0 && f.candidate(2).code == nullptr);

    g_run_calls = 0;
    p.n_allowed = 0;
    CHECK(f.run(&p) == TRANSCRIBE_ERR_INVALID_ARG);
    const char * unknown[] = { "aa", "zz" };
    p.allowed              = unknown;
    p.n_allowed            = 2;
    CHECK(f.run(&p) == TRANSCRIBE_ERR_UNSUPPORTED_LANGUAGE);
    CHECK(g_run_calls == 0);
    CHECK(f.result().n_candidates == 2);  // previous result kept
}

// The crop keeps the first info.max_audio_ms, and the minimum applies to the
// audio actually scored.
void test_crop_and_minimum() {
    {
        Fixture f;
        f.pcm.assign(16 * (30000 + 1000), 0.0f);  // 31 s, scored as its first 30 s
        CHECK(f.run() == TRANSCRIBE_OK);
        CHECK(g_last_n == 16 * 30000 && g_last_pcm == f.pcm.data());
        CHECK(f.result().n_candidates == 5);
        f.pcm.resize(16 * 30000);  // exactly max_audio_ms: no crop
        CHECK(f.run() == TRANSCRIBE_OK);
        CHECK(g_last_n == 16 * 30000 && g_last_pcm == f.pcm.data());
    }
    {
        Fixture f;
        CHECK(transcribe_langid_run(f.session, f.pcm.data(), 8000, nullptr) == TRANSCRIBE_OK);  // exactly 500 ms
        CHECK(g_last_n == 8000);
        g_run_calls = 0;
        CHECK(transcribe_langid_run(f.session, f.pcm.data(), 7999, nullptr) == TRANSCRIBE_ERR_INPUT_TOO_SHORT);
        CHECK(g_run_calls == 0);
        CHECK(f.result().n_candidates == 5);  // previous result kept
    }
}

// Non-finite PCM is rejected up front; non-finite or miscounted logits fail
// the run instead of ranking garbage.
void test_non_finite() {
    Fixture f;
    CHECK(f.run() == TRANSCRIBE_OK);
    g_run_calls = 0;

    std::vector<float> bad(16000, 0.0f);
    bad[3] = std::numeric_limits<float>::quiet_NaN();
    CHECK(transcribe_langid_run(f.session, bad.data(), 16000, nullptr) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(g_run_calls == 0);
    CHECK(f.result().n_candidates == 5);

    g_logits = { 1.0f, std::numeric_limits<float>::quiet_NaN(), 0.0f, 0.0f, 0.0f };
    CHECK(f.run() == TRANSCRIBE_ERR_BACKEND);
    CHECK(f.result().n_candidates == 0 && f.result().allowed_mass == 0.0f);
    g_logits = { 1.0f, 2.0f };  // 2 logits for 5 labels
    CHECK(f.run() == TRANSCRIBE_ERR_BACKEND);
}

// The abort callback reaches the family, and the aborted run clears the result.
void test_abort() {
    Fixture f;
    CHECK(f.run() == TRANSCRIBE_OK);
    g_check_abort = true;
    transcribe_langid_set_abort_callback(f.session, [](void *) { return true; }, nullptr);
    CHECK(f.run() == TRANSCRIBE_ERR_ABORTED);
    CHECK(f.result().n_candidates == 0);
    transcribe_langid_set_abort_callback(f.session, nullptr, nullptr);
    CHECK(f.run() == TRANSCRIBE_OK);
}

}  // namespace

int main() {
    transcribe_log_set(nullptr, nullptr);

    if (transcribe::build_langid_labels({ "aa", "bb", "cc", "dd", "ee" }, { "Aa", "Bb", "Cc", "Dd", "Ee" }, { "xx=aa" },
                                        "test", g_labels) != TRANSCRIBE_OK) {
        std::fprintf(stderr, "FAIL: fixture label table\n");
        return EXIT_FAILURE;
    }

    test_label_table();
    test_role_checks_and_labels();
    test_ranking();
    test_allowed_set();
    test_crop_and_minimum();
    test_non_finite();
    test_abort();

    if (g_failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return EXIT_FAILURE;
    }
    std::printf("langid_dispatch_unit: ok\n");
    return EXIT_SUCCESS;
}

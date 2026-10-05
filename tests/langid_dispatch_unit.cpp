// langid_dispatch_unit.cpp - LANGID role dispatcher (transcribe-langid.cpp)
// against a fake arch: role checks, label table validation (H6), the
// allowed set (H2), crop and minimum length (H1), non-finite input and
// logits (H3), softmax / ranking / top-k, accessors.

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
transcribe_status  g_run_status  = TRANSCRIBE_OK;
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
    if (g_run_status != TRANSCRIBE_OK) {
        return g_run_status;
    }
    s->t_mel_us    = 1000;
    s->t_encode_us = 2000;
    logits         = g_logits;
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

void reset_fake() {
    g_run_calls   = 0;
    g_last_n      = 0;
    g_last_pcm    = nullptr;
    g_run_status  = TRANSCRIBE_OK;
    g_logits      = { 1.0f, 3.0f, 2.0f, 0.0f, -1.0f };
    g_check_abort = false;
}

struct Fixture {
    transcribe_model            model;
    transcribe_langid_session * session = nullptr;
    std::vector<float>          pcm     = std::vector<float>(16000, 0.0f);  // 1 s

    explicit Fixture(int32_t max_audio_ms = 0) {
        reset_fake();
        model.arch  = &k_arch;
        model.roles = TRANSCRIBE_ROLE_LANGID;
        transcribe_langid_session_params sp;
        transcribe_langid_session_params_init(&sp);
        sp.max_audio_ms = max_audio_ms;
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

transcribe_status labels_from(std::vector<std::string> codes,
                              std::vector<std::string> names,
                              std::vector<std::string> aliases) {
    transcribe::LangidLabels out;
    return transcribe::build_langid_labels(std::move(codes), std::move(names), aliases, "test", out);
}

// H6: malformed label metadata fails the load instead of being last-wins.
void test_label_table() {
    const std::vector<std::string> codes = { "aa", "bb" };
    const std::vector<std::string> names = { "A", "B" };
    CHECK(labels_from(codes, names, {}) == TRANSCRIBE_OK);
    CHECK(labels_from(codes, names, { "xx=aa", "yy=bb" }) == TRANSCRIBE_OK);

    CHECK(labels_from({}, {}, {}) == TRANSCRIBE_ERR_GGUF);
    CHECK(labels_from(codes, { "A" }, {}) == TRANSCRIBE_ERR_GGUF);
    CHECK(labels_from({ "aa", "" }, names, {}) == TRANSCRIBE_ERR_GGUF);
    CHECK(labels_from({ "aa", "aa" }, names, {}) == TRANSCRIBE_ERR_GGUF);
    CHECK(labels_from(codes, names, { "xx" }) == TRANSCRIBE_ERR_GGUF);
    CHECK(labels_from(codes, names, { "=aa" }) == TRANSCRIBE_ERR_GGUF);
    CHECK(labels_from(codes, names, { "xx=" }) == TRANSCRIBE_ERR_GGUF);
    CHECK(labels_from(codes, names, { "xx=zz" }) == TRANSCRIBE_ERR_GGUF);           // unknown target
    CHECK(labels_from(codes, names, { "bb=aa" }) == TRANSCRIBE_ERR_GGUF);           // alias remaps a code
    CHECK(labels_from(codes, names, { "xx=aa", "xx=bb" }) == TRANSCRIBE_ERR_GGUF);  // alias repeated
    CHECK(labels_from(codes, names, { "xx=aa", "yy=xx" }) == TRANSCRIBE_ERR_GGUF);  // alias of an alias

    transcribe::LangidLabels out;
    CHECK(transcribe::build_langid_labels({ "aa" }, { "A" }, { "xx=aa" }, "test", out) == TRANSCRIBE_OK);
    CHECK(out.codes.size() == 1 && out.index.size() == 2 && out.index.at("xx") == 0);
}

void test_role_checks_and_labels() {
    reset_fake();
    transcribe_model model;
    model.arch  = &k_arch;
    model.roles = TRANSCRIBE_ROLE_ASR;  // no LANGID bit

    transcribe_langid_session * s = reinterpret_cast<transcribe_langid_session *>(0x1);
    CHECK(transcribe_langid_session_init(&model, nullptr, &s) == TRANSCRIBE_ERR_UNSUPPORTED_ROLE);
    CHECK(s == nullptr);
    transcribe_langid_info info;
    transcribe_langid_info_init(&info);
    CHECK(transcribe_langid_get_info(&model, &info) == TRANSCRIBE_ERR_UNSUPPORTED_ROLE);
    CHECK(transcribe_langid_label_code(&model, 0) == nullptr);
    CHECK(transcribe_langid_label_name(&model, 0) == nullptr);
    CHECK(transcribe_langid_label_index(&model, "aa") == -1);

    model.roles = TRANSCRIBE_ROLE_LANGID;
    CHECK(transcribe_langid_get_info(&model, &info) == TRANSCRIBE_OK);
    CHECK(info.sample_rate == 16000 && info.n_labels == 5 && info.min_audio_ms == 500);
    CHECK(transcribe_langid_get_info(nullptr, &info) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_langid_get_info(&model, nullptr) == TRANSCRIBE_ERR_INVALID_ARG);
    info.struct_size = 0;
    CHECK(transcribe_langid_get_info(&model, &info) == TRANSCRIBE_ERR_BAD_STRUCT_SIZE);

    CHECK(std::strcmp(transcribe_langid_label_code(&model, 1), "bb") == 0);
    CHECK(std::strcmp(transcribe_langid_label_name(&model, 4), "Ee") == 0);
    CHECK(transcribe_langid_label_code(&model, 5) == nullptr);
    CHECK(transcribe_langid_label_code(&model, -1) == nullptr);
    CHECK(transcribe_langid_label_code(nullptr, 0) == nullptr);
    CHECK(transcribe_langid_label_index(&model, "cc") == 2);
    CHECK(transcribe_langid_label_index(&model, "xx") == 0);
    CHECK(transcribe_langid_label_index(&model, "zz") == -1);
    CHECK(transcribe_langid_label_index(&model, "") == -1);
    CHECK(transcribe_langid_label_index(&model, nullptr) == -1);
    CHECK(transcribe_langid_label_index(nullptr, "aa") == -1);

    transcribe_langid_session_params sp;
    transcribe_langid_session_params_init(&sp);
    sp.n_threads = -1;
    CHECK(transcribe_langid_session_init(&model, &sp, &s) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(s == nullptr);
    sp.n_threads    = 2;
    sp.max_audio_ms = 499;  // H1: a window below the minimum
    CHECK(transcribe_langid_session_init(&model, &sp, &s) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(s == nullptr);
    sp.max_audio_ms = -1;
    CHECK(transcribe_langid_session_init(&model, &sp, &s) == TRANSCRIBE_ERR_INVALID_ARG);
    sp.struct_size = 0;
    CHECK(transcribe_langid_session_init(&model, &sp, &s) == TRANSCRIBE_ERR_BAD_STRUCT_SIZE);
    CHECK(s == nullptr);
    transcribe_langid_session_params_init(&sp);
    sp.max_audio_ms = 500;
    CHECK(transcribe_langid_session_init(&model, &sp, &s) == TRANSCRIBE_OK);
    CHECK(s != nullptr);
    transcribe_langid_session_free(s);
    CHECK(transcribe_langid_session_init(nullptr, nullptr, &s) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_langid_session_init(&model, nullptr, nullptr) == TRANSCRIBE_ERR_INVALID_ARG);
    transcribe_langid_session_free(nullptr);
}

void test_accessors_before_run() {
    Fixture                  f;
    transcribe_langid_result r = f.result();
    CHECK(r.n_candidates == 0 && r.n_allowed == 0 && r.allowed_mass == 0.0f && r.audio_ms == 0);
    transcribe_langid_candidate c = f.candidate(0);
    CHECK(c.code == nullptr && c.name == nullptr && c.index == 0);

    CHECK(transcribe_langid_get_result(f.session, nullptr) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_langid_get_result(nullptr, &r) == TRANSCRIBE_ERR_INVALID_ARG);
    r.struct_size = 0;
    CHECK(transcribe_langid_get_result(f.session, &r) == TRANSCRIBE_ERR_BAD_STRUCT_SIZE);
    CHECK(transcribe_langid_get_candidate(f.session, 0, nullptr) == TRANSCRIBE_ERR_INVALID_ARG);
    c.struct_size = 0;
    CHECK(transcribe_langid_get_candidate(f.session, 0, &c) == TRANSCRIBE_ERR_BAD_STRUCT_SIZE);
    transcribe_langid_candidate_init(&c);
    CHECK(transcribe_langid_get_candidate(nullptr, 0, &c) == TRANSCRIBE_OK);
    CHECK(c.code == nullptr);

    transcribe_timings tm;
    transcribe_timings_init(&tm);
    CHECK(transcribe_langid_get_timings(f.session, &tm) == TRANSCRIBE_OK);
    CHECK(tm.mel_ms == 0.0f && tm.encode_ms == 0.0f);
    CHECK(transcribe_langid_get_timings(nullptr, &tm) == TRANSCRIBE_ERR_INVALID_ARG);
}

void test_unrestricted_ranking() {
    Fixture f;
    CHECK(f.run() == TRANSCRIBE_OK);
    CHECK(g_run_calls == 1 && g_last_n == 16000 && g_last_pcm == f.pcm.data());

    const transcribe_langid_result r = f.result();
    CHECK(r.n_candidates == 5 && r.n_allowed == 5 && r.allowed_mass == 1.0f && r.audio_ms == 1000);

    // Logits {1, 3, 2, 0, -1} rank bb, cc, aa, dd, ee.
    const int    want[] = { 1, 2, 0, 3, 4 };
    double       sum    = 0.0;
    const double denom  = std::exp(1.0) + std::exp(3.0) + std::exp(2.0) + std::exp(0.0) + std::exp(-1.0);
    for (int i = 0; i < 5; ++i) {
        const transcribe_langid_candidate c = f.candidate(i);
        CHECK(c.index == want[i]);
        CHECK(c.code != nullptr && std::strcmp(c.code, g_labels.codes[want[i]].c_str()) == 0);
        CHECK(c.name != nullptr && std::strcmp(c.name, g_labels.names[want[i]].c_str()) == 0);
        CHECK(c.p == c.p_unrestricted);
        CHECK(c.logit == g_logits[want[i]]);
        CHECK(near(c.p, std::exp(static_cast<double>(g_logits[want[i]])) / denom));
        sum += c.p;
    }
    CHECK(near(sum, 1.0));
    CHECK(f.candidate(5).code == nullptr);  // out of range: zeroed row

    transcribe_timings tm;
    transcribe_timings_init(&tm);
    CHECK(transcribe_langid_get_timings(f.session, &tm) == TRANSCRIBE_OK);
    CHECK(tm.mel_ms == 1.0f && tm.encode_ms == 2.0f);

    // top_k truncates the rows but not n_allowed.
    transcribe_langid_params p;
    transcribe_langid_params_init(&p);
    p.top_k = 2;
    CHECK(f.run(&p) == TRANSCRIBE_OK);
    CHECK(f.result().n_candidates == 2 && f.result().n_allowed == 5);
    CHECK(f.candidate(1).index == 2 && f.candidate(2).code == nullptr);
    p.top_k = 9;
    CHECK(f.run(&p) == TRANSCRIBE_OK);
    CHECK(f.result().n_candidates == 5);

    // Ties keep label order.
    g_logits = { 0.5f, 0.5f, 0.5f, 0.5f, 0.5f };
    CHECK(f.run() == TRANSCRIBE_OK);
    for (int i = 0; i < 5; ++i) {
        CHECK(f.candidate(i).index == i);
    }
}

// H2: only NULL means "all"; every other malformed allowed list is rejected.
void test_allowed_set() {
    Fixture                  f;
    transcribe_langid_params p;
    transcribe_langid_params_init(&p);

    const char * dd_bb[] = { "dd", "bb" };
    p.allowed            = dd_bb;
    p.n_allowed          = 2;
    CHECK(f.run(&p) == TRANSCRIBE_OK);
    transcribe_langid_result r = f.result();
    CHECK(r.n_candidates == 2 && r.n_allowed == 2);
    const double e_bb = std::exp(3.0);
    const double e_dd = std::exp(0.0);
    const double all  = std::exp(1.0) + e_bb + std::exp(2.0) + e_dd + std::exp(-1.0);
    CHECK(near(r.allowed_mass, (e_bb + e_dd) / all));
    const transcribe_langid_candidate c0 = f.candidate(0);
    const transcribe_langid_candidate c1 = f.candidate(1);
    CHECK(c0.index == 1 && c1.index == 3);
    CHECK(near(c0.p, e_bb / (e_bb + e_dd)) && near(c1.p, e_dd / (e_bb + e_dd)));
    CHECK(near(c0.p_unrestricted, e_bb / all));
    CHECK(near(c0.p + c1.p, 1.0));

    const char * one[] = { "cc" };
    p.allowed          = one;
    p.n_allowed        = 1;
    CHECK(f.run(&p) == TRANSCRIBE_OK);
    CHECK(f.result().n_allowed == 1 && f.candidate(0).index == 2 && f.candidate(0).p == 1.0f);

    const char * alias[] = { "xx" };
    p.allowed            = alias;
    CHECK(f.run(&p) == TRANSCRIBE_OK);
    CHECK(f.candidate(0).index == 0 && std::strcmp(f.candidate(0).code, "aa") == 0);

    const char * dup[] = { "bb", "bb", "xx", "aa" };
    p.allowed          = dup;
    p.n_allowed        = 4;
    CHECK(f.run(&p) == TRANSCRIBE_OK);
    CHECK(f.result().n_allowed == 2 && f.result().n_candidates == 2);

    const char * every[] = { "aa", "bb", "cc", "dd", "ee" };
    p.allowed            = every;
    p.n_allowed          = 5;
    CHECK(f.run(&p) == TRANSCRIBE_OK);
    CHECK(f.result().n_allowed == 5 && f.result().allowed_mass == 1.0f);
    CHECK(f.candidate(0).p == f.candidate(0).p_unrestricted);

    // Rejections never reach the family and keep the last result.
    reset_fake();
    p.n_allowed = -1;
    CHECK(f.run(&p) == TRANSCRIBE_ERR_INVALID_ARG);
    p.n_allowed = 0;
    CHECK(f.run(&p) == TRANSCRIBE_ERR_INVALID_ARG);
    p.allowed   = nullptr;
    p.n_allowed = 2;
    CHECK(f.run(&p) == TRANSCRIBE_ERR_INVALID_ARG);
    const char * with_null[] = { "aa", nullptr };
    p.allowed                = with_null;
    CHECK(f.run(&p) == TRANSCRIBE_ERR_INVALID_ARG);
    const char * unknown[] = { "aa", "zz" };
    p.allowed              = unknown;
    CHECK(f.run(&p) == TRANSCRIBE_ERR_UNSUPPORTED_LANGUAGE);
    CHECK(g_run_calls == 0);
    CHECK(f.result().n_allowed == 5);
}

// H1: the crop keeps the last max_audio_ms, and the minimum applies to the
// audio actually scored.
void test_crop_and_minimum() {
    {
        Fixture f(500);
        CHECK(f.run() == TRANSCRIBE_OK);  // 1 s scored as its last 500 ms
        CHECK(g_last_n == 8000 && g_last_pcm == f.pcm.data() + 8000);
        CHECK(f.result().audio_ms == 500);
    }
    {
        Fixture            f;
        std::vector<float> pcm(8000, 0.0f);  // exactly 500 ms
        CHECK(transcribe_langid_run(f.session, pcm.data(), 8000, nullptr) == TRANSCRIBE_OK);
        CHECK(f.result().audio_ms == 500);
        g_run_calls = 0;
        CHECK(transcribe_langid_run(f.session, pcm.data(), 7999, nullptr) == TRANSCRIBE_ERR_INPUT_TOO_SHORT);
        CHECK(g_run_calls == 0);
        CHECK(f.result().audio_ms == 500);  // previous result kept
    }
    {
        Fixture            f(1000);
        std::vector<float> pcm(16000 * 45, 0.0f);
        CHECK(transcribe_langid_run(f.session, pcm.data(), static_cast<int>(pcm.size()), nullptr) == TRANSCRIBE_OK);
        CHECK(g_last_n == 16000 && g_last_pcm == pcm.data() + pcm.size() - 16000);
        CHECK(f.result().audio_ms == 1000);
    }
    {
        Fixture            f;  // default window 30 s
        std::vector<float> pcm(16000 * 45, 0.0f);
        CHECK(transcribe_langid_run(f.session, pcm.data(), static_cast<int>(pcm.size()), nullptr) == TRANSCRIBE_OK);
        CHECK(f.result().audio_ms == 30000);
    }
}

// H3: non-finite PCM is rejected up front; non-finite or miscounted logits
// fail the run instead of ranking garbage.
void test_non_finite() {
    Fixture f;
    CHECK(f.run() == TRANSCRIBE_OK);
    g_run_calls = 0;

    std::vector<float> bad(16000, 0.0f);
    bad[3] = std::numeric_limits<float>::quiet_NaN();
    CHECK(transcribe_langid_run(f.session, bad.data(), 16000, nullptr) == TRANSCRIBE_ERR_INVALID_ARG);
    bad[3] = std::numeric_limits<float>::infinity();
    CHECK(transcribe_langid_run(f.session, bad.data(), 16000, nullptr) == TRANSCRIBE_ERR_INVALID_ARG);
    std::vector<float> all_nan(16000, std::numeric_limits<float>::quiet_NaN());
    CHECK(transcribe_langid_run(f.session, all_nan.data(), 16000, nullptr) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(g_run_calls == 0);
    CHECK(f.result().n_candidates == 5);

    g_logits = { 1.0f, std::numeric_limits<float>::quiet_NaN(), 0.0f, 0.0f, 0.0f };
    CHECK(f.run() == TRANSCRIBE_ERR_BACKEND);
    CHECK(f.result().n_candidates == 0 && f.result().allowed_mass == 0.0f);
    g_logits = { 1.0f, -std::numeric_limits<float>::infinity(), 0.0f, 0.0f, 0.0f };
    CHECK(f.run() == TRANSCRIBE_ERR_BACKEND);
    g_logits = { 1.0f, 2.0f };
    CHECK(f.run() == TRANSCRIBE_ERR_BACKEND);
}

void test_bad_args_and_failures() {
    Fixture f;
    CHECK(f.run() == TRANSCRIBE_OK);
    g_run_calls = 0;

    CHECK(transcribe_langid_run(nullptr, f.pcm.data(), 16000, nullptr) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_langid_run(f.session, nullptr, 16000, nullptr) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_langid_run(f.session, f.pcm.data(), 0, nullptr) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_langid_run(f.session, f.pcm.data(), -5, nullptr) == TRANSCRIBE_ERR_INVALID_ARG);
    transcribe_langid_params p;
    transcribe_langid_params_init(&p);
    p.top_k = -1;
    CHECK(f.run(&p) == TRANSCRIBE_ERR_INVALID_ARG);
    p.top_k       = 0;
    p.struct_size = 0;
    CHECK(f.run(&p) == TRANSCRIBE_ERR_BAD_STRUCT_SIZE);
    CHECK(g_run_calls == 0);
    CHECK(f.result().n_candidates == 5);

    // A family failure after the commit point clears the result.
    g_run_status = TRANSCRIBE_ERR_OOM;
    CHECK(f.run() == TRANSCRIBE_ERR_OOM);
    CHECK(f.result().n_candidates == 0 && f.result().audio_ms == 0);
    g_run_status = TRANSCRIBE_OK;

    // The abort callback reaches the family.
    CHECK(f.run() == TRANSCRIBE_OK);
    g_check_abort = true;
    transcribe_langid_set_abort_callback(f.session, [](void *) { return true; }, nullptr);
    CHECK(f.run() == TRANSCRIBE_ERR_ABORTED);
    CHECK(f.result().n_candidates == 0);
    transcribe_langid_set_abort_callback(f.session, nullptr, nullptr);
    CHECK(f.run() == TRANSCRIBE_OK);
    transcribe_langid_set_abort_callback(nullptr, nullptr, nullptr);
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
    test_accessors_before_run();
    test_unrestricted_ranking();
    test_allowed_set();
    test_crop_and_minimum();
    test_non_finite();
    test_bad_args_and_failures();

    if (g_failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return EXIT_FAILURE;
    }
    std::printf("langid_dispatch_unit: ok\n");
    return EXIT_SUCCESS;
}

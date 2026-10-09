// Public .bin loading: stored-weight parity and named parser safety regressions.
#include "ggml.h"
#include "gguf.h"
#include "transcribe.h"
#include "transcribe/vad.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {
int          failures  = 0;
const char * test_case = "fixture";
#define CHECK(c)                                                                 \
    do {                                                                         \
        if (!(c)) {                                                              \
            std::fprintf(stderr, "FAIL %d [%s]: %s\n", __LINE__, test_case, #c); \
            ++failures;                                                          \
        }                                                                        \
    } while (0)
using Bytes = std::vector<uint8_t>;

struct Tensor {
    std::string          source, target;
    std::vector<int32_t> shape;
    std::vector<float>   values;
};

void u32(Bytes & b, uint32_t v) {
    for (int i = 0; i < 4; ++i) {
        b.push_back(static_cast<uint8_t>(v >> (8 * i)));
    }
}

void patch(Bytes & b, size_t off, uint32_t v) {
    for (int i = 0; i < 4; ++i) {
        b[off + i] = static_cast<uint8_t>(v >> (8 * i));
    }
}

void text(Bytes & b, const std::string & s) {
    b.insert(b.end(), s.begin(), s.end());
}

std::vector<Tensor> tensors() {
    std::vector<Tensor> t;
    const int32_t       ch[] = { 129, 128, 64, 64, 128 };
    for (int i = 0; i < 4; ++i) {
        const std::string src = "_model.encoder." + std::to_string(i) + ".reparam_conv.";
        const std::string dst = "enc." + std::to_string(i) + ".conv.";
        t.push_back({
            src + "weight", dst + "weight", { 3, ch[i], ch[i + 1] },
                  {}
        });
        t.push_back({ src + "bias", dst + "bias", { ch[i + 1] }, {} });
    }
    for (const char * name : { "weight_ih", "weight_hh", "bias_ih", "bias_hh" }) {
        const bool weight = name[0] == 'w';
        t.push_back({
            std::string("_model.decoder.rnn.") + name,
            std::string("lstm.") + name,
            weight ? std::vector<int32_t>{ 128, 512 }
              : std::vector<int32_t>{ 512 },
            {}
        });
    }
    t.push_back({ "_model.decoder.decoder.2.weight", "head.weight", { 128 }, {} });
    t.push_back({ "_model.decoder.decoder.2.bias", "head.bias", {}, {} });
    t.push_back({
        "_model.stft.forward_basis_buffer", "frontend.stft_basis", { 256, 1, 258 },
          {}
    });
    size_t slot = 0;
    for (Tensor & x : t) {
        size_t n = 1;
        for (int32_t d : x.shape) {
            n *= static_cast<size_t>(d);
        }
        for (size_t j = 0; j < n; ++j) {
            // Distinct per tensor, F16-exact, and not a regenerated STFT basis.
            x.values.push_back((static_cast<int>((13 * j + 7 * slot) % 31) - 15) / 128.0f);
        }
        ++slot;
    }
    t[13].values[0] = -0.25f;
    return t;
}

Bytes header(bool v5) {
    Bytes b;
    u32(b, 0x67676d6c);
    u32(b, 10);
    text(b, "silero-16k");
    u32(b, v5 ? 5 : 6);
    u32(b, v5 ? 1 : 2);
    u32(b, v5 ? 2 : 0);
    for (uint32_t v :
         { 512u, 64u, 4u, 129u, 128u, 3u, 128u, 64u, 3u, 64u, 64u, 3u, 64u, 128u, 3u, 128u, 128u, 128u, 1u }) {
        u32(b, v);
    }
    return b;
}

Bytes record(const Tensor & t, int type) {
    Bytes b;
    u32(b, static_cast<uint32_t>(t.shape.size()));
    u32(b, static_cast<uint32_t>(t.source.size()));
    u32(b, static_cast<uint32_t>(type));
    for (int32_t d : t.shape) {
        u32(b, static_cast<uint32_t>(d));
    }
    text(b, t.source);
    for (float v : t.values) {
        if (type == 1) {
            const uint16_t h = ggml_fp32_to_fp16(v);
            b.push_back(static_cast<uint8_t>(h));
            b.push_back(static_cast<uint8_t>(h >> 8));
        } else {
            uint32_t bits;
            std::memcpy(&bits, &v, 4);
            u32(b, bits);
        }
    }
    return b;
}

Bytes binary(const std::vector<Tensor> & t, bool v5) {
    Bytes b = header(v5);
    for (size_t i = 0; i < t.size(); ++i) {
        const bool  half = t[i].shape.size() == 3 || t[i].target == "head.weight";
        const Bytes rec  = record(t[i], half ? 1 : 0);
        b.insert(b.end(), rec.begin(), rec.end());
    }
    return b;
}

void write(const std::filesystem::path & p, const Bytes & b) {
    std::ofstream f(p, std::ios::binary);
    f.write(reinterpret_cast<const char *>(b.data()), static_cast<std::streamsize>(b.size()));
    CHECK(f.good());
}

void gguf(const std::filesystem::path & p, const std::vector<Tensor> & tensors) {
    ggml_init_params ip{};
    ip.mem_size        = 15 * ggml_tensor_overhead();
    ip.no_alloc        = true;
    ggml_context * ctx = ggml_init(ip);
    gguf_context * g   = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "silero_vad");
    const char *   keys[] = { "stt.frontend.sample_rate",       "stt.frontend.n_fft",
                              "stt.frontend.hop_length",        "stt.vad.frame_samples",
                              "stt.silero_vad.context_samples", "stt.silero_vad.reflect_pad",
                              "stt.silero_vad.lstm_hidden" };
    const uint32_t vals[] = { 16000, 256, 128, 512, 64, 64, 128 };
    for (int i = 0; i < 7; ++i) {
        gguf_set_val_u32(g, keys[i], vals[i]);
    }
    const int32_t ch[] = { 129, 128, 64, 64, 128 }, stride[] = { 1, 2, 2, 1 };
    gguf_set_arr_data(g, "stt.silero_vad.encoder_channels", GGUF_TYPE_INT32, ch, 5);
    gguf_set_arr_data(g, "stt.silero_vad.encoder_strides", GGUF_TYPE_INT32, stride, 4);
    for (const Tensor & x : tensors) {
        int64_t ne[] = { 1, 1, 1 };
        for (size_t i = 0; i < x.shape.size(); ++i) {
            ne[i] = x.shape[i];
        }
        if (x.target == "frontend.stft_basis") {
            ne[1] = 258;
            ne[2] = 1;
        }
        ggml_tensor * t = ggml_new_tensor(ctx, GGML_TYPE_F32, 3, ne);
        ggml_set_name(t, x.target.c_str());
        t->data = const_cast<float *>(x.values.data());
        gguf_add_tensor(g, t);
    }
    CHECK(gguf_write_to_file(g, p.string().c_str(), false));
    gguf_free(g);
    ggml_free(ctx);
}

transcribe_model * load(const std::filesystem::path & p, transcribe_model_load_params * mp = nullptr) {
    transcribe_model * m = nullptr;
    CHECK(transcribe_model_load_file(p.string().c_str(), mp, &m) == TRANSCRIBE_OK);
    CHECK(m != nullptr);
    return m;
}

std::vector<float> run(transcribe_model * m) {
    transcribe_vad_session_params sp;
    transcribe_vad_session_params_init(&sp);
    sp.n_threads               = 1;
    transcribe_vad_session * s = nullptr;
    CHECK(transcribe_vad_session_init(m, &sp, &s) == TRANSCRIBE_OK);
    if (!s) {
        return {};
    }
    std::vector<float> pcm(8193);
    for (size_t i = 0; i < pcm.size(); ++i) {
        pcm[i] = 0.7f * std::sin(static_cast<float>(i) * 0.1f);
    }
    CHECK(transcribe_vad_run(s, pcm.data(), static_cast<int>(pcm.size()), nullptr) == TRANSCRIBE_OK);
    transcribe_vad_result r;
    transcribe_vad_result_init(&r);
    CHECK(transcribe_vad_get_result(s, &r) == TRANSCRIBE_OK);
    const float *      out = transcribe_vad_probs(s);
    std::vector<float> probs;
    if (out) {
        probs.assign(out, out + r.n_probs);
    }
    CHECK(probs.size() == 17);
    for (float p : probs) {
        CHECK(std::isfinite(p) && p >= 0 && p <= 1);
    }
    transcribe_vad_session_free(s);
    return probs;
}

void malformed(const char * name, const std::filesystem::path & path, const Bytes & bytes) {
    test_case = name;
    write(path, bytes);
    transcribe_model * m = reinterpret_cast<transcribe_model *>(uintptr_t(1));
    CHECK(transcribe_model_load_file(path.string().c_str(), nullptr, &m) != TRANSCRIBE_OK);
    CHECK(m == nullptr);
}
}  // namespace

int main() {
    transcribe_log_set(nullptr, nullptr);
    const std::filesystem::path dir = TRANSCRIBE_TEST_BIN_DIR;
    std::filesystem::create_directories(dir);
    const auto t     = tensors();
    const auto path  = dir / "weights.not-bin";
    const auto gpath = dir / "weights.not-gguf";
    gguf(gpath, t);
    transcribe_model * g = load(gpath);
    if (!g) {
        return 1;
    }
    const auto expected = run(g);
    CHECK(!expected.empty() &&
          *std::min_element(expected.begin(), expected.end()) != *std::max_element(expected.begin(), expected.end()));
    for (bool v5 : { false, true }) {
        test_case = v5 ? "v5-mixed-weight-parity" : "v6-mixed-weight-parity";
        write(path, binary(t, v5));
        transcribe_model * m = load(path);
        if (!m) {
            continue;
        }
        CHECK(std::string(transcribe_model_backend(m)) == "CPU");
        CHECK(transcribe_model_roles(m) == TRANSCRIBE_ROLE_VAD);
        CHECK(std::string(transcribe_model_variant_string(m)) == (v5 ? "silero-vad-v5.1.2" : "silero-vad-v6.2.0"));
        CHECK(std::string(transcribe_model_meta_val_str(m, "general.version")) == (v5 ? "5.1.2" : "6.2.0"));
        CHECK(expected == run(m));
        transcribe_model_free(m);
    }

    const Bytes  valid  = binary(t, false);
    const size_t first  = header(false).size();
    size_t       scalar = first;
    for (size_t i = 0; i < 13; ++i) {
        scalar += record(t[i], (t[i].shape.size() == 3 || t[i].target == "head.weight") ? 1 : 0).size();
    }
    const size_t scalar_payload = scalar + 12 + t[13].source.size();

    const struct {
        const char * name;
        size_t       end;
    } truncated[] = {
        { "short-magic",          3                  },
        { "short-geometry",       first - 1          },
        { "short-record-header",  first + 11         },
        { "short-tensor-name",    first + 24         },
        { "short-scalar-payload", scalar_payload + 3 },
        { "short-final-payload",  valid.size() - 1   },
    };

    for (const auto & c : truncated) {
        malformed(c.name, path, Bytes(valid.begin(), valid.begin() + c.end));
    }

    const struct {
        const char * name;
        size_t       offset;
        uint32_t     value;
    } invalid[] = {
        { "bad-magic",              0,              0           },
        { "unbounded-tag",          4,              0xffffffffu },
        { "bad-tag",                8,              0           },
        { "unsupported-version",    18,             7           },
        { "unsupported-patch",      26,             1           },
        { "wrong-frame",            30,             513         },
        { "wrong-encoder-channels", 42,             0           },
        { "wrong-lstm-size",        94,             64          },
        { "unsupported-rank",       first,          4           },
        { "empty-name",             first + 4,      0           },
        { "unsupported-type",       first + 8,      2           },
        { "unsafe-dimension",       first + 12,     0x7fffffffu },
        { "unknown-tensor",         first + 24,     0           },
        { "nonfinite-f32",          scalar_payload, 0x7fc00000  },
    };

    for (const auto & c : invalid) {
        Bytes b = valid;
        patch(b, c.offset, c.value);
        malformed(c.name, path, b);
    }
    Bytes        b            = valid;
    const size_t half_payload = first + 24 + t[0].source.size();
    b[half_payload]           = 0;
    b[half_payload + 1]       = 0x7e;
    malformed("nonfinite-f16", path, b);
    b = valid;
    b.push_back(0);
    malformed("trailing-partial-record", path, b);
    const Bytes duplicate = record(t[13], 0);
    b                     = valid;
    b.insert(b.end(), duplicate.begin(), duplicate.end());
    malformed("duplicate-tensor", path, b);
    auto missing = t;
    missing.erase(missing.begin() + 13);
    malformed("missing-required-tensor", path, binary(missing, false));
    auto wrong_shape = t;
    std::swap(wrong_shape[0].shape[0], wrong_shape[0].shape[1]);
    malformed("same-size-wrong-shape", path, binary(wrong_shape, false));
    wrong_shape           = t;
    wrong_shape[13].shape = { 1 };
    malformed("scalar-bias-is-not-a-vector", path, binary(wrong_shape, false));

    transcribe_model_free(g);
    std::filesystem::remove_all(dir);
    std::printf("silero_vad_bin_smoke: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}

// voxtral_tokenize_parity.cpp - real-model gated test that
// transcribe_tokenize() on a Voxtral GGUF produces the same token ids as
// mistral-common's Tekkenizer.
//
// Expected ids: Tekkenizer.from_file(tekken.json).encode(text, bos=False,
// eos=False) for Voxtral-Mini-3B-2507.
//
// Gated by TRANSCRIBE_VOXTRAL_GGUF. Exits 77 (cmake SKIP_RETURN_CODE)
// when unset.

#include "transcribe.h"

#include <sys/stat.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

bool file_exists(const std::string & path) {
    struct stat st{};
    return ::stat(path.c_str(), &st) == 0;
}

struct Case {
    const char *         text;
    std::vector<int32_t> expected;
};

void check_case(const struct transcribe_model * model, const Case & c) {
    int32_t   tokens[64];
    const int n = transcribe_tokenize(model, c.text, tokens, 64);
    if (n < 0 || static_cast<size_t>(n) != c.expected.size()) {
        std::fprintf(stderr, "FAIL tokenize(%s): n=%d, expected size %zu\n", c.text, n, c.expected.size());
        ++g_failures;
        return;
    }
    for (int i = 0; i < n; ++i) {
        if (tokens[i] != c.expected[i]) {
            std::fprintf(stderr, "FAIL tokenize(%s): tokens[%d]=%d, expected %d\n", c.text, i, tokens[i],
                         c.expected[i]);
            ++g_failures;
            return;
        }
    }
}

}  // namespace

int main() {
    const char * env = std::getenv("TRANSCRIBE_VOXTRAL_GGUF");
    if (env == nullptr || env[0] == '\0') {
        std::fprintf(stderr,
                     "voxtral_tokenize_parity: TRANSCRIBE_VOXTRAL_GGUF "
                     "not set; skipping.\n");
        return 77;
    }
    const std::string model_path = env;
    if (!file_exists(model_path)) {
        std::fprintf(stderr, "voxtral_tokenize_parity: model not found: %s\n", model_path.c_str());
        return 77;
    }

    transcribe_model_load_params mp;
    transcribe_model_load_params_init(&mp);
    mp.backend                      = TRANSCRIBE_BACKEND_CPU;
    struct transcribe_model * model = nullptr;
    const transcribe_status   st    = transcribe_model_load_file(model_path.c_str(), &mp, &model);
    if (st != TRANSCRIBE_OK || model == nullptr) {
        std::fprintf(stderr, "FAIL load: %s\n", transcribe_status_string(st));
        return EXIT_FAILURE;
    }

    const std::vector<Case> cases = {
        // The language hint build_transcription_prompt encodes.
        { "lang:de",                                                                     { 9909, 1058, 1558 }                              },
        { "Transcribe this audio.",                                                      { 6881, 13089, 1593, 16023, 1046 }                },
        { "HTTPServer iPhone McDonald",                                                  { 30499, 11473, 1623, 16742, 5303, 31609 }        },
        { u8"\u0928\u092e\u0938\u094d\u0924\u0947 \u0926\u0941\u0928\u093f\u092f\u093e",
         { 2485, 3525, 22475, 1803, 115801 }                                                                                               },
        { u8"\u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e35\u0e04\u0e23\u0e31\u0e1a",
         { 8335, 61695, 8335, 64836, 38686, 21941 }                                                                                        },
        { u8"\u00dcber Publikum na\u00efve \u00d6l\u00c4nderung",                        { 79904, 99618, 98355, 86231, 86169, 8793, 1851 } },
        { u8"e\u0301cole",                                                               { 1101, 1204, 1129, 17242 }                       },
        { "http://example.com/a/b\n",                                                    { 3809, 2345, 16609, 2354, 22139, 15836, 1010 }   },
        { "3rd 1234",                                                                    { 1051, 7989, 1032, 1049, 1050, 1051, 1052 }      },
        { "  multi  spaces  ",                                                           { 1032, 8549, 1032, 18971, 1256 }                 },
        { "",                                                                            {}                                                },
    };
    for (const Case & c : cases) {
        check_case(model, c);
    }

    transcribe_model_free(model);

    if (g_failures > 0) {
        std::fprintf(stderr, "voxtral_tokenize_parity: %d failures\n", g_failures);
        return EXIT_FAILURE;
    }
    std::fprintf(stdout, "voxtral_tokenize_parity: ok\n");
    return EXIT_SUCCESS;
}

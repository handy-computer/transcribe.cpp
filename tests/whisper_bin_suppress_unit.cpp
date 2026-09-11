// Regression coverage for legacy Whisper .bin suppression metadata.
// The format omits generation_config, so the adapter must choose ids for the
// tokenizer family and compute control-token shifts from the vocabulary size.

#include "arch/whisper/bin_load.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
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

const int32_t k_english_expected[] = {
    1,     2,     7,     8,     9,     10,    14,    25,    26,    27,    28,    29,    31,    58,    59,
    60,    61,    62,    63,    90,    91,    92,    93,    357,   366,   438,   532,   685,   705,   796,
    930,   1058,  1220,  1267,  1279,  1303,  1343,  1377,  1391,  1635,  1782,  1875,  2162,  2361,  2488,
    3467,  4008,  4211,  4600,  4808,  5299,  5855,  6329,  7203,  9609,  9959,  10563, 10786, 11420, 11709,
    11907, 13163, 13697, 13700, 14808, 15306, 16410, 16791, 17992, 19203, 19510, 20724, 22305, 22935, 27007,
    30109, 30420, 33409, 34949, 40283, 40493, 40549, 47282, 49146, 50257, 50357, 50358, 50359, 50360, 50361,
};

const int32_t k_multilingual_expected[] = {
    1,     2,     7,     8,     9,     10,    14,    25,    26,    27,    28,    29,    31,    58,    59,
    60,    61,    62,    63,    90,    91,    92,    93,    359,   503,   522,   542,   873,   893,   902,
    918,   922,   931,   1350,  1853,  1982,  2460,  2627,  3246,  3253,  3268,  3536,  3846,  3961,  4183,
    4667,  6585,  6647,  7273,  9061,  9383,  10428, 10929, 11938, 12033, 12331, 12562, 13793, 14157, 14635,
    15265, 15618, 16553, 16604, 18362, 18956, 20075, 21675, 22520, 26130, 26161, 26435, 28279, 29464, 31650,
    32302, 32470, 36865, 42863, 47425, 49870, 50254, 50258, 50358, 50359, 50360, 50361, 50362,
};

template <size_t N> void check_exact(const std::vector<int32_t> & actual, const int32_t (&expected)[N]) {
    CHECK(actual.size() == N);
    if (actual.size() != N) {
        return;
    }
    for (size_t i = 0; i < N; ++i) {
        if (actual[i] != expected[i]) {
            std::fprintf(stderr, "FAIL suppression id %zu: got %d, expected %d\n", i, actual[i], expected[i]);
            ++g_failures;
        }
    }
}

void test_english() {
    const std::vector<int32_t> actual = transcribe::whisper::synthesize_bin_suppress_tokens(false, 51864);
    check_exact(actual, k_english_expected);

    // These ids are ordinary English word pieces under the multilingual list.
    CHECK(std::find(actual.begin(), actual.end(), 922) == actual.end());   // " good"
    CHECK(std::find(actual.begin(), actual.end(), 2627) == actual.end());  // " became"
    CHECK(std::find(actual.begin(), actual.end(), 357) != actual.end());
}

void test_multilingual() {
    const std::vector<int32_t> actual = transcribe::whisper::synthesize_bin_suppress_tokens(true, 51865);
    check_exact(actual, k_multilingual_expected);
}

void test_large_v3_special_shift() {
    const std::vector<int32_t> actual = transcribe::whisper::synthesize_bin_suppress_tokens(true, 51866);
    CHECK(actual.size() == 88);

    const int32_t expected_tail[] = { 50258, 50359, 50360, 50361, 50362, 50363 };
    CHECK(actual.size() >= 6);
    if (actual.size() >= 6) {
        for (size_t i = 0; i < 6; ++i) {
            CHECK(actual[actual.size() - 6 + i] == expected_tail[i]);
        }
    }
}

}  // namespace

int main() {
    test_english();
    test_multilingual();
    test_large_v3_special_shift();

    if (g_failures != 0) {
        std::fprintf(stderr, "FAILED: %d check(s)\n", g_failures);
        return 1;
    }
    std::fprintf(stderr, "OK\n");
    return 0;
}

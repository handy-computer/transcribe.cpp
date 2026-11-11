#include "arch/whisper/alignment.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string_view>
#include <vector>

namespace {

using transcribe::whisper::align_attention;
using transcribe::whisper::alignment_head;
using transcribe::whisper::alignment_heads_for_variant;

int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

void test_presets() {
    const struct {
        std::string_view variant;
        int              n_layers;
        int              n_heads;
        size_t           count;
    } presets[] = {
        { "whisper-tiny",           4,  6,  6  },
        { "whisper-tiny.en",        4,  6,  8  },
        { "whisper-base",           6,  8,  8  },
        { "whisper-base.en",        6,  8,  5  },
        { "whisper-small",          12, 12, 10 },
        { "whisper-small.en",       12, 12, 19 },
        { "whisper-medium",         24, 16, 6  },
        { "whisper-medium.en",      24, 16, 18 },
        { "whisper-large",          32, 20, 9  },
        { "whisper-large-v1",       32, 20, 9  },
        { "whisper-large-v2",       32, 20, 23 },
        { "whisper-large-v3",       32, 20, 10 },
        { "whisper-large-v3-turbo", 4,  20, 6  },
    };

    for (const auto & preset : presets) {
        const auto heads = alignment_heads_for_variant(preset.variant, preset.n_layers, preset.n_heads);
        CHECK(heads.size() == preset.count);
        int prev_index = -1;
        for (const alignment_head & head : heads) {
            CHECK(head.layer >= 0 && head.layer < preset.n_layers);
            CHECK(head.head >= 0 && head.head < preset.n_heads);
            const int index = head.layer * preset.n_heads + head.head;
            CHECK(index > prev_index);
            prev_index = index;
        }
        CHECK(alignment_heads_for_variant(preset.variant, preset.n_layers + 1, preset.n_heads).empty());
        CHECK(alignment_heads_for_variant(preset.variant, preset.n_layers, preset.n_heads + 1).empty());
    }

    const auto           base_en    = alignment_heads_for_variant("whisper-base.en", 6, 8);
    const alignment_head expected[] = {
        { 3, 3 },
        { 4, 7 },
        { 5, 1 },
        { 5, 5 },
        { 5, 7 }
    };
    if (base_en.size() == 5) {
        for (size_t i = 0; i < base_en.size(); ++i) {
            CHECK(base_en[i].layer == expected[i].layer && base_en[i].head == expected[i].head);
        }
    }
    CHECK(alignment_heads_for_variant("whisper", 6, 8).empty());
    CHECK(alignment_heads_for_variant("breeze-asr-25", 32, 20).empty());
}

void test_reference_attention() {
    // OpenAI Whisper v20250625, Base.en, first 12,800 samples of samples/jfk.wav
    // at 16 kHz. Decoder input is SOT, no_timestamps, " And", " so", EOT.
    // Two preset heads are softmaxed over the 40 valid frames and stored as
    // round(probability * 65535). OpenAI's normalization, median filter and DTW
    // give the same boundaries before and after quantizing this fixture.
    constexpr uint16_t attention_packed[2][5][40] = {
        // Decoder (layer, head) (3, 3).
        {
         {
                5707, 4695, 6863, 3603, 2143, 1492, 1443, 2335, 763, 894, 627, 989, 1541, 2542,
                2676, 3656, 3033, 2988, 2528, 1948, 1160, 845,  529, 395, 479, 504, 467,  674,
                725,  625,  619,  649,  723,  781,  870,  677,  665, 550, 521, 608,
            }, {
                2119, 1663, 4720, 3429, 2347, 2468, 2075, 3843, 1363, 1464, 1211, 1184, 1736, 2994,
                3242, 4185, 4251, 3723, 3910, 3050, 2124, 1535, 874,  511,  443,  522,  325,  344,
                289,  284,  451,  572,  447,  328,  366,  413,  302,  174,  125,  129,
            }, {
                125,  101,  184,  171,  78,   44,    50,    63,   29,   33,   61,  72,  101, 114,
                98,   44,   28,   25,   25,   24,    28,    43,   41,   27,   16,  35,  441, 3625,
                4219, 4113, 4135, 4742, 8874, 10027, 11708, 6401, 3304, 1210, 784, 292,
            }, {
                577, 195,  481,  227,  122, 71,  67,  144, 58, 68, 64, 59, 69, 96,  170, 368, 632,  901,  667,  657,
                891, 1167, 1253, 1261, 965, 805, 293, 80,  34, 12, 12, 17, 38, 107, 181, 850, 3534, 8257, 9838, 30245,
            }, {
                7090, 2075, 4603, 2117, 1140, 916,  951,  1161, 430, 416,  436,  595,   712,  1168,
                1928, 2201, 1453, 1172, 1029, 1054, 1145, 985,  830, 833,  813,  1258,  1164, 719,
                350,  202,  190,  138,  124,  189,  367,  457,  693, 1665, 2413, 18356,
            }, },
        // Decoder (layer, head) (5, 1).
        {
         {
                1622, 2711, 12745, 1913, 472,  602, 769, 1112, 613, 1032, 716, 1554, 3134, 6150,
                7885, 7791, 5882,  2737, 1511, 709, 357, 250,  234, 194,  268, 416,  374,  231,
                202,  114,  178,   199,  154,  79,  84,  97,   78,  88,   94,  187,
            },   {
                5094, 2289, 1595, 1949,  487,  245,  240,  526,  401,  589, 836, 1013, 1004, 864,
                823,  1590, 4021, 10059, 6158, 3502, 1704, 864,  503,  428, 482, 631,  773,  585,
                644,  712,  718,  1033,  1306, 1604, 2965, 1996, 1228, 777, 996, 2298,
            },                {
                228,  126,  139,  212,  59,   30,   27,   81,   35,   44,   34,   33,   32,   21,
                16,   32,   89,   338,  508,  706,  936,  1263, 1576, 1944, 3221, 5347, 6303, 9402,
                8239, 7149, 2712, 1879, 3017, 3358, 3212, 1411, 749,  294,  314,  417,
            },                             {
                788, 349, 325, 346, 95, 48, 46, 150, 50, 78, 55, 84, 87,  98,  68,  87,   154,  332,  335,   349,
                254, 229, 174, 123, 99, 66, 34, 29,  34, 52, 59, 78, 149, 267, 623, 1100, 2022, 4657, 12960, 38603,
            },                             {
                4373, 2307, 1982, 3439, 833,  453,  407,  760,  423,  543,  483,  595,   524, 525,
                496,  776,  1415, 2114, 1597, 1241, 776,  549,  376,  325,  335,  314,   251, 326,
                490,  662,  573,  584,  696,  980,  1920, 2641, 3397, 3580, 5771, 15703,
            }, },
    };

    std::vector<float> attention;
    for (const auto & head : attention_packed) {
        for (const auto & row : head) {
            for (uint16_t weight : row) {
                attention.push_back(weight / 65535.0f);
            }
        }
    }
    std::vector<int> frames;
    CHECK(align_attention(attention, 2, 5, 40, 1, 3, frames));
    CHECK((frames == std::vector<int>{ 0, 22, 36 }));
}

void test_short_attention() {
    std::vector<int>         frames;
    // Two identical rows still occupy distinct DTW rows and may share a frame.
    const std::vector<float> repeated = { 0.875f, 0.125f, 0.125f, 0.875f, 0.125f, 0.875f };
    CHECK(align_attention(repeated, 1, 3, 2, 0, 3, frames));
    CHECK((frames == std::vector<int>{ 0, 1, 1 }));

    const std::vector<float> one_frame = { 1.0f, 1.0f, 1.0f };
    CHECK(align_attention(one_frame, 1, 3, 1, 0, 3, frames));
    CHECK((frames == std::vector<int>{ 0, 0, 0 }));
}

void test_flat_attention() {
    // A zero-variance head contributes zero to all cells. Frame advances win
    // DTW cost ties, so the path visits every token in the first frame.
    const std::vector<float> flat(2 * 4 * 8, 0.125f);
    std::vector<int>         frames;
    CHECK(align_attention(flat, 2, 4, 8, 1, 3, frames));
    CHECK((frames == std::vector<int>{ 0, 0, 0 }));
}

void test_tied_predecessors() {
    // At the center of the second row, the diagonal and token predecessors
    // have equal minimum cost. Choosing the frame predecessor would lose it.
    const std::vector<float> attention = { 0.625f, 0.25f, 0.125f, 0.125f, 0.25f, 0.625f };
    std::vector<int>         frames;
    CHECK(align_attention(attention, 1, 2, 3, 0, 2, frames));
    CHECK((frames == std::vector<int>{ 0, 1 }));
}

void test_invalid_input() {
    std::vector<int> frames = { 123 };
    CHECK(!align_attention({}, 1, 1, 1, 0, 1, frames));
    CHECK(frames.empty());
    CHECK(!align_attention({ 1.0f, 1.0f }, 1, 1, 1, 0, 1, frames));
    CHECK(!align_attention({ 1.0f }, 0, 1, 1, 0, 1, frames));
    CHECK(!align_attention({ 1.0f }, 1, 1, 0, 0, 1, frames));
    CHECK(!align_attention({ 1.0f }, 1, 1, 1, -1, 1, frames));
    CHECK(!align_attention({ 1.0f }, 1, 1, 1, 1, 1, frames));
    CHECK(!align_attention({ 1.0f }, 1, 1, 1, 0, 0, frames));
    CHECK(!align_attention({ 1.0f }, 1, std::numeric_limits<int>::max(), 2, 0, 1, frames));
    CHECK(!align_attention({ std::numeric_limits<float>::quiet_NaN(), 1.0f }, 1, 2, 1, 1, 1, frames));
    CHECK(!align_attention({ -0.01f }, 1, 1, 1, 0, 1, frames));
    CHECK(!align_attention({ 1.01f }, 1, 1, 1, 0, 1, frames));
}

}  // namespace

int main() {
    test_presets();
    test_reference_attention();
    test_short_attention();
    test_flat_attention();
    test_tied_predecessors();
    test_invalid_input();

    if (g_failures != 0) {
        std::fprintf(stderr, "FAILED: %d check(s)\n", g_failures);
        return 1;
    }
    std::fprintf(stderr, "OK\n");
    return 0;
}

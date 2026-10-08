// Unit test for the parakeet long-form segmenter. Expected values come from
// kestrel run on the synthetic inputs below; the port must match exactly.

#include "arch/parakeet/longform.h"

#include <algorithm>
#include <cstdio>
#include <utility>
#include <vector>

namespace lf = transcribe::parakeet::longform;

namespace {

const float kProbs[] = {
    0.4500615f,    0.541782081f,  0.589053869f,   0.632300556f,  0.725424111f,  0.766298711f,  0.882188976f,
    1.0f,          0.964906871f,  1.0f,           1.0f,          1.0f,          1.0f,          1.0f,
    1.0f,          1.0f,          1.0f,           1.0f,          0.997145414f,  0.993273318f,  0.923860729f,
    0.955549121f,  0.849034488f,  0.865483701f,   0.794413388f,  0.707816124f,  0.518682182f,  0.542591631f,
    0.490662545f,  0.42179355f,   0.263071239f,   0.240475029f,  0.142465368f,  0.0811041072f, 0.108715832f,
    0.0f,          0.0f,          0.0f,           0.0f,          0.0f,          0.0f,          0.0f,
    0.0f,          0.0f,          0.0f,           0.0f,          0.0f,          0.0f,          0.0f,
    0.0046699564f, 0.200000003f,  0.200000003f,   0.0800082237f, 0.172352418f,  0.205019042f,  0.322391063f,
    0.360662341f,  0.474132925f,  0.589590609f,   0.559957325f,  0.678235888f,  0.716615975f,  0.814344347f,
    0.812493563f,  0.901661336f,  0.973779976f,   1.0f,          1.0f,          1.0f,          1.0f,
    1.0f,          1.0f,          1.0f,           1.0f,          1.0f,          1.0f,          1.0f,
    1.0f,          0.960966825f,  0.995472193f,   0.838040769f,  0.779455125f,  0.743299723f,  0.647330642f,
    0.568745613f,  0.44659245f,   0.424773604f,   0.32652244f,   0.331584215f,  0.232644737f,  0.128517687f,
    0.0967499092f, 0.0f,          0.0f,           0.0f,          0.0f,          0.0f,          0.0f,
    0.0f,          0.0f,          0.0f,           0.0f,          0.0f,          0.0f,          0.0f,
    0.0f,          0.0f,          0.0f,           0.0f,          0.0582143404f, 0.170369923f,  0.231782749f,
    0.227596164f,  0.350877106f,  0.433307052f,   0.455748111f,  0.597687185f,  0.616392076f,  0.77995801f,
    0.809655845f,  0.5f,          0.49999997f,    0.971639037f,  0.925000072f,  1.0f,          1.0f,
    1.0f,          1.0f,          1.0f,           1.0f,          1.0f,          1.0f,          1.0f,
    0.989855468f,  1.0f,          1.0f,           0.9228127f,    0.853196561f,  0.794649124f,  0.685372293f,
    0.717112184f,  0.560526371f,  0.508935392f,   0.394910723f,  0.32653895f,   0.218337402f,  0.271486223f,
    0.130255073f,  0.119354121f,  0.00940343179f, 0.0f,          0.0f,          0.0f,          0.0f,
    0.0f,          0.0f,          0.0f,           0.0f,          0.0f,          0.0f,          0.0f,
    0.0f,          0.0f,          0.0f,           0.0f,          0.0f,          0.0f,          0.163776666f,
    0.195986852f,  0.273916125f,  0.308085114f,   0.445057273f,  0.472305208f,  0.568481386f,  0.595051348f,
    0.66207397f,   0.85877192f,   0.831622362f,   0.931624353f,  0.969768107f,  0.997514367f,  1.0f,
    1.0f,          1.0f,          1.0f,           1.0f,          1.0f,          1.0f,          1.0f,
    1.0f,          1.0f,          1.0f,           1.0f,          0.925792038f,  0.901364446f,  0.849656701f,
    0.7840994f,    0.717028797f,  0.573927283f,   0.596431971f,  0.381474435f,  0.410085887f,  0.315895259f,
    0.26099813f,   0.240198702f,  0.15307337f,    0.0f,          0.0f,          0.0f,          0.0f,
    0.0f,          0.0f,          0.0f,           0.0f,          0.0f,          0.0f,          0.0f,
    0.0f,          0.0f,          0.0f,           0.0f,          0.0f,          0.0f,          0.0784521028f,
    0.115800112f,  0.208901092f,  0.211974606f,   0.30368039f,   0.363146096f,  0.44572714f,   0.576306403f,
    0.602455378f,  0.732159436f,  0.800900459f,   0.950690985f,  0.840485454f,  1.0f,          1.0f,
    1.0f,          1.0f,          1.0f,           1.0f,          1.0f,          1.0f,          1.0f,
    1.0f,          1.0f,          0.965194583f,   1.0f,          0.893005252f,  0.776896119f,  0.855068624f,
    0.885141909f,  0.753255546f,  0.621070743f,   0.558684886f,  0.586390138f,  0.460915029f,  0.378571242f,
    0.297528207f,  0.227962047f,  0.194860205f,   0.114337049f,  0.0340040736f, 0.0f,          0.0f,
    0.0f,          0.0f,          0.0f,           0.0f,          0.0f,          0.0f,          0.0f,
    0.0f,          0.0f,          0.0f,           0.0f,          0.0f,          0.0f,          0.0f,
    0.0497603379f, 0.0569235757f, 0.166568398f,   0.243654519f,  0.386890411f,  0.420606196f,  0.4805637f,
    0.633873701f,  0.604948103f,  0.686315656f,   0.684934199f,  0.920274138f,  0.951346159f,  1.0f,
    1.0f,          1.0f,          1.0f,           1.0f,          1.0f,          1.0f,          1.0f,
    1.0f,          1.0f,          1.0f,           1.0f,          1.0f,          1.0f,          0.949401021f,
    0.871494353f,  0.770826578f,  0.755863607f,   0.732079327f,  0.595077693f,  0.527659595f,  0.451211095f,
    0.390835404f,  0.229579404f,  0.223035902f,   0.120263144f,  0.138858676f,  0.0f,          0.000436047936f,
    0.0f,          0.0f,          0.0f,           0.0f,          0.0f,          0.0f,          0.0f,
    0.0f,          0.0f,          0.0f,           0.0f,          0.0f,          0.0f,          0.0f,
    0.042441126f,  0.0f,          0.156791449f,   0.247761667f,  0.256696194f,  0.346078932f,  0.492292017f,
    0.461781859f,  0.518645465f,  0.556123555f,   0.699259102f,  0.841531754f,  0.882116437f,  0.848950803f,
    0.909813464f,  0.977825403f,  1.0f,           1.0f,          1.0f,          1.0f,          1.0f,
    1.0f,          1.0f,          1.0f,           1.0f,          1.0f,          1.0f,          1.0f,
    0.868186772f,  0.915354133f,  0.73669523f,    0.697008312f,  0.739785373f,  0.658972323f,  0.540671527f,
    0.385989219f,  0.375983864f,  0.284346163f,   0.275449067f,  0.284312338f,  0.113458477f,  0.0f,
    0.0f,          0.0f,          0.0f,           0.0f,          0.0f,          0.0f,          0.0f,
    0.0f,          0.0f,          0.0f,           0.0f,          0.0f,          0.0f,          0.0f,
    0.0f,          0.0f,          0.0703902617f,  0.0851784274f, 0.173921421f,  0.285013676f,  0.222699255f,
    0.346277177f
};

const std::vector<std::pair<double, double>> kSpeechRegions = {
    { 0.08,               2.24  },
    { 4.64,               6.8   },
    { 9.28,               11.44 },
    { 13.84,              16.0  },
    { 18.400000000000002, 20.56 },
    { 22.96,              25.12 },
    { 27.52,              29.68 }
};

// Absolute pauses (seconds) driving the synthetic speech source.
const std::vector<std::pair<double, double>> kPauses = {
    { 11.64,  11.72  },
    { 23.35,  23.51  },
    { 33.94,  34.84  },
    { 48.25,  48.65  },
    { 55.85,  56.01  },
    { 66.31,  66.55  },
    { 78.57,  78.97  },
    { 89.01,  89.17  },
    { 93.68,  93.76  },
    { 104.99, 105.23 },
    { 108.01, 108.41 },
    { 110.6,  111.0  },
    { 124.5,  124.9  },
    { 131.81, 132.05 },
    { 142.69, 142.93 },
    { 153.7,  153.94 },
    { 156.96, 157.86 },
    { 166.56, 166.8  },
    { 179.98, 180.38 },
    { 187.82, 187.9  },
    { 197.47, 197.55 },
    { 200.44, 200.68 },
    { 209.8,  209.96 },
    { 214.3,  214.46 },
    { 227.01, 227.25 },
    { 234.7,  234.78 },
    { 245.78, 246.68 },
    { 255.32, 255.72 },
    { 267.41, 267.49 },
    { 276.93, 277.17 },
    { 289.0,  289.16 },
    { 298.86, 299.26 },
    { 306.14, 306.22 },
    { 312.97, 313.21 },
    { 324.14, 325.04 },
    { 332.64, 332.8  },
    { 343.87, 344.77 },
    { 350.82, 351.06 },
    { 362.98, 363.88 },
    { 376.02, 376.18 },
    { 387.6,  388.5  },
    { 399.06, 399.3  },
    { 401.71, 402.11 }
};

struct Case {
    int64_t                                  n_samples;
    std::vector<int64_t>                     blocks;
    std::vector<std::pair<int64_t, int64_t>> segments;  // (start, n_samples)
};

const Case kCases[] = {
    { 4800123,
     { 1920000, 1920000, 960123 },
     { { 0, 480000 },
        { 480000, 295200 },
        { 775200, 287680 },
        { 1062880, 197440 },
        { 1260320, 471040 },
        { 1731360, 379520 },
        { 2110880, 407680 },
        { 2518560, 364320 },
        { 2882880, 326080 },
        { 3208960, 425120 },
        { 3634080, 454240 },
        { 4088320, 344480 },
        { 4432800, 367323 } }                        },
    { 1925000,
     { 1917000, 8000 },
     { { 0, 480000 },
        { 480000, 295200 },
        { 775200, 287680 },
        { 1062880, 197440 },
        { 1260320, 471040 },
        { 1731360, 193640 } }                        },
    { 480001,  { 480001 },         { { 0, 480000 } } },
};

int g_failures = 0;

void check(bool ok, const char * what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_failures;
    }
}

}  // namespace

int main() {
    const lf::Params p;  // kestrel defaults: 16 kHz, 0.08 s frames, 0.5/0.1/0.1, 30/1/0.2 s, 120 s blocks

    // vad.speech_regions
    const lf::Regions got = lf::speech_regions(kProbs, static_cast<int>(sizeof(kProbs) / sizeof(kProbs[0])), p);
    check(got == lf::Regions(kSpeechRegions.begin(), kSpeechRegions.end()), "speech_regions matches kestrel");

    // fits_one_segment / round_half_even
    check(lf::fits_one_segment(480000, p) && !lf::fits_one_segment(480001, p), "fits_one_segment boundary at 30 s");
    check(lf::round_half_even(2.5) == 2 && lf::round_half_even(3.5) == 4 && lf::round_half_even(-0.5) == 0,
          "round_half_even ties to even");

    for (const Case & c : kCases) {
        char label[96];
        std::snprintf(label, sizeof(label), "scan_block_sizes(%lld)", static_cast<long long>(c.n_samples));
        check(lf::scan_block_sizes(c.n_samples, p) == c.blocks, label);

        // The same synthetic source the fixture used: speech everywhere except
        // the absolute pauses, reported per block relative to its start.
        const lf::SpeechFn speech = [&](int64_t start_sample, int64_t len, lf::Regions & regions) {
            const double start = static_cast<double>(start_sample) / p.sample_rate;
            const double dur   = static_cast<double>(len) / p.sample_rate;
            double       cur   = start;
            regions.clear();
            for (const auto & pz : kPauses) {
                if (pz.second <= start || pz.first >= start + dur) {
                    continue;
                }
                if (pz.first > cur) {
                    regions.emplace_back(cur - start, pz.first - start);
                }
                cur = std::max(cur, pz.second);
            }
            if (cur < start + dur) {
                regions.emplace_back(cur - start, dur);
            }
            return TRANSCRIBE_OK;
        };
        std::vector<lf::Segment> segs;
        check(lf::pause_segments(c.n_samples, p, speech, segs) == TRANSCRIBE_OK, "pause_segments status");
        bool same = segs.size() == c.segments.size();
        for (size_t i = 0; same && i < segs.size(); ++i) {
            same = segs[i].start == c.segments[i].first && segs[i].n_samples == c.segments[i].second;
        }
        std::snprintf(label, sizeof(label), "pause_segments(%lld) matches kestrel",
                      static_cast<long long>(c.n_samples));
        check(same, label);
    }

    if (g_failures == 0) {
        std::printf("parakeet longform unit: OK\n");
    }
    return g_failures == 0 ? 0 : 1;
}

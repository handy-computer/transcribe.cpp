// parakeet_mel_valid_length_unit.cpp - drop_nemo_padding_mel_frame (#252).
//
// The Parakeet frontend emits floor(n / hop) + 1 mel frames; NeMo counts
// floor(n / hop) of them as valid. The helper must drop exactly the trailing
// frame of every mel row, keep the row-major [n_mels, n_frames] layout, and
// leave kestrel-masked checkpoints alone.

#include "arch/parakeet/parakeet.h"

#include <cstdio>
#include <vector>

namespace {

int g_failures = 0;

void check(bool ok, const char * what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_failures;
    }
}

// Value of mel bin m at frame t: unique per cell.
float cell(int m, int t) {
    return static_cast<float>(m * 10000 + t);
}

std::vector<float> make_mel(int n_mels, int n_frames) {
    std::vector<float> mel(static_cast<size_t>(n_mels) * static_cast<size_t>(n_frames));
    for (int m = 0; m < n_mels; ++m) {
        for (int t = 0; t < n_frames; ++t) {
            mel[static_cast<size_t>(m) * static_cast<size_t>(n_frames) + static_cast<size_t>(t)] = cell(m, t);
        }
    }
    return mel;
}

void run_case(int n_samples) {
    using transcribe::parakeet::ParakeetHParams;
    ParakeetHParams hp{};
    hp.fe_hop_length = 160;

    const int n_mels   = 7;
    const int n_frames = n_samples / hp.fe_hop_length + 1;  // MelFrontend framing
    const int valid    = n_samples / hp.fe_hop_length;

    std::vector<float> mel = make_mel(n_mels, n_frames);
    const int          got = transcribe::parakeet::drop_nemo_padding_mel_frame(hp, mel, n_mels, n_frames, n_samples);
    check(got == valid, "returns floor(n / hop)");
    check(mel.size() == static_cast<size_t>(n_mels) * static_cast<size_t>(valid), "buffer resized to n_mels * valid");
    bool layout_ok = true;
    for (int m = 0; m < n_mels && layout_ok; ++m) {
        for (int t = 0; t < valid; ++t) {
            if (mel[static_cast<size_t>(m) * static_cast<size_t>(valid) + static_cast<size_t>(t)] != cell(m, t)) {
                layout_ok = false;
                break;
            }
        }
    }
    check(layout_ok, "rows compacted in place, frame order preserved");

    // Kestrel-masked checkpoints mask the padding in-graph: untouched.
    hp.kestrel_length_masking = true;
    std::vector<float> kmel   = make_mel(n_mels, n_frames);
    const int          kgot = transcribe::parakeet::drop_nemo_padding_mel_frame(hp, kmel, n_mels, n_frames, n_samples);
    check(kgot == n_frames, "kestrel: frame count unchanged");
    check(kmel == make_mel(n_mels, n_frames), "kestrel: buffer unchanged");
}

}  // namespace

int main() {
    // The #252 clips (80640 -> 505 frames, 168960 -> 1057) plus their
    // one-sample-shorter neighbours and a few unaligned lengths.
    for (const int n : { 80640, 80639, 168960, 168959, 480000, 479999, 160, 161, 1601, 16000 }) {
        run_case(n);
    }

    // Fewer samples than one hop: nothing valid to keep; leave as is.
    {
        transcribe::parakeet::ParakeetHParams hp{};
        hp.fe_hop_length       = 160;
        std::vector<float> mel = make_mel(3, 1);
        check(transcribe::parakeet::drop_nemo_padding_mel_frame(hp, mel, 3, 1, 100) == 1, "sub-hop input unchanged");
        check(mel.size() == 3, "sub-hop buffer unchanged");
    }

    if (g_failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("parakeet_mel_valid_length_unit: OK\n");
    return 0;
}

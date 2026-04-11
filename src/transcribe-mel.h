// transcribe-mel.h - native C++ log-mel feature extractor.
//
// PRIVATE header. Lives under src/ and is not exported. The public C
// ABI in include/transcribe.h does not surface this; the family
// handlers (currently only Parakeet) construct one and call it from
// inside transcribe_run when phase 4 lands.
//
// Phase 3 scope: pure CPU, no ggml dependency. Audio in, mel out,
// std::vector<float> for the buffer. The encoder builder in phase 4
// is responsible for copying the result into a ggml_tensor at the
// backend boundary.
//
// Algorithm pinned by phase 3 preflight (see RESUME.md "Parakeet
// frontend spec"). The implementation matches NeMo's
// AudioToMelSpectrogramPreprocessor / FilterbankFeatures pipeline as
// exported in nemo128.onnx, verified against the upstream NeMo
// features.py source and a standalone PyTorch reproduction. The
// per-NeMo numerical constants (log_eps = 2^-24, unbiased variance,
// reflect padding, fp64 STFT, mag_power = 2.0) are hardcoded in the
// .cpp file rather than added to the GGUF schema; see PLAN.md
// "complete contract" rule. They become per-family knobs only when a
// second family wants different values.
//
// Construction is one-shot: the constructor precomputes the periodic
// hann window (zero-padded from win_length to n_fft) and the
// Slaney-normalized mel filterbank. compute() is then a function of
// (config, audio) only.

#pragma once

#include "transcribe.h"

#include <cstddef>
#include <string>
#include <vector>

namespace transcribe {

// Frontend configuration. Mirrors the stt.frontend.* KV the loader
// reads out of ParakeetHParams::fe_*. Family-agnostic: SenseVoice
// will fill the same struct with different values when it lands.
//
// The defaults are the real Parakeet 0.6B values (matching
// nemo128.onnx). Constructing a default-initialized MelConfig and
// calling MelFrontend(cfg) gives a working extractor for v2 / v3.
struct MelConfig {
    int   sample_rate    = 16000;
    int   num_mels       = 128;
    int   n_fft          = 512;
    int   win_length     = 400;
    int   hop_length     = 160;
    float pre_emphasis   = 0.97f;     // 0.0 disables
    float f_min          = 0.0f;
    float f_max          = 8000.0f;
    std::string pad_mode = "reflect"; // "reflect" or "constant"

    // Optional checkpoint-provided mel filterbank [num_mels * (n_fft/2+1)]
    // row-major, Slaney-normalised. When non-empty, used instead of
    // computing from scratch. Set by the loader when the GGUF contains
    // frontend.mel_filterbank.
    std::vector<float> filterbank;

    // Optional checkpoint-provided window [win_length]. When non-empty,
    // used instead of computing a periodic Hann window.
    std::vector<float> window;
};

// Pure C++ log-mel extractor. Construct once, call compute() any
// number of times. Thread-safety: const after construction; multiple
// threads may call compute() concurrently.
class MelFrontend {
public:
    explicit MelFrontend(const MelConfig & cfg);

    // Run the full pipeline. pcm must be 16 kHz mono float32 in
    // [-1, 1]; n_samples is the number of input samples. The output
    // mel buffer is resized to num_mels * n_frames in row-major
    // [num_mels, n_frames] layout.
    //
    // Returns:
    //   TRANSCRIBE_OK              normal success.
    //   TRANSCRIBE_ERR_INVALID_ARG pcm is null, or n_samples is too
    //                              short to produce >= 2 frames
    //                              (per-feature normalize divides by
    //                              n_frames - 1, which would NaN).
    transcribe_status compute(
        const float * pcm,
        size_t n_samples,
        std::vector<float> & out_mel,
        int & out_n_mels,
        int & out_n_frames) const;

    // Number of mel bins (matches MelConfig::num_mels).
    int num_mels() const { return cfg_.num_mels; }

    // Frame count for a given audio length, before calling compute().
    // Matches NeMo: floor(n_samples / hop_length) + 1.
    int n_frames_for(size_t n_samples) const;

    // Read-only accessors for unit tests. Not part of the runtime
    // API; the goal is to validate the precomputed buffers in
    // isolation without running the full pipeline.
    const std::vector<double> & window()     const { return window_; }
    const std::vector<float>  & filterbank() const { return mel_fb_; }
    const MelConfig            & config()     const { return cfg_; }
    int                          n_freq()     const { return n_freq_; }

private:
    MelConfig cfg_;
    int n_freq_;                      // n_fft/2 + 1
    std::vector<double> window_;      // [n_fft], periodic hann zero-padded
    std::vector<float>  mel_fb_;      // [num_mels * n_freq] row-major, Slaney
};

} // namespace transcribe

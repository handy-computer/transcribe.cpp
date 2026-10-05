// arch/ecapa_tdnn/mel.h - SpeechBrain log-mel front end (60-bin,
// sentence-mean normalised) for the ecapa_tdnn family.
//
// INTERNAL to src/arch/ecapa_tdnn/. Pure CPU, no ggml dependency:
// audio in, std::vector<float> mel out; the family graph builder copies
// the result into a ggml_tensor at the backend boundary.
//
// Reproduces SpeechBrain's Fbank pipeline (speechbrain/processing/
// features.py: STFT -> spectral_magnitude(power=1, log=False) ->
// Filterbank -> InputNormalization(norm_type="sentence", std_norm=False))
// as configured by speechbrain/lang-id-voxlingua107-ecapa:
//
//   * torch.stft(n_fft=400, hop=160, win_length=400, center=True,
//     pad_mode="constant", onesided=True) -> 201 bins, T = n/160 + 1.
//   * Window: torch.hamming_window(400) — PERIODIC,
//     w[n] = 0.54 - 0.46*cos(2*pi*n/400).
//   * Power spectrum re^2 + im^2 (SpeechBrain's spectral_magnitude with
//     power=1 on the squared-modulus tensor).
//   * Mel: 60 SpeechBrain triangles, applied as a plain [60, 201] matmul.
//     The filterbank is NOT rebuilt here — its triangle geometry is
//     SpeechBrain-specific and is dumped by the converter into the GGUF
//     tensor `frontend.mel_filterbank`. MelConfig::filterbank is required.
//   * Log: 10*log10(max(x, 1e-10)) (Filterbank.log_mel, amin=1e-10,
//     ref_value=1.0 so the reference term vanishes).
//   * Top-dB floor: per utterance over time AND frequency,
//     x = max(x, max(x) - 80).
//   * Normalisation: subtract each mel bin's mean over all frames.
//
// No pre-emphasis, no dither, no deltas: SpeechBrain's compute_features
// for this checkpoint applies none of them.
//
// Construction is one-shot: the constructor validates the config and
// precomputes the window and the FFT twiddle LUT. compute() is then a
// pure function of (config, audio) and is const / thread-safe.

#pragma once

#include "transcribe.h"

#include <cstdint>
#include <string>
#include <vector>

namespace transcribe::ecapa_tdnn {

// Front-end configuration. Mirrors the stt.frontend.* GGUF metadata the
// loader reads; the defaults are the real VoxLingua107 values, so a
// default-constructed MelConfig plus the checkpoint's filterbank gives a
// working extractor.
struct MelConfig {
    int sample_rate = 16000;
    int n_mels      = 60;
    int n_fft       = 400;
    int win_length  = 400;  // must equal n_fft; SpeechBrain uses no sub-window pad
    int hop_length  = 160;

    // STFT window shape. Only "hamming_periodic" is implemented:
    // torch.hamming_window(N) with the default periodic=True,
    // w[n] = 0.54 - 0.46*cos(2*pi*n/N). Any other value is rejected.
    std::string window_type = "hamming_periodic";

    // STFT input padding. Only "constant" is implemented: torch's
    // center=True with pad_mode="constant", i.e. n_fft/2 zeros on each
    // side. Any other value is rejected.
    std::string pad_mode = "constant";

    // Log floor: 10*log10(max(power, log_floor)).
    float log_floor = 1e-10f;

    // Per-utterance dynamic-range floor applied over time and frequency:
    // x = max(x, max_all(x) - top_db). <= 0 disables it.
    float top_db = 80.0f;

    // Normalisation mode:
    //   "sentence_mean" — subtract each mel bin's mean over all frames
    //                     (SpeechBrain InputNormalization, norm_type=
    //                     "sentence", std_norm=False). Default.
    //   "none"          — emit the log-mel as-is.
    std::string normalize = "sentence_mean";

    // Mel filterbank [n_mels * n_freq] row-major MEL-MAJOR (each filter's
    // n_freq weights contiguous), n_freq = n_fft/2 + 1. REQUIRED: the
    // front end never rebuilds it, because SpeechBrain's triangle
    // geometry (symmetric filters whose half-width is the distance to
    // the left neighbour) is not the librosa/Slaney formula. Supplied by
    // the loader from the GGUF tensor `frontend.mel_filterbank`.
    std::vector<float> filterbank;
};

// Pure C++ log-mel extractor. Construct once, call compute() any number
// of times. Thread-safety: const after construction; multiple threads may
// call compute() concurrently.
class MelFrontend {
  public:
    explicit MelFrontend(const MelConfig & cfg);

    // TRANSCRIBE_OK when the config was accepted, else TRANSCRIBE_ERR_GGUF
    // (the offending field is named in a log message). compute() returns
    // the same status without touching the audio.
    transcribe_status status() const { return status_; }

    // Frame count for a given audio length, before calling compute().
    // Matches torch center=True: floor(n_samples / hop_length) + 1. No
    // trailing-frame drop — the trailing frames see the zero pad, exactly
    // as torch does, and SpeechBrain keeps them.
    int n_frames_for(int64_t n_samples) const;

    // Run the full pipeline. pcm is 16 kHz mono float32 in [-1, 1].
    //
    // out is resized to n_frames * n_mels and written FRAME-MAJOR: frame
    // t's n_mels values are contiguous at out[t * n_mels]. That is
    // exactly ggml ne = [n_mels, T] and the reference dump layout
    // [T, 60], so the graph builder can memcpy it and the dumper can
    // write it without a transpose.
    //
    // n_threads controls STFT parallelism: 0 means default_n_threads()
    // (capped at 8), further capped at n_frames. The result is
    // bit-identical for any thread count — every frame is computed
    // independently into its own output slot, and the log / top-dB /
    // mean passes are single-threaded.
    //
    // Returns:
    //   TRANSCRIBE_OK               normal success.
    //   TRANSCRIBE_ERR_GGUF         bad config (see status()).
    //   TRANSCRIBE_ERR_INVALID_ARG  pcm is NULL or n_samples < 1. The
    //                               500 ms minimum is enforced by the
    //                               LANGID role dispatcher, not here.
    transcribe_status compute(const float *        pcm,
                              int64_t              n_samples,
                              std::vector<float> & out,
                              int &                out_n_frames,
                              int                  n_threads = 0) const;

    // Read-only accessors for unit tests and the parity tool.
    //
    // window() and n_freq() are the CONSTRUCTED state and are only valid
    // when status() == TRANSCRIBE_OK: a rejected config leaves the window
    // empty and n_freq() at 0, while config() still returns the caller's
    // struct verbatim. Check status() before trusting either.
    const std::vector<float> & window() const { return window_; }

    const MelConfig & config() const { return cfg_; }

    int n_freq() const { return n_freq_; }

  private:
    MelConfig         cfg_;
    transcribe_status status_ = TRANSCRIBE_OK;
    int               n_freq_ = 0;  // n_fft/2 + 1

    std::vector<float> window_;     // [n_fft], periodic Hamming, fp32

    // cos/sin LUT for the mixed-radix FFT, indexed by 2*pi*i / n_fft.
    // Sized to n_fft so every recursion level N (n_fft -> n_fft/2 -> ...
    // -> odd leaf) divides the LUT exactly and lookups never fall back to
    // live std::cos / std::sin. Stored fp32 to match the FFT precision
    // and avoid a per-load cast in the inner butterfly.
    std::vector<float> cos_lut_;
    std::vector<float> sin_lut_;
};

// --- test hook ---------------------------------------------------------
//
// Run the production mixed-radix FFT over a 400-sample real input,
// building the twiddle LUT exactly as MelFrontend's constructor does, and
// write the 201 one-sided bins. Exists so tests/ecapa_tdnn_mel_unit.cpp can compare
// the FFT against a naive O(N^2) fp64 DFT without including the .cpp or
// running the whole pipeline. Not used by the runtime.
void mel_test_fft(const float * in400, float * out_re201, float * out_im201);

}  // namespace transcribe::ecapa_tdnn

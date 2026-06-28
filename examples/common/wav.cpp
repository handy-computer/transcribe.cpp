// wav.cpp - implementation of the example CLI's tiny WAV loader.

#include "wav.h"

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

namespace transcribe_cli {

namespace {

constexpr unsigned int kRequiredSampleRate = 16000;

// Sanity cap on per-call audio length, applied after we've validated the
// sample rate. 24 hours of 16 kHz mono float32 is ~5.5 GB, which is well
// past anything a one-shot transcribe call should be doing. Treat
// anything above this as a malformed or malicious header rather than
// trying to allocate it.
constexpr size_t kMaxFrames = static_cast<size_t>(24) * 60 * 60 * kRequiredSampleRate;

std::string format_size_t(size_t v) {
    // C++17, no <format> yet on every toolchain we care about.
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%zu", v);
    return std::string(buf);
}

}  // namespace

bool load_wav_mono_16k(const std::string & path, std::vector<float> & out_pcm, std::string & out_error) {
    out_pcm.clear();
    out_error.clear();

    drwav wav;
    if (!drwav_init_file(&wav, path.c_str(), nullptr)) {
        out_error = "could not open WAV file '" + path + "'";
        return false;
    }

    // Snapshot every wav.* field we care about BEFORE drwav_uninit. The
    // dr_wav struct fields are documented as readable after uninit (it
    // only frees buffers), but reading them back later is brittle and
    // makes the control flow harder to follow. Pin them now.
    const unsigned int sample_rate  = wav.sampleRate;
    const unsigned int channels     = wav.channels;
    const size_t       total_frames = static_cast<size_t>(wav.totalPCMFrameCount);

    if (sample_rate != kRequiredSampleRate) {
        out_error = "WAV sample rate is " + std::to_string(sample_rate) +
                    " Hz; transcribe.cpp v1 only accepts 16000 Hz." + " Resample first, e.g.:\n" + "  sox '" + path +
                    "' -r 16000 -c 1 out.wav\n" + "  ffmpeg -i '" + path + "' -ar 16000 -ac 1 out.wav";
        drwav_uninit(&wav);
        return false;
    }

    if (channels == 0) {
        out_error = "WAV file reports 0 channels";
        drwav_uninit(&wav);
        return false;
    }

    if (total_frames == 0) {
        out_error = "WAV file is empty";
        drwav_uninit(&wav);
        return false;
    }

    if (total_frames > kMaxFrames) {
        out_error = "WAV file is implausibly long (" + format_size_t(total_frames) + " frames); refusing to load";
        drwav_uninit(&wav);
        return false;
    }

    // Overflow-safe interleaved-buffer size: total_frames * channels.
    // Both inputs are bounded by kMaxFrames * UINT_MAX in the worst case,
    // but we still have to defend against the multiplication wrapping
    // size_t on a 32-bit host, or against a malformed header reporting
    // an absurd channel count even though we already gated total_frames.
    if (channels > std::numeric_limits<size_t>::max() / total_frames) {
        out_error = "WAV interleaved buffer size overflows size_t (" + format_size_t(total_frames) + " frames * " +
                    std::to_string(channels) + " channels)";
        drwav_uninit(&wav);
        return false;
    }
    const size_t interleaved_count = total_frames * channels;

    // Read everything as interleaved float32, then downmix to mono if
    // necessary by averaging channels. dr_wav handles the source sample
    // format conversions for us (PCM int16/24/32, float32/64, A-law,
    // mu-law).
    std::vector<float> interleaved(interleaved_count);
    const size_t       got = drwav_read_pcm_frames_f32(&wav, total_frames, interleaved.data());
    drwav_uninit(&wav);

    if (got != total_frames) {
        out_error =
            "WAV decode short read: expected " + format_size_t(total_frames) + " frames, got " + format_size_t(got);
        return false;
    }

    if (channels == 1) {
        out_pcm = std::move(interleaved);
        return true;
    }

    out_pcm.resize(total_frames);
    const float inv_channels = 1.0f / static_cast<float>(channels);
    for (size_t i = 0; i < total_frames; ++i) {
        float sum = 0.0f;
        for (unsigned int c = 0; c < channels; ++c) {
            sum += interleaved[i * channels + c];
        }
        out_pcm[i] = sum * inv_channels;
    }
    return true;
}

}  // namespace transcribe_cli

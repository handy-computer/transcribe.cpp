// wav.h - tiny WAV loader for the example CLI.
//
// Loads a RIFF WAV file as mono float32 PCM at 16 kHz, which is the only
// input format the v1 transcribe runtime accepts. The WAV loader does NOT
// resample. Files at any other sample rate are rejected with a message
// pointing the user at sox/ffmpeg, per PLAN.md "Sample Rate Policy".
//
// This file lives under examples/common/ on purpose: dr_wav and the WAV
// loader are not part of the runtime library. The core library never
// decodes WAV files.

#pragma once

#include <string>
#include <vector>

namespace transcribe_cli {

// Load a WAV file from disk into mono float32 samples at 16 kHz.
//
// On success: returns true, populates out_pcm with the samples in
// [-1.0, 1.0], and leaves out_error empty.
//
// On failure: returns false, leaves out_pcm in an unspecified state, and
// populates out_error with a human-readable message.
//
// Stereo (or higher channel count) input is downmixed to mono by
// averaging channels. Any sample rate other than 16 kHz is an error.
bool load_wav_mono_16k(const std::string &  path,
                       std::vector<float> & out_pcm,
                       std::string &        out_error);

} // namespace transcribe_cli

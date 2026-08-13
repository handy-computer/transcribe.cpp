// arch/whisper/encoder.h - whisper family binding onto the shared Whisper
// encoder graph builder in src/whisper_graph/encoder.h.
//
// The graph itself is family-agnostic (it reads only dimensions off HParams),
// so it is shared with `crisperwhisper`. This header re-exports the builder
// and its result types into `transcribe::whisper` so the family's model.cpp
// keeps calling them unqualified.

#pragma once

#include "whisper_graph/encoder.h"

namespace transcribe::whisper {

using whisper_graph::build_encoder_graph;
using EncoderBuild = whisper_graph::EncoderBuild;
using EncoderDumps = whisper_graph::EncoderDumps;

}  // namespace transcribe::whisper

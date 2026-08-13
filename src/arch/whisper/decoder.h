// arch/whisper/decoder.h - whisper family binding onto the shared Whisper
// decoder graph builders in src/whisper_graph/decoder.h.
//
// The four graph families (prefill / cross-KV / KV-cached / static step, plus
// the batched variants) read only dimensions off HParams and carry no
// decode-contract logic, so they are shared with `crisperwhisper`. This
// header re-exports them into `transcribe::whisper` so the family's model.cpp
// keeps calling them unqualified.

#pragma once

#include "whisper_graph/decoder.h"

namespace transcribe::whisper {

using whisper_graph::build_cross_kv_graph;
using whisper_graph::build_cross_kv_graph_batched;
using whisper_graph::build_decoder_graph_kv;
using whisper_graph::build_decoder_prefill_graph;
using whisper_graph::build_step_graph;
using whisper_graph::build_step_graph_batched;

using DecoderBuild     = whisper_graph::DecoderBuild;
using DecoderDumps     = whisper_graph::DecoderDumps;
using StepBuild        = whisper_graph::StepBuild;
using StepBuildBatched = whisper_graph::StepBuildBatched;

}  // namespace transcribe::whisper

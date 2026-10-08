// cli.h - shared declarations for the transcribe-cli example.
//
// main.cpp parses arguments, owns process setup (log sink, output file), and
// routes each model to its role's driver (asr.cpp, diarize.cpp, langid.cpp). Shared
// helpers live in namespace transcribe_cli next to the WAV loader in
// examples/common.

#pragma once

#include "transcribe.h"
#include "transcribe/whisper.h"

#include <fstream>
#include <string>
#include <vector>

namespace transcribe_cli {

struct cli_args {
    std::string                wav_path;
    std::string                model_path;
    std::string                language;
    std::string                target_language;       // --target-language: target lang for translation
    std::string                batch_file;            // --batch: one wav path per line
    int                        batch_size   = 0;      // --batch-size: >1 groups utterances into
                                                      // transcribe_run_batch calls (offline only).
                                                      // 0/1 keeps the per-file serial loop.
    bool                       translate    = false;
    bool                       instruct     = false;  // --task instruct
    bool                       quiet        = false;
    bool                       list_devices = false;  // --list-devices: print devices and exit
    bool                       batch_jsonl  = false;  // --batch-jsonl: output JSONL
    std::string                output_path;           // -o/--output: write raw text here
    int                        repeat       = 1;
    int                        n_threads    = 0;      // 0 = library default (all cores)
    int                        n_ctx        = 0;      // 0 = model's true max; >0 lowers the cap
    transcribe_kv_type         kv_type      = TRANSCRIBE_KV_TYPE_AUTO;
    transcribe_backend_request backend      = TRANSCRIBE_BACKEND_AUTO;
    int                        device_index = -1;  // --device N: -1 = auto, >=0 = exact registry device
    transcribe_timestamp_kind  timestamps   = TRANSCRIBE_TIMESTAMPS_AUTO;

    // Generic prompting (transcribe_run_params::vocabulary / prompt / prefix).
    std::vector<std::string> vocabulary;  // --vocabulary TERMS / --vocabulary-file PATH
    std::string              prompt;      // --prompt TEXT
    std::string              prefix;      // --prefix TEXT

    // Whisper-family knobs. Ignored for non-Whisper models.
    std::string                              initial_prompt;                    // --initial-prompt TEXT
    bool                                     whisper_set              = false;
    bool                                     temperature_set          = false;  // --temperature F
    float                                    temperature              = 0.0f;   // tier-0 sampling temp
    bool                                     condition_on_prev_tokens = false;  // --condition-on-prev-tokens
    enum transcribe_whisper_prompt_condition prompt_condition =
        TRANSCRIBE_WHISPER_PROMPT_FIRST_SEGMENT;                                // --prompt-condition first|all

    // SenseVoice / FunASR-Nano family knobs. The `--itn` flag is shared:
    // it routes to whichever family the loaded model belongs to. Ignored
    // by non-ITN-aware families. Unset leaves the library default in place,
    // which differs per family (sensevoice: on; funasr-nano: off), so the
    // initializer here is only read once --itn / --no-itn has been seen.
    bool use_itn           = false;  // --itn / --no-itn
    bool itn_set           = false;
    bool keep_special_tags = false;  // --raw-tokens

    // Canary family knobs. Ignored by non-Canary families.
    bool canary_pnc     = true;   // default: punctuation+caps on
    bool canary_pnc_set = false;  // --pnc / --no-pnc set this

    // Speaker diarization toggle (moss / granite-plus). Unset = library
    // default (OFF). --diarize / --no-diarize set this.
    bool diarize     = true;
    bool diarize_set = false;

    // Streaming demo: when > 0, the single-file path feeds the WAV
    // through transcribe_stream_begin/feed/finalize in fixed-size
    // ms-aligned chunks instead of one transcribe_run call. Requires
    // the loaded model to advertise supports_streaming. Set by
    // --stream-chunk-ms N.
    int stream_chunk_ms      = 0;
    // Parakeet streaming: pick a right-context (lookahead) setting
    // from the model's training menu. -1 = model default (max
    // accuracy / max latency); 0/1/6/13 select the published
    // nemotron-speech-streaming-en-0.6b settings. Set by
    // --stream-att-right N. Ignored when stream_chunk_ms == 0.
    int stream_att_right     = -1;
    // Parakeet buffered streaming (parakeet-unified-en-0.6b): override
    // the (L, C, R) attention context tuple in milliseconds. -1 = use
    // the model's default (highest-accuracy row of the training menu).
    // Frame-aligned: the lib rounds each value down to the nearest
    // post-subsample frame (80ms at 4x subsampling). Ignored when
    // stream_chunk_ms == 0 or when the model is not buffered-streaming.
    int stream_buf_left_ms   = -1;
    int stream_buf_chunk_ms  = -1;
    int stream_buf_right_ms  = -1;
    // Voxtral Realtime streaming: transcription delay in 12.5 Hz audio
    // tokens (80 ms each). -1 = model default (6 = 480 ms). Set by
    // --stream-voxtral-delay N. Ignored when stream_chunk_ms == 0 or when
    // the model is not voxtral_realtime.
    int stream_voxtral_delay = -1;
    // Speculative-decode draft length passed through to
    // transcribe_run_params::spec_k_drafts on the offline path. -1 = family
    // default (each family picks its tuned K). 0 = explicitly off. >0 =
    // explicit K. Silently ignored by families without
    // supports_spec_decode. Set by --spec-k-drafts N.
    int spec_k_drafts        = -1;

    // LANGID role (language ID models). --allow restricts the decision to
    // these labels (codes or aliases); --top prints only the best N ranked
    // candidates (0 = all). The library always ranks every allowed label.
    std::vector<std::string> langid_allow;
    int                      langid_top = 0;
};

// -o/--output: write `text` (newline-terminated) to `output` if non-null.
// Returns false (after an error message naming `path`) on a write failure.
bool write_output_file(std::ofstream * output, const std::string & path, const char * text);

// ASR batch driver (asr.cpp): loads the model, runs every listed file, prints,
// and returns the process exit code.
int run_asr_batch(const cli_args & args, std::ofstream * output);

// Single-file drivers, one per role. main.cpp loads the audio and the model,
// prints them, and dispatches on transcribe_model_roles(); the driver takes
// ownership of the model and returns the process exit code.
int run_asr_file(const cli_args &           args,
                 transcribe_model *         model,
                 const std::vector<float> & pcm,
                 double                     duration_s,
                 std::ofstream *            output);

// With -o, writes the segment lines.
int run_diarize_file(const cli_args &           args,
                     transcribe_model *         model,
                     const std::vector<float> & pcm,
                     double                     duration_s,
                     std::ofstream *            output);

// With -o, writes the candidate lines.
int run_langid_file(const cli_args &           args,
                    transcribe_model *         model,
                    const std::vector<float> & pcm,
                    std::ofstream *            output);

}  // namespace transcribe_cli

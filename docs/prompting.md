# Prompting

Generic prompting fields on `transcribe_run_params` (CLI flag in parentheses).
Probe `transcribe_model_supports()` for the matching feature bit.

| Field | Feature bit | Effect |
|---|---|---|
| `vocabulary` (`--vocabulary`, `--vocabulary-file`) | `VOCABULARY` | Custom terms, priority order, formatted per family. |
| `prompt` (`--prompt`) | `CONTEXT_PROMPT` | Context text in the model's conditioning slot. |
| `task = INSTRUCT` + `prompt` (`--task instruct`) | `INSTRUCT` | `prompt` replaces the transcription request; output is free text. |
| `prefix` (`--prefix`) | `TRANSCRIPT_PREFIX` | Transcript text the model continues from. Unsupported is an error. |

| Family | Vocabulary | Context prompt | Instruct | Prefix |
|---|---|---|---|---|
| Whisper | yes | yes | | yes |
| Qwen3-ASR | yes | yes | | |
| Voxtral (2507) | | | yes | |
| Granite 4.0-1b / 4.1-2b | yes | | | |
| Granite 4.1-2b-plus | yes | | | yes |
| Canary 180m-flash / 1b-flash / 1b-v2 | | | | yes |
| Fun-ASR-Nano | yes | | | |
| MOSS-Transcribe-Diarize | yes | | | |

Per-model formats and restrictions are in each model doc's **Prompting:**
note under [`models/`](models/).

## Limits

- Over the prompt budget, `vocabulary` drops terms from the end of the list and
  `prompt` keeps its most recent text, both with a WARN. An INSTRUCT prompt
  that does not fit is an error.
- INSTRUCT requires `target_language == NULL` and timestamps NONE or AUTO.
- `prefix`: the audio must contain the prefix's speech, and long-form models
  apply it to the first window only.
- A feature bit covers plain transcription. Under another task or output mode
  a model may ignore `vocabulary` or `prompt` with a WARN; its model doc says
  when.

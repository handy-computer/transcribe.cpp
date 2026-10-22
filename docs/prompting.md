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
| Parakeet (TDT and RNN-T heads) | yes | | | |

Per-model formats and restrictions are in each model doc's **Prompting:**
note under [`models/`](models/).

## Parakeet vocabulary

Parakeet has no prompt slot, so its transducer decoders take vocabulary terms
as phrase boosting: each term is tokenized the way the model writes it, starting
at a word boundary, and the terms form an Aho-Corasick automaton over token
ids. The model alone decides between blank and a token on every greedy step.
When it emits a token, the vocabulary re-ranks the non-blank tokens. Once the
model has begun a term on its own, each further piece that extends the match
earns a logit bonus (8.0 by default, chosen on a held-out term set), a token
that breaks the match gives back what the unfinished match earned, and a
completed term keeps its bonus. Biasing therefore changes which word is
written, never whether one is, so it cannot invent speech on silence. The
reported token confidences stay the model's own, and with no terms decoding is
bit-exact. Terms with characters the tokenizer cannot write are skipped with a
WARN. Offline runs and batches take terms; streaming ignores them, and CTC
heads do not take them yet.

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

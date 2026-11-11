# Community fine-tunes

Fine-tunes of supported models, published by their authors rather than by this
project.

These models are not part of the [catalog](../catalog/). Nobody here has
verified them numerically or measured their WER, and their GGUFs are not hosted
under [`handy-computer`](https://huggingface.co/handy-computer). The accuracy
figures below are the authors' own. Each author maintains their own model, so
questions about one go to its repo.

If a fine-tune keeps the base model's architecture, it should load through that
family's existing code. Check its licence before you use it: some are
non-commercial.

## Models

| Model | Base | Family | For | GGUFs | Licence | Author's result |
| --- | --- | --- | --- | --- | --- | --- |
| [Fritter](https://huggingface.co/danielrosehill/Fritter) | `parakeet-tdt-0.6b-v2` | `parakeet` | English with technical and AI vocabulary | [F16, Q8_0, Q4_K_M](https://huggingface.co/danielrosehill/Fritter/tree/main/gguf) | CC-BY-NC-4.0 | Tech-sentence WER 7.53% to 4.01% on v1.0. Held-out recordings of one speaker, who is also the only speaker trained on, so other voices may gain less. |

## Using one

Download a GGUF and pass it to the CLI:

```bash
transcribe-cli -m Fritter-Q8_0.gguf audio.wav
```

## Adding a model

Open a PR that adds one row to the table, and link the repo that holds the GGUFs.

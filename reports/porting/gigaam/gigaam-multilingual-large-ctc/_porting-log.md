# gigaam / gigaam-multilingual-large-ctc porting log

Surprises worth feeding back into the docs and tooling.

- **The package pin cannot load this variant by name.** `gigaam` at
  6e4b027c has no `multilingual_large_ctc` key, and newer package commits
  download from the author CDN. `scripts/lib/gigaam_hf.py` builds the same
  package classes from the pinned Hugging Face snapshot instead, so the
  family keeps one reference implementation and one package pin.
- **Bumping the package pin was considered and rejected for this port.**
  Upstream added the `multilingual_large_ctc` key in 559d88d, but that
  commit loads the checkpoint from the author CDN by MD5, not from the
  Hugging Face commit the GGUF records, and 8555893 and 559d88d change
  `gigaam/encoder.py`. A bump moves the reference of the four shipped v3
  variants as well, so it needs their dumps, tolerances, and WER baselines
  redone; that is a family decision for the maintainers, not part of
  adding one variant.
- **The revision pins had three copies.** The dumper, the WER runner, and
  the golden manifests each listed the commits, the WER runner with `main`
  for every v3 variant, and `validate.py` did not pass the manifest's
  `hf_revision` to the gigaam dumper. `scripts/lib/gigaam_hf.py` `VARIANTS`
  is now the only table; a unit test compares it with the manifests, and
  the dumper refuses a `--revision` that differs from it.
- **`$TRANSCRIBE_MODELS_DIR/<slug>/` is keyed by repo, not by commit.**
  With the Hub unreachable, huggingface_hub 1.14 returns that directory
  without checking which commit it holds, so a pinned load could read a
  file of another commit or a file another tool saved there.
  `lib.hf_source.download_snapshot` now downloads a pinned commit hash
  into `<slug>@<commit>/`, for every converter that pins one; a checkout
  in the old `<slug>/` is not reused and can be deleted.
- **`mel_scale` and `mel_norm` are not in the Multilingual config.** The v3
  configs list `htk` and `null`, but `FeatureExtractor` never passes either
  to torchaudio, and it defaults `center` to true where a config omits it.
  The converter now reads the frontend values the module sets from the
  instantiated module and refuses a config value that disagrees with it; the four v3
  GGUFs are byte-identical to the previous converter's output.
- **The author CDN served an expired TLS certificate** during this port,
  so `gigaam.load_model("v3_*")` failed and the v3 regression check could
  not use the converter's own download path. It used the v3 weights from
  the `ai-sage/GigaAM-v3` Hugging Face branches instead.
- **`scripts/intake.py inspect` has no `--revision`.** It reads the repo's
  `main` branch, which for `ai-sage/GigaAM-Multilingual` is the 220M `ctc`
  model, not `large_ctc`. The intake was written by hand from the pinned
  commit.
- **`validate.py --mel-from-ref` is whisper-only**, although the gigaam
  runtime honors `TRANSCRIBE_MEL_FROM_REF`. The frontend isolation
  (subsample drift unchanged with the reference mel) was run by setting
  `TRANSCRIBE_DUMP_DIR` and `TRANSCRIBE_MEL_FROM_REF` by hand and calling
  `compare_tensors.py`.
- **The WER reference runner was v3-only.**
  `scripts/wer/run_reference_gigaam_author.py` called `gigaam.load_model`
  with a v3 key; it now loads through `scripts/lib/gigaam_hf.py` and
  accepts `multilingual_large_ctc`. The WER baseline has not been run.
- **No C++ change.** The 24-layer, d_model=1024 encoder and the 71-class
  CTC head run on the graph built for v3, which reads both from GGUF
  metadata.
- **Q4_K_M changes one letter** on one kk FLEURS utterance
  (`оцелоттар` for `ацелоттар`) on Metal; Q5_K_M and above match the
  reference on all nine checked clips.

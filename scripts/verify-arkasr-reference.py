# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = [
#   "librosa",
#   "soundfile",
#   "torch",
#   "transformers",
# ]
# ///
"""Run the official ARK-ASR checkpoint as an independent parity oracle."""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "mps" else torch.float32

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=dtype,
        attn_implementation="sdpa",
    ).to(device)
    model.eval()

    conversation = [{
        "role": "user",
        "content": [
            {"type": "audio", "path": args.audio},
            {"type": "text", "text": "Please transcribe this audio."},
        ],
    }]
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_tensors="pt",
        sampling_rate=16000,
        audio_padding="longest",
        text_kwargs={"padding": "longest"},
        audio_max_length=30 * 16000,
    ).to(device)
    if "audios" in inputs:
        inputs["audios"] = inputs["audios"].to(dtype=dtype)

    eos_ids = tokenizer.eos_token_id
    keep_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids or [])
    bad_ids = set(tokenizer.all_special_ids) - keep_ids
    bad_ids.update(
        token_id
        for token, token_id in tokenizer.get_added_vocab().items()
        if token.startswith("<") and token.endswith(">") and token_id not in keep_ids
    )

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=256,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bad_words_ids=[[token_id] for token_id in sorted(bad_ids)],
        )
    print(tokenizer.batch_decode(
        outputs[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )[0])


if __name__ == "__main__":
    main()

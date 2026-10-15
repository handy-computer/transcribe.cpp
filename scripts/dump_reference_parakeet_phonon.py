#!/usr/bin/env python3
"""Phonon-2 goldens from the publisher's independent Transformers loader.

Run with scripts/envs/parakeet-phonon. Publisher scripts are pinned and
hash-checked; the converter's container reader is never used as the oracle.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.phonon import BASE_REVISION, REPO, REVISION, base_directory, source_directory, verify_sha256
from lib.ref_dump import write_tensor, write_transcript

PUBLISHER_FILES = {
    "reference_transformers.py": "8481cfc11537e6dc1e40a8296c93aa15f9b7437b1514564caf647f8773c0dcb3",
    "fermion_container.py": "5cf172e8cf6b313e951c4ef588ca913c7285b0aa3b92de413c3d7ec2b939c0e4",
}


def publisher_reference():
    from huggingface_hub import hf_hub_download
    import shutil
    root = Path(__file__).resolve().parent.parent / "build" / "references" / "phonon-2"
    root.mkdir(parents=True, exist_ok=True)
    for name, sha in PUBLISHER_FILES.items():
        path = Path(hf_hub_download(REPO, name, revision=REVISION))
        verify_sha256(path, sha)
        # The publisher resolves __file__ before importing its sibling.
        # HF cache symlinks resolve into blobs/, so stage real files together.
        shutil.copyfile(path, root / name)
    spec = importlib.util.spec_from_file_location("phonon_publisher_reference", root / "reference_transformers.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load(args):
    import torch
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(0)
    root = source_directory(args.model, args.revision)
    return publisher_reference().load_model(str(root / "model.fermion"), str(base_directory()), dtype=torch.float32)


def main() -> int:
    import soundfile as sf
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["container", "encoder", "decode"])
    parser.add_argument("--model", default=REPO)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--audio")
    parser.add_argument("--out", required=True)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--language", choices=["en"], default=None)
    args = parser.parse_args()
    out_dir = Path(args.out)
    if args.stage == "container":
        import hashlib
        import json
        from lib.phonon import read_container
        root = source_directory(args.model, args.revision)
        reference, _ = publisher_reference().container_state_dict(str(root / "model.fermion"))
        def signature(value):
            return value.shape, str(value.dtype), hashlib.sha256(value.tobytes()).hexdigest()
        expected = {key: signature(value.numpy()) for key, value in reference.items()}
        del reference
        actual = read_container(root / "model.fermion")
        if expected.keys() != actual.keys():
            raise ValueError("container tensor names differ from publisher reference")
        for key, value in actual.items():
            if signature(value) != expected[key]:
                raise ValueError(f"container reconstruction differs: {key}")
        out_dir.mkdir(parents=True, exist_ok=True)
        report = {"tensors": len(actual), "exact": True, "revision": REVISION,
                  "reference": "independent publisher container_state_dict"}
        (out_dir / "container-parity.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report))
        return 0
    if not args.audio:
        parser.error("--audio is required for encoder/decode")
    model, processor, info = load(args)
    pcm, sr = sf.read(args.audio, dtype="float32", always_2d=False)
    if sr != 16000 or pcm.ndim != 1:
        raise ValueError("reference cases must be 16 kHz mono WAV")
    inputs = processor([pcm], sampling_rate=16000, return_tensors="pt", padding=True)
    source = {"framework": "publisher_transformers", "model": REPO, "revision": REVISION,
              "base_revision": BASE_REVISION, "torch": torch.__version__, "dtype": "float32"}

    def dump(name, value, stage):
        array = value.detach().float().cpu().numpy() if isinstance(value, torch.Tensor) else value
        write_tensor(name, np.ascontiguousarray(array, dtype=np.float32), stage,
                     {**source, "hook": name}, out_dir=out_dir)

    hooks = []
    if args.stage == "encoder":
        dump("enc.mel.in", inputs.input_features[0].T, "frontend.mel.norm")
        def tap(name):
            def hook(_module, _inputs, output):
                dump(name, output[0], "encoder")
            return hook
        hooks.append(model.encoder.subsampling.register_forward_hook(tap("enc.pre_encode.out")))
        hooks.append(model.encoder.encode_positions.register_forward_hook(tap("enc.pos_emb")))
        for i in (0, 12, 23):
            hooks.append(model.encoder.layers[i].register_forward_hook(tap(f"enc.block.{i}.out")))
        # Norm inputs are the residual stream after each sub-block. Hooks
        # observe the canonical forward without reimplementing its math.
        for tag, norm in (("ff1", "norm_self_att"), ("attn", "norm_conv"),
                          ("conv", "norm_feed_forward2"), ("ff2", "norm_out")):
            def before_norm(_module, values, tag=tag):
                dump(f"enc.block.0.{tag}", values[0][0], "encoder.block.subblock")
            hooks.append(getattr(model.encoder.layers[0], norm).register_forward_pre_hook(before_norm))

    with torch.inference_mode():
        encoded = model.encoder(**inputs)
        valid = int(encoded.attention_mask[0].sum())
        if args.stage == "encoder":
            dump("enc.final", encoded.last_hidden_state[0, :valid], "encoder.final")
        else:
            dump("dec.enc_out", encoded.last_hidden_state[0, :valid], "decoder.enc_out")
            token = torch.tensor([[model.config.blank_token_id]])
            embedded = model.decoder.embedding(token)
            dump("dec.embed.0", embedded[0, 0], "decoder.embed")
            hidden, (h, c) = model.decoder.lstm(embedded)
            for i in range(h.shape[0]):
                dump(f"dec.lstm.{i}.h.0", h[i, 0], "decoder.lstm")
                dump(f"dec.lstm.{i}.c.0", c[i, 0], "decoder.lstm")
            prediction = model.decoder.decoder_projector(hidden)
            projection = model.encoder_projector(encoded.last_hidden_state[:, :1])
            logits = model.joint(prediction, projection)
            dump("dec.joint.0", torch.log_softmax(logits.float(), -1).reshape(-1), "decoder.joint")
            # Exactly the publisher's public generation/decoding path.
            generated = model.generate(**inputs)
            sequences = getattr(generated, "sequences", generated)
            text = processor.batch_decode(sequences, skip_special_tokens=True)[0].strip()
            write_transcript(out_dir, text, source=source, tokens=sequences[0].tolist())
            print(text)
    for hook in hooks:
        hook.remove()
    import json
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "reference-load.json").write_text(json.dumps(info, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

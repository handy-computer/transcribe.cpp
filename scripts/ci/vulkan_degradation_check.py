#!/usr/bin/env python3
"""Vulkan degradation-contract checks, shared by every CI lane that proves
"ship Vulkan by default, never crash without it".

    python vulkan_degradation_check.py --tier no-loader [--model M --audio A]
    python vulkan_degradation_check.py --tier loader    [--model M --audio A]

Tiers (one invocation per machine state):

  no-loader   The machine has NO Vulkan loader. The bundled module must stay
              quietly unloaded: CPU devices exist and answer available,
              vulkan answers unavailable. With --model: an explicit
              backend="vulkan" load must be a clean TranscribeError (never a
              crash), and a default-backend transcription still works on CPU.

  loader      A loader (+ driver, e.g. lavapipe or a real GPU) is present.
              The SAME artifacts must discover a Vulkan device. With
              --model: a real transcription runs with backend="vulkan".

Used by: provider-dl-vulkan (python-bindings.yml, hand-assembled provider
dir), clean-install (python-wheels.yml, the shipped wheel — ships in the
smoke-assets artifact for that checkout-less job). vulkan-hw keeps its
bespoke real-GPU + CPU-tier assertions inline.

Stdlib only (the clean-install container installs nothing but the wheels).
"""

from __future__ import annotations

import argparse
import array
import sys
import wave


def load_pcm(path: str) -> "array.array":
    with wave.open(path, "rb") as w:
        assert w.getsampwidth() == 2 and w.getframerate() == 16000, (
            f"{path} must be 16 kHz 16-bit"
        )
        pcm16 = array.array("h")
        pcm16.frombytes(w.readframes(w.getnframes()))
    if sys.byteorder == "big":
        pcm16.byteswap()
    return array.array("f", (s / 32768.0 for s in pcm16))


def transcribe(t, model: str, audio: str, backend: str | None) -> str:
    kwargs = {"backend": backend} if backend else {}
    with t.Model(model, **kwargs) as m:
        print(f"model backend: {m.backend}")
        with m.session() as s:
            return s.run(load_pcm(audio)).text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tier", required=True, choices=["no-loader", "loader"])
    ap.add_argument("--model", help="canary GGUF for the transcription half")
    ap.add_argument("--audio", help="16 kHz mono WAV (jfk.wav)")
    ap.add_argument("--expect-provider",
                    help="assert native_provider() equals this name")
    args = ap.parse_args()
    if bool(args.model) != bool(args.audio):
        ap.error("--model and --audio go together")

    import transcribe_cpp as t

    devs = [(d.name, d.kind) for d in t.backends()]
    print(f"devices ({args.tier}):", devs)
    print("provider:", t.native_provider())
    if args.expect_provider:
        assert t.native_provider() == args.expect_provider, t.native_provider()
    assert any(k == "cpu" for _, k in devs), devs
    assert t.backend_available("cpu")

    if args.tier == "no-loader":
        assert not t.backend_available("vulkan"), devs
        if args.model:
            # An explicit vulkan request without a loader must be a clean
            # Python exception, not a crash.
            try:
                t.Model(args.model, backend="vulkan")
            except t.TranscribeError as exc:
                print("vulkan request correctly rejected:", exc)
            else:
                raise AssertionError(
                    "backend='vulkan' must fail without a loader")
            text = transcribe(t, args.model, args.audio, backend=None)
            print("text (default backend):", text.strip())
            assert "country" in text.lower(), text
        print("ok: no loader — CPU fine, vulkan quietly unavailable")
    else:
        assert t.backend_available("vulkan"), devs
        assert any(k == "vulkan" for _, k in devs), devs
        if args.model:
            text = transcribe(t, args.model, args.audio, backend="vulkan")
            print("text (vulkan):", text.strip())
            assert "country" in text.lower(), text
        print("ok: loader present — the same artifacts discovered Vulkan")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

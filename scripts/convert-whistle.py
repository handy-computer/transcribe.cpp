#!/usr/bin/env python3
"""Convert Cactus Whistle (whistle.cact) to a reference-dtype GGUF.

Source: the deployed `whistle.cact` (user decision 2026-10-09; see
docs/porting/families/whistle.md). It is a nameless, positional Needle-3
archive. CQ (Cactus Quants, 2/4-bit Lloyd-Max + Walsh-Hadamard) matrices are
dequantized to F32 exactly as the format specifies (scripts/lib/cact.py);
FP16/FP32 records are carried by value. Every value in the GGUF is therefore
the value the engine's weights decode to.

Tensor identity. The record order of the text decoder follows
cactus-compute/needle needle/model/export.py::_tensors (embedding, per-layer
self-attention + HadamardMLP, mHC, Hadamard permutations, Engram sites, final
norm). The audio records that follow (speech config, cross-attention,
pe_gate, encoder layers, encoder mHC, encoder final norm, conv stem, mel
filterbank) have no published order; their layout below was established by
matching every record against the named F32 QAT masters in
checkpoints/whistle.safetensors and is re-proved on every run by
--verify-masters (default on when the masters are present):
  - FP16 / FP32 records must equal float16(master) exactly
  - CQ records must re-quantize from the master to identical codes and norms
    (the export algorithm, applied to the master slice)
GGUF names stay close to the upstream parameter names; no block semantics
(pre/post, macaron order) are asserted here.

Output: models/<slug>/<slug>-F32.gguf (llama.cpp naming). With --hadamard-domain:
models/<slug>/<slug>-HR-F32.gguf, every CQ matrix kept in its Walsh-Hadamard domain
(the input to `transcribe-quantize --quant Q2_K_HR`; see docs/porting/families/whistle.md).

Usage:
    uv run --project scripts/envs/whistle scripts/convert-whistle.py \\
        Cactus-Compute/whistle --repo-id Cactus-Compute/whistle \\
        --revision b358ddadd89b7a713b5aa131f23032d3cca1b251
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from gguf import GGMLQuantizationType  # noqa: E402

from lib.cact import CQ, FP16, FP32, RAW, Cact, _hadamard  # noqa: E402
from lib.gguf_common import (  # noqa: E402
    TOKEN_TYPE_BYTE,
    TOKEN_TYPE_CONTROL,
    TOKEN_TYPE_NORMAL,
    TOKEN_TYPE_UNKNOWN,
    TOKEN_TYPE_USER,
    add_general_identity,
    encode_for_gguf,
    gguf_name,
    gguf_writer,
    reference_dtype_for,
    slug_from_repo_id,
)

DEFAULT_REPO = "Cactus-Compute/whistle"
DEFAULT_REVISION = "b358ddadd89b7a713b5aa131f23032d3cca1b251"
LANGUAGES = ["en", "de", "fr", "es", "it", "nl", "pl"]
REFERENCE_TYPE = GGMLQuantizationType.F32

# SentencePiece piece type -> llama.cpp token_type.
_SP_TYPE = {0: TOKEN_TYPE_NORMAL, 1: TOKEN_TYPE_UNKNOWN, 2: TOKEN_TYPE_CONTROL,
            3: TOKEN_TYPE_USER, 4: TOKEN_TYPE_BYTE}

_HMLP = ("d1", "d2", "b2", "d3", "d4", "w1a", "w1b", "w2a", "w2b", "w3a", "w3b", "cond_v", "cond_u")
_MHC = (("a_pre", None), ("a_post", None), ("a_res", None), ("b_pre", None), ("b_post", None),
        ("b_res", None), ("phi_pre", ".weight"), ("phi_post", ".weight"), ("phi_res", ".weight"))


def _hmlp(prefix: str, gguf_prefix: str, mod: str, layer: int) -> list:
    return [(f"{gguf_prefix}.{mod}.{p}", f"{prefix}/{'hadamard_mlp' if mod == 'hmlp' else 'hadamard_mlp_0'}/{p}", layer)
            for p in _HMLP]


def build_template(c: Cact, n_enc_layers: int) -> list:
    """(gguf_name | special, master_key | None, layer | None) for every record, in order."""
    L = c.header["num_layers"]
    taps = c.header["qkv_conv_taps"]
    n_sites = len(c.header["engram_sites"])
    sb = "stack/layers/block"
    eb = "encoder/layers/block"
    t = [("dec.token_embd.weight", "embedding/embedding", None)]
    for i in range(L):
        p = f"dec.blocks.{i}"
        t += [(f"{p}.norm_in.weight", f"{sb}/ZCRMSNorm_0/scale", i),
              (f"{p}.attn.q.weight", f"{sb}/self_attn/q_proj/kernel", i),
              (f"{p}.attn.k.weight", f"{sb}/self_attn/k_proj/kernel", i),
              (f"{p}.attn.v.weight", f"{sb}/self_attn/v_proj/kernel", i)]
        if taps:
            t += [(f"{p}.attn.{x}_taps", f"{sb}/self_attn/{x}_taps", i) for x in "qkv"]
        t += [(f"{p}.attn.q_norm.weight", f"{sb}/self_attn/q_norm/scale", i),
              (f"{p}.attn.k_norm.weight", f"{sb}/self_attn/k_norm/scale", i),
              (f"{p}.attn.gate.weight", f"{sb}/self_attn/gate_proj/kernel", i),
              (f"{p}.attn.out.weight", f"{sb}/self_attn/out_proj/kernel", i),
              (f"{p}.norm_post_attn.weight", f"{sb}/post_attn_norm/scale", i),
              (f"{p}.attn_gate", f"{sb}/attn_gate", i),
              (f"{p}.norm_hmlp.weight", f"{sb}/pre_hada_norm/scale", i)]
        t += _hmlp(sb, p, "hmlp", i)
    t += [(f"dec.mhc.{n}{sfx or ''}", f"stack/mhc_{n}", None) for n, sfx in _MHC]
    t += [("hada.perm1", None, None), ("hada.perm2", None, None)]
    for s in range(n_sites):
        p = f"dec.engram.{s}"
        t += [(f"{p}.tables.weight", f"engrams_{s}/embedding", None),
              (f"{p}.key_proj.weight", f"engrams_{s}/key_proj/kernel", None),
              (f"{p}.value_proj.weight", f"engrams_{s}/value_proj/kernel", None),
              (f"{p}.conv_taps", f"engrams_{s}/taps", None)]
    t += [("dec.final_norm.weight", "stack/final_norm/scale", None),
          ("@speech_config", None, None)]
    for i in range(L):
        p = f"dec.blocks.{i}"
        t += [(f"{p}.norm_cross.weight", f"{sb}/cross_norm/scale", i),
              (f"{p}.cross.q.weight", f"{sb}/cross_attn/q_proj/kernel", i),
              (f"{p}.cross.k.weight", f"{sb}/cross_attn/k_proj/kernel", i),
              (f"{p}.cross.v.weight", f"{sb}/cross_attn/v_proj/kernel", i),
              (f"{p}.cross.gate.weight", f"{sb}/cross_attn/gate_proj/kernel", i),
              (f"{p}.cross.out.weight", f"{sb}/cross_attn/out_proj/kernel", i),
              (f"{p}.cross.q_norm.weight", f"{sb}/cross_attn/q_norm/scale", i),
              (f"{p}.cross.k_norm.weight", f"{sb}/cross_attn/k_norm/scale", i),
              (f"{p}.norm_post_cross.weight", f"{sb}/post_cross_norm/scale", i),
              (f"{p}.cross_gate", f"{sb}/cross_gate", i)]
    t += [("dec.pe_gate", "pe_gate", None)]
    for i in range(n_enc_layers):
        p = f"enc.blocks.{i}"
        t += [(f"{p}.norm_hmlp_0.weight", f"{eb}/pre_hada_norm_0/scale", i),
              (f"{p}.norm_in.weight", f"{eb}/ZCRMSNorm_0/scale", i),
              (f"{p}.norm_post_attn.weight", f"{eb}/post_attn_norm/scale", i),
              (f"{p}.norm_conv.weight", f"{eb}/conv_norm/scale", i),
              (f"{p}.norm_conv_out.weight", f"{eb}/conv_out_norm/scale", i),
              (f"{p}.norm_hmlp.weight", f"{eb}/pre_hada_norm/scale", i),
              (f"{p}.attn.q.weight", f"{eb}/self_attn/q_proj/kernel", i),
              (f"{p}.attn.k.weight", f"{eb}/self_attn/k_proj/kernel", i),
              (f"{p}.attn.v.weight", f"{eb}/self_attn/v_proj/kernel", i),
              (f"{p}.attn.gate.weight", f"{eb}/self_attn/gate_proj/kernel", i),
              (f"{p}.attn.out.weight", f"{eb}/self_attn/out_proj/kernel", i),
              (f"{p}.attn.q_norm.weight", f"{eb}/self_attn/q_norm/scale", i),
              (f"{p}.attn.k_norm.weight", f"{eb}/self_attn/k_norm/scale", i),
              (f"{p}.attn_gate", f"{eb}/attn_gate", i),
              (f"{p}.conv_pw1.weight", f"{eb}/pw1/kernel", i),
              (f"{p}.conv_pw2.weight", f"{eb}/pw2/kernel", i),
              (f"{p}.conv.dw.weight", f"{eb}/dw", i)]
        t += _hmlp(eb, p, "hmlp_0", i)
        t += _hmlp(eb, p, "hmlp", i)
    t += [(f"enc.mhc.{n}{sfx or ''}", f"encoder/mhc_{n}", None) for n, sfx in _MHC]
    t += [("enc.final_norm.weight", "encoder/final_norm/scale", None),
          ("enc.stem.conv.w.weight", "stem/w", None),
          ("enc.stem.conv.dw_1.weight", "stem/dw_1", None),
          ("enc.stem.pw_1.weight", "stem/pw_1/kernel", None),
          ("enc.stem.conv.dw_2.weight", "stem/dw_2", None),
          ("enc.stem.pw_2.weight", "stem/pw_2/kernel", None),
          ("enc.stem.out.weight", "stem/out/kernel", None),
          ("frontend.mel_filterbank", None, None),
          ("@tokenizer", None, None)]
    return t


# ---------------------------------------------------------------- masters ---

class Masters:
    """Named F32 QAT masters (checkpoints/whistle.safetensors), viewed in .cact layout."""

    def __init__(self, path: Path):
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.hdr = json.loads(f.read(n))
            self.hdr.pop("__metadata__", None)
            self.data = f.read()

    def get(self, key: str, layer: int | None) -> np.ndarray:
        v = self.hdr[key]
        a, b = v["data_offsets"]
        x = np.frombuffer(self.data, "<f4", (b - a) // 4, a).reshape(v["shape"])
        if layer is not None:
            x = x[layer]
        if key.endswith("/kernel"):
            return x.T
        if "mhc_phi" in key:
            n_layers, n_c, lanes = x.shape
            return x.transpose(0, 2, 1).reshape(n_layers * lanes, n_c)
        if key.startswith("engrams_") and key.endswith("/embedding"):
            return x.reshape(-1, x.shape[-1])
        if key.startswith("stem/") or key.endswith("block/dw"):
            return x.reshape(-1, x.shape[-1])
        return x.reshape(1) if x.ndim == 0 else x


def _requant_codes(c: Cact, rec, w: np.ndarray):
    """The export algorithm (needle export.py::_cq_pack) on a master slice."""
    g, bits = rec.group, rec.bits
    out, d = w.shape
    pad = (-d) % g
    wp = np.pad(w, ((0, 0), (0, pad))) if pad else w
    rot = wp.reshape(out, -1, g).astype(np.float32) @ _hadamard(g).astype(np.float32)
    norm = np.sqrt((rot ** 2).sum(-1, keepdims=True))
    unit = rot / np.maximum(norm, 1e-12)
    cb = c.codebooks[bits].astype(np.float32)
    flat = unit.reshape(-1)
    pos = np.clip(np.searchsorted(cb, flat), 1, len(cb) - 1)
    left, right = cb[pos - 1], cb[pos]
    idx = np.where(np.abs(flat - left) <= np.abs(flat - right), pos - 1, pos)
    return idx.reshape(out, -1), norm[..., 0].astype(np.float16)


def _cact_codes(c: Cact, rec):
    from lib.cact import _unpack_lsb
    out, d = rec.shape
    in_pad = (d + rec.group - 1) // rec.group * rec.group
    n_packed = out * in_pad * rec.bits // 8
    blob = c.raw(rec.index)
    packed = np.frombuffer(blob, np.uint8, n_packed).reshape(out, -1)
    norms = np.frombuffer(blob, "<f2", out * in_pad // rec.group, n_packed).reshape(out, -1)
    return _unpack_lsb(packed, rec.bits, in_pad), norms


def verify_masters(c: Cact, template: list, masters: Masters) -> dict:
    """Prove the positional map: exact FP16 equality, exact CQ re-quantization."""
    report = {"checked": 0, "fp_exact": 0, "cq_code_match": [], "failures": []}
    for rec, (name, key, layer) in zip(c.records, template):
        if key is None:
            continue
        m = masters.get(key, layer).astype(np.float32)
        if tuple(m.shape) != tuple(rec.shape):
            report["failures"].append(f"{rec.index} {name}: shape {rec.shape} != master {m.shape}")
            continue
        report["checked"] += 1
        if rec.dtype == FP16:
            if np.array_equal(c.array(rec.index), m.astype(np.float16)):
                report["fp_exact"] += 1
            else:
                diff = np.abs(c.array(rec.index).astype(np.float32) - m).max()
                report["failures"].append(f"{rec.index} {name}: FP16 != f16(master), max diff {diff:.3g}")
        elif rec.dtype == CQ:
            idx_m, nrm_m = _requant_codes(c, rec, m)
            idx_c, nrm_c = _cact_codes(c, rec)
            frac = float((idx_m == idx_c).mean())
            norm_rel = float(np.abs(nrm_m.astype(np.float32) - nrm_c.astype(np.float32)).max()
                             / max(np.abs(nrm_c.astype(np.float32)).max(), 1e-12))
            report["cq_code_match"].append((rec.index, name, frac, norm_rel))
            if frac < 0.999 or norm_rel > 1e-2:
                report["failures"].append(f"{rec.index} {name}: CQ codes match {frac:.4f}, norm rel {norm_rel:.3g}")
        else:
            report["failures"].append(f"{rec.index} {name}: unexpected dtype {rec.kind} for a mastered tensor")
    return report


# ---------------------------------------------------------------- convert ---

def _resolve(source: str, revision: str) -> tuple[Path, Path | None]:
    p = Path(source)
    if p.suffix == ".cact" and p.exists():
        st = p.parent / "checkpoints" / "whistle.safetensors"
        return p, (st if st.exists() else None)
    if p.is_dir():
        st = p / "checkpoints" / "whistle.safetensors"
        return p / "whistle.cact", (st if st.exists() else None)
    from huggingface_hub import hf_hub_download
    cact = Path(hf_hub_download(source, "whistle.cact", revision=revision))
    st = Path(hf_hub_download(source, "checkpoints/whistle.safetensors", revision=revision))
    return cact, st


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", nargs="?", default=DEFAULT_REPO,
                    help="HF repo id, a snapshot directory, or a whistle.cact path")
    ap.add_argument("--repo-id", default=DEFAULT_REPO, help="derives the output slug")
    ap.add_argument("--revision", default=DEFAULT_REVISION)
    ap.add_argument("--out", type=Path, default=None, help="override output path")
    ap.add_argument("--no-verify-masters", action="store_true",
                    help="skip the positional-map proof against checkpoints/whistle.safetensors")
    ap.add_argument("--report", type=Path, default=None, help="write the verification report JSON here")
    ap.add_argument("--hadamard-domain", action="store_true",
                    help="store every CQ matrix in its Walsh-Hadamard domain (codebook[idx] * norm, the "
                         "archive's codes before the inverse rotation) and set stt.whistle.hadamard_group; "
                         "the runtime then rotates the matching activations. Experimental (low-bit quants)")
    args = ap.parse_args()

    cact_path, masters_path = _resolve(args.source, args.revision)
    c = Cact(cact_path)
    h = c.header
    print(f"source: {cact_path}  ({len(c.records)} records)")

    # Speech config vector: [?, ?, lang_base, enc_layers, enc_lanes, conv_kernel, stem, stem_stages,
    #                        mels, rate, fft, win, hop]
    sc_idx = 1 + h["num_layers"] * (27 if h["qkv_conv_taps"] else 24) + 9 + 2 + 4 * len(h["engram_sites"]) + 1
    sc = c.array(sc_idx)
    if c.records[sc_idx].dtype != FP32 or sc.shape != (13,):
        raise SystemExit(f"error: record {sc_idx} is not the 13-value speech config vector")
    sc = [int(round(v)) for v in sc]
    n_enc_layers = sc[3]
    template = build_template(c, n_enc_layers)
    if len(template) != len(c.records):
        raise SystemExit(f"error: template has {len(template)} entries, archive has {len(c.records)}")
    if template[sc_idx][0] != "@speech_config":
        raise SystemExit("error: speech config position disagrees with the template")
    for rec, (name, _, _) in zip(c.records, template):
        expected = {"@tokenizer": RAW, "@speech_config": FP32, "frontend.mel_filterbank": FP32,
                    "hada.perm1": FP32, "hada.perm2": FP32}.get(name)
        if expected is not None and rec.dtype != expected:
            raise SystemExit(f"error: record {rec.index} ({name}) is {rec.kind}, expected dtype {expected}")

    verification = None
    if not args.no_verify_masters:
        if masters_path is None:
            raise SystemExit("error: checkpoints/whistle.safetensors not found; pass --no-verify-masters to skip")
        verification = verify_masters(c, template, Masters(masters_path))
        cq = verification["cq_code_match"]
        print(f"verify: {verification['checked']} mastered records; FP exact {verification['fp_exact']}; "
              f"CQ min code match {min(x[2] for x in cq):.5f} over {len(cq)} records")
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(verification, indent=2) + "\n")
        if verification["failures"]:
            for f in verification["failures"][:20]:
                print("  FAIL", f)
            raise SystemExit(f"error: {len(verification['failures'])} records failed master verification")

    tok = c.tokenizer()
    lang_ids = [tok["pieces"].index(f"<|{l}|>") for l in LANGUAGES]
    if lang_ids != list(range(sc[2], sc[2] + len(LANGUAGES))):
        raise SystemExit(f"error: language tokens {lang_ids} do not start at lang base {sc[2]}")

    slug = slug_from_repo_id(args.repo_id)
    # The Hadamard-domain file is an intermediate for transcribe-quantize --quant Q2_K_HR.
    out = args.out or (REPO_ROOT / "models" / slug / gguf_name(slug, "HR-F32" if args.hadamard_domain else "F32"))
    out.parent.mkdir(parents=True, exist_ok=True)
    w = gguf_writer(str(out), "whistle")
    add_general_identity(
        w, name="Whistle", basename="whistle", size_label="55M",
        file_type=int(GGMLQuantizationType.F32), languages=LANGUAGES,
        author="Cactus Compute", organization="Cactus-Compute", version="2.0.0",
        license="apache-2.0", repo_url=f"https://huggingface.co/{args.repo_id}",
        source_url="https://github.com/cactus-compute/needle",
        description="Whistle speech-to-text (log-mel + conv stem + 8-layer encoder, 8-layer Needle-3 "
                    "decoder with gated cross-attention). Weights dequantized from the deployed CQ "
                    "whistle.cact.")
    w.add_string("stt.variant", slug)
    w.add_string("stt.source.hf_revision", args.revision)
    w.add_bool("stt.capability.lang_detect", True)
    w.add_bool("stt.capability.translate", False)
    w.add_bool("stt.capability.streaming", False)
    w.add_bool("stt.capability.speaker_diarization", False)
    w.add_bool("stt.capability.word_timestamps", True)

    # Frontend: only the facts the archive states. Window, log, normalization and
    # centering are engine-internal and resolved at Stage 4 bring-up.
    w.add_string("stt.frontend.type", "mel")
    w.add_uint32("stt.frontend.sample_rate", sc[9])
    w.add_uint32("stt.frontend.num_mels", sc[8])
    w.add_uint32("stt.frontend.n_fft", sc[10])
    w.add_uint32("stt.frontend.win_length", sc[11])
    w.add_uint32("stt.frontend.hop_length", sc[12])
    w.add_string("stt.frontend.mel_norm", "slaney")

    # Architecture geometry (from the .cact header and speech config vector).
    W = "stt.whistle"
    w.add_uint32(f"{W}.d_model", h["d_model"])
    w.add_uint32(f"{W}.n_heads", h["num_heads"])
    w.add_uint32(f"{W}.n_kv_heads", h["num_kv_heads"])
    w.add_uint32(f"{W}.qk_head_dim", h["qk_head_dim"])
    w.add_uint32(f"{W}.v_head_dim", h["v_head_dim"])
    w.add_uint32(f"{W}.decoder.n_layers", h["num_layers"])
    w.add_uint32(f"{W}.decoder.max_seq_len", h["max_seq_len"])
    w.add_uint32(f"{W}.decoder.qkv_conv_taps", h["qkv_conv_taps"])
    w.add_uint32(f"{W}.encoder.n_layers", n_enc_layers)
    w.add_uint32(f"{W}.encoder.conv_kernel", sc[5])
    w.add_uint32(f"{W}.stem.channels", sc[6])
    w.add_uint32(f"{W}.stem.stages", sc[7])
    w.add_uint32(f"{W}.mhc_lanes", h["mhc_lanes"])
    w.add_uint32(f"{W}.hada_n", h["hada_n"])
    w.add_float32(f"{W}.rope_theta", h["rope_theta"])
    w.add_uint32(f"{W}.vocab_size", h["vocab"])
    w.add_uint32(f"{W}.kv_bits", h["kv_bits"])
    w.add_uint32(f"{W}.engram.slots", h["engram_slots"])
    w.add_uint32(f"{W}.engram.sub_dim", h["engram_sub_dim"])
    w.add_uint32(f"{W}.engram.n_tables", h["num_engram_tables"])
    w.add_uint32(f"{W}.engram.conv_taps", h["engram_conv_taps"])
    w.add_uint32(f"{W}.engram.conv_dilation", h["engram_conv_dilation"])
    w.add_uint32(f"{W}.engram.seed_heads", h["engram_seed_heads"])
    w.add_array(f"{W}.engram.orders", [int(x) for x in h["engram_orders"]])
    w.add_array(f"{W}.engram.sites", [int(x) for x in h["engram_sites"]])
    w.add_array(f"{W}.speech_config", sc)
    w.add_uint32(f"{W}.max_audio_samples", 30 * sc[9])
    w.add_uint32(f"{W}.lang_token_base", sc[2])
    w.add_array(f"{W}.languages", LANGUAGES)
    w.add_array(f"{W}.lang_token_ids", lang_ids)
    hada_group = 128
    if args.hadamard_domain:
        # Every CQ-sourced matrix W is stored as C = W @ blockdiag(H_g) along its input axis
        # (H_g = Walsh-Hadamard(g) / sqrt(g), symmetric and involutory), so W x = C (H_g x).
        w.add_uint32(f"{W}.hadamard_group", hada_group)

    # Tokenizer: the archive's SentencePiece BPE dump.
    w.add_string("tokenizer.ggml.model", "bpe")
    w.add_array("tokenizer.ggml.tokens", tok["pieces"])
    w.add_array("tokenizer.ggml.scores", [float(s) for s in tok["scores"]])
    w.add_array("tokenizer.ggml.token_type", [_SP_TYPE[t] for t in tok["types"]])
    w.add_bool("tokenizer.ggml.byte_fallback", bool(tok["byte_fallback"]))
    w.add_bool("tokenizer.ggml.add_space_prefix", bool(tok["add_dummy_prefix"]))
    w.add_uint32("tokenizer.ggml.bos_token_id", tok["bos"])
    w.add_uint32("tokenizer.ggml.eos_token_id", tok["eos"])
    w.add_uint32("tokenizer.ggml.padding_token_id", tok["pad"])
    w.add_uint32("tokenizer.ggml.unknown_token_id", tok["unk"])

    n_written = 0
    for rec, (name, _, _) in zip(c.records, template):
        if name.startswith("@"):
            continue
        if args.hadamard_domain and rec.dtype == CQ:
            idx, norms = _cact_codes(c, rec)
            out_dim, in_dim = rec.shape
            if in_dim % rec.group or rec.group != hada_group:
                raise SystemExit(f"error: record {rec.index} ({name}) group {rec.group} / in {in_dim} "
                                 f"is not {hada_group}-aligned")
            unit = c.codebooks[rec.bits][idx].reshape(out_dim, -1, rec.group)
            arr = (unit * norms.astype(np.float64)[..., None]).reshape(out_dim, in_dim).astype(np.float32)
        else:
            arr = c.array(rec.index).astype(np.float32)
        ttype = reference_dtype_for(name, REFERENCE_TYPE)
        data, ttype = encode_for_gguf(np.ascontiguousarray(arr), ttype)
        w.add_tensor(name, data, raw_dtype=ttype)
        n_written += 1

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out} ({n_written} tensors, {out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

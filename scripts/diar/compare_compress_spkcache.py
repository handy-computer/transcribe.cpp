#!/usr/bin/env python3
"""
compare_compress_spkcache.py - Verify the Sortformer AOSC `_compress_spkcache`
step bit-for-bit between the NeMo reference and the transcribe.cpp C++ port
(residual-uncertainty #1).

Both sides run the SAME clip at the SAME streaming preset, so the sequence of
compression calls lines up 1:1. For each call we compare:

  * frame_idx    the sorted selected frame indices (C++ `frame_idx` vs NeMo
                 `_get_topk_indices` -> `topk_indices_sorted`, post-remainder,
                 disabled entries set to 0). An EXACT match confirms the topk
                 tie-break hypothesis (ATen CPU topk: value desc, flat-index
                 asc; -inf picks -> max_index before the ascending sort) on real
                 drifted scores.
  * is_disabled  the disabled mask (silence-pad / -inf placeholder frames).
  * spkcache_preds  the gathered [spkcache_len, n_spk] preds (numeric; small
                 F32 drift expected, reported as max_abs).

NeMo dump: `compress.{k:03d}.{topk_indices,is_disabled,spkcache_preds}.npy`
           (scripts/dump_reference_sortformer_nemo.py diarize --dump-compress).
C++ dump:  `compress.{k:03d}.{frame_idx,is_disabled,spkcache_preds}.{f32,json}`
           (TRANSCRIBE_SORTFORMER_COMPRESS_DUMP=1 + TRANSCRIBE_DUMP_DIR).

Usage:
  uv run scripts/diar/compare_compress_spkcache.py \
    --ref-dir  build/validate/sortformer/.../compress-ref \
    --cpp-dir  build/validate/sortformer/.../compress-cpp
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _boost_topk(scores: np.ndarray, k: int, scale: float) -> np.ndarray:
    """Increase the k highest scores per speaker column. Tie-break: value desc,
    frame-index asc (ATen CPU topk). Mirrors _boost_topk_scores."""
    if k <= 0:
        return scores
    delta = scale * np.log(0.5)  # negative -> subtracting boosts
    n = scores.shape[0]
    for s in range(scores.shape[1]):
        col = scores[:, s]
        order = np.lexsort((np.arange(n), -col))  # primary value desc, secondary idx asc
        col[order[:min(k, n)]] -= delta
    return scores


def select_indices(preds: np.ndarray, spkcache_len: int, sil_per_spk: int = 3,
                   pred_thr: float = 0.25, boost_latest: float = 0.05,
                   strong_rate: float = 0.75, weak_rate: float = 1.5,
                   min_pos_rate: float = 0.5, max_index: int = 99999):
    """Independent numpy reimplementation of the Sortformer AOSC frame selection
    (_get_log_pred_scores -> _disable_low_scores -> scores_boost_latest ->
    _boost_topk_scores x2 -> silence pad -> _get_topk_indices). Returns
    (frame_idx [spkcache_len], is_disabled [spkcache_len]). A third, neutral
    arbiter for both the NeMo and C++ dumps."""
    N, n_spk = preds.shape
    per_spk = spkcache_len // n_spk - sil_per_spk
    import math
    strong = math.floor(per_spk * strong_rate)
    weak = math.floor(per_spk * weak_rate)
    min_pos = math.floor(per_spk * min_pos_rate)

    lp = np.log(np.clip(preds, pred_thr, None))
    l1 = np.log(np.clip(1.0 - preds, pred_thr, None))
    scores = lp - l1 + l1.sum(1, keepdims=True) - math.log(0.5)

    is_speech = preds > 0.5
    scores = np.where(is_speech, scores, -np.inf)
    is_pos = scores > 0
    n_pos = is_pos.sum(0, keepdims=True)
    scores = np.where((~is_pos) & is_speech & (n_pos >= min_pos), -np.inf, scores)

    if boost_latest > 0:
        scores[spkcache_len:, :] += boost_latest  # -inf + x stays -inf

    scores = _boost_topk(scores, strong, 2.0)
    scores = _boost_topk(scores, weak, 1.0)

    if sil_per_spk > 0:
        scores = np.concatenate([scores, np.full((sil_per_spk, n_spk), np.inf)], axis=0)
    n_frames = scores.shape[0]
    n_frames_no_sil = n_frames - sil_per_spk

    flat = scores.T.reshape(-1)  # flat index f = s * n_frames + i
    order = np.lexsort((np.arange(flat.size), -flat))  # value desc, flat-idx asc
    topk = order[:spkcache_len].copy()
    topk = np.where(flat[topk] != -np.inf, topk, max_index)
    topk_sorted = np.sort(topk)
    is_disabled = topk_sorted == max_index
    frame = np.remainder(topk_sorted, n_frames)
    is_disabled = is_disabled | (frame >= n_frames_no_sil)
    frame = np.where(is_disabled, 0, frame)
    return frame.astype(np.int64), is_disabled.astype(bool)


def _load_cpp_f32(cpp_dir: Path, stem: str) -> np.ndarray:
    """Load a C++ dump_host_f32 artifact (.f32 payload + .json shape)."""
    meta = json.loads((cpp_dir / f"{stem}.json").read_text())
    shape = tuple(int(x) for x in meta["shape"])  # slow-to-fast
    data = np.fromfile(cpp_dir / f"{stem}.f32", dtype="<f4")
    if data.size != int(np.prod(shape)):
        raise SystemExit(f"error: {stem} size {data.size} != shape {shape} product")
    return data.reshape(shape).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref-dir", required=True, help="NeMo --dump-compress output dir")
    ap.add_argument("--cpp-dir", required=True, help="C++ TRANSCRIBE_DUMP_DIR (compress.* artifacts)")
    ap.add_argument("--preds-tol", type=float, default=1e-2,
                    help="Max allowed |Δ| on gathered spkcache_preds (F32 drift).")
    ap.add_argument("--spkcache-len", type=int, default=188,
                    help="spkcache_len for the numpy adjudicator (188 for the shipped presets, "
                         "24 for `small`). Only used when input_preds dumps are present.")
    args = ap.parse_args()

    ref_dir, cpp_dir = Path(args.ref_dir), Path(args.cpp_dir)
    ref_calls = sorted(ref_dir.glob("compress.*.topk_indices.npy"))
    cpp_calls = sorted(cpp_dir.glob("compress.*.frame_idx.json"))
    n_ref, n_cpp = len(ref_calls), len(cpp_calls)
    print(f"compression calls: ref={n_ref} cpp={n_cpp}")
    if n_ref == 0:
        raise SystemExit("error: no reference compression dumps found "
                         "(did the preset actually trigger AOSC compression?)")
    if n_ref != n_cpp:
        print(f"MISMATCH: call count differs (ref {n_ref} vs cpp {n_cpp}) -> streaming "
              "trajectories diverged before/around compression", file=sys.stderr)
        return 1

    all_idx_ok = True
    all_dis_ok = True
    max_preds = 0.0
    # Adjudicator bookkeeping: does each side's own dumped selection match an
    # independent numpy reimplementation run on that side's own input preds?
    ref_self_ok = cpp_self_ok = True
    ref_self_seen = cpp_self_seen = False
    first_traj_div = None  # first call where ref/cpp selections diverge
    for k in range(n_ref):
        ref_idx = np.load(ref_dir / f"compress.{k:03d}.topk_indices.npy").astype(np.int64)
        ref_dis = np.load(ref_dir / f"compress.{k:03d}.is_disabled.npy").astype(bool)
        cpp_idx = _load_cpp_f32(cpp_dir, f"compress.{k:03d}.frame_idx").astype(np.int64)
        cpp_dis = _load_cpp_f32(cpp_dir, f"compress.{k:03d}.is_disabled").astype(bool)

        # Independent-arbiter check: run the numpy selection on each side's own
        # input preds and confirm it reproduces that side's dumped selection.
        ref_ipf = ref_dir / f"compress.{k:03d}.input_preds.npy"
        cpp_ipf = cpp_dir / f"compress.{k:03d}.input_preds.json"
        if ref_ipf.exists():
            ref_self_seen = True
            nf, nd = select_indices(np.load(ref_ipf).astype(np.float32), args.spkcache_len)
            if int(np.sum(nf != ref_idx)) or int(np.sum(nd != ref_dis)):
                ref_self_ok = False
                print(f"  [{k:03d}] numpy-arbiter != NeMo dump "
                      f"(idx {int(np.sum(nf != ref_idx))}, dis {int(np.sum(nd != ref_dis))})", file=sys.stderr)
        if cpp_ipf.exists():
            cpp_self_seen = True
            nf, nd = select_indices(_load_cpp_f32(cpp_dir, f"compress.{k:03d}.input_preds"), args.spkcache_len)
            if int(np.sum(nf != cpp_idx)) or int(np.sum(nd != cpp_dis)):
                cpp_self_ok = False
                print(f"  [{k:03d}] numpy-arbiter != C++ dump "
                      f"(idx {int(np.sum(nf != cpp_idx))}, dis {int(np.sum(nd != cpp_dis))})", file=sys.stderr)

        if ref_idx.shape != cpp_idx.shape:
            print(f"  [{k:03d}] SHAPE MISMATCH idx ref{ref_idx.shape} cpp{cpp_idx.shape}", file=sys.stderr)
            all_idx_ok = False
            continue
        idx_mism = int(np.sum(ref_idx != cpp_idx))
        dis_mism = int(np.sum(ref_dis != cpp_dis))
        all_idx_ok = all_idx_ok and idx_mism == 0
        all_dis_ok = all_dis_ok and dis_mism == 0
        if idx_mism and first_traj_div is None:
            first_traj_div = k

        preds_line = ""
        ref_pf = ref_dir / f"compress.{k:03d}.spkcache_preds.npy"
        cpp_pf = cpp_dir / f"compress.{k:03d}.spkcache_preds.json"
        if ref_pf.exists() and cpp_pf.exists():
            ref_p = np.load(ref_pf).astype(np.float32)
            cpp_p = _load_cpp_f32(cpp_dir, f"compress.{k:03d}.spkcache_preds")
            if ref_p.shape == cpp_p.shape:
                m = float(np.max(np.abs(ref_p - cpp_p))) if ref_p.size else 0.0
                max_preds = max(max_preds, m)
                preds_line = f"  preds_max_abs {m:.2e}"
            else:
                preds_line = f"  preds SHAPE MISMATCH ref{ref_p.shape} cpp{cpp_p.shape}"

        flag = "OK " if (idx_mism == 0 and dis_mism == 0) else "*** "
        print(f"  [{k:03d}] {flag}n={ref_idx.shape[0]:4d} idx_mismatch={idx_mism:4d} "
              f"dis_mismatch={dis_mism:4d}{preds_line}")

    print()
    print("== end-to-end trajectory (each side fed its OWN drifted preds) ==")
    print(f"frame_idx    : {'MATCH (bit-exact)' if all_idx_ok else 'MISMATCH'}"
          + ("" if all_idx_ok else f" (first divergence at call {first_traj_div})"))
    print(f"is_disabled  : {'MATCH (bit-exact)' if all_dis_ok else 'MISMATCH'}")
    print(f"spkcache_preds max_abs: {max_preds:.2e} (tol {args.preds_tol:.1e})")

    # The definitive residual-#1 check: is the compression FUNCTION correct?
    # Independent numpy arbiter vs each side's own dump on that side's own input.
    print()
    print("== compression function (independent numpy arbiter, per-side own input) ==")
    if ref_self_seen:
        print(f"NeMo  selection == numpy arbiter : {'MATCH' if ref_self_ok else 'MISMATCH'}")
    if cpp_self_seen:
        print(f"C++   selection == numpy arbiter : {'MATCH' if cpp_self_ok else 'MISMATCH'}")
    if not (ref_self_seen and cpp_self_seen):
        print("(input_preds dumps absent on one/both sides -> function-level arbiter skipped; "
              "re-dump with the updated hooks to enable it)")

    # PASS criteria: if input_preds are present, the compression function is
    # proven bit-exact against a neutral arbiter on both sides' real inputs
    # (trajectory drift is expected and does NOT fail the check). If they are
    # absent, fall back to end-to-end bit-exactness.
    print()
    if ref_self_seen and cpp_self_seen:
        ok = ref_self_ok and cpp_self_ok
        note = ("compression function bit-exact on both sides vs neutral arbiter"
                if ok else "compression function disagrees with arbiter")
        if ok and not all_idx_ok:
            note += (f"; end-to-end trajectories diverge from call {first_traj_div} "
                     "purely via upstream F32 preds drift (near-tie topk flips)")
    else:
        ok = all_idx_ok and all_dis_ok and max_preds <= args.preds_tol
        note = "compression internal state matches" if ok else "internal state mismatch"
    print("VERDICT:", ("PASS - " + note) if ok else ("FAIL - " + note))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

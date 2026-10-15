"""Numpy reference model for Whistle (reference for intermediate tensors).

The Whistle oracle is the closed Cactus Needle engine, which exposes only the
encoder output (needle_embed) and the final decode. This module is the
float64 forward pass that was identified against that engine by weight
surgery and black-box probing (reports/porting/whistle/forward-map.md); it
matches the engine's encoder output at the engine's own 8-bit noise floor
(median per-frame cosine >= 0.9999 on jfk, fleurs-de/fr/es, noise,
jobs-silence) and reproduces its transcripts and per-word probabilities.
User decision 2026-10-09: it serves as the reference for the intermediate
gate tensors; enc.final and transcripts stay engine-sourced.

Weights are read from the deployed whistle.cact (scripts/lib/cact.py), the
same values the GGUF carries. Needle-3 building blocks follow
cactus-compute/needle needle/model/architecture.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .cact import Cact

D_MODEL = 512
N_HEADS, N_KV, QK, VD = 8, 2, 48, 64
LANES = 4
ROPE_THETA = 1e5
ENGRAM_SEED, ENGRAM_PRIME = 0x9E3779B9, 0x01000193
ENGRAM_ORDERS, ENGRAM_HEADS, ENGRAM_SLOTS, ENGRAM_TAPS = (2, 3), 2, 18432, 4
ENGRAM_SITES = {3: 0, 7: 1}


# ---------------------------------------------------------------------------
# Weights: positional .cact records named by scripts/convert-whistle.py
# ---------------------------------------------------------------------------

class Weights:
    def __init__(self, cact_path: str | Path):
        import importlib.util

        conv = Path(__file__).resolve().parents[1] / "convert-whistle.py"
        spec = importlib.util.spec_from_file_location("convert_whistle", conv)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.c = Cact(cact_path)
        template = mod.build_template(self.c, 8)
        self.index = {name: i for i, (name, _k, _l) in enumerate(template)}
        self._cache: dict[str, np.ndarray] = {}

    def __call__(self, name: str) -> np.ndarray:
        if name not in self._cache:
            self._cache[name] = self.c.array(self.index[name]).astype(np.float64)
        return self._cache[name]


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def silu(x):
    return x * sigmoid(x)


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))  # overflow-free


def softmax(x, axis=-1):
    x = x - x.max(axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis, keepdims=True)


def zcrms(x, scale, eps=1e-6):
    return (1.0 + scale) * x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps)


def rms_unit(x, eps=1e-6):
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps)


def rope_tables(hd, T, theta=ROPE_THETA):
    f = 1.0 / theta ** (np.arange(0, hd, 2) / hd)
    a = np.outer(np.arange(T), f)
    return np.cos(a), np.sin(a)


def apply_rope(x, cos, sin):  # x [H, T, hd], half-split
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)


def sinkhorn(lk, iters=20):
    def lse(x, axis):
        m = x.max(axis, keepdims=True)
        return m + np.log(np.exp(x - m).sum(axis, keepdims=True))

    for _ in range(iters):
        lk = lk - lse(lk, -1)
        lk = lk - lse(lk, -2)
    return np.exp(lk)


# ---------------------------------------------------------------------------
# Frontend (probed): frames = n // 160 centered at 160*i, symmetric Hann(400)
# in a 512 rfft, |X|^2 @ fb, ln(mel + 1e-8 * max(mel)), per-bin mean/std.
# ---------------------------------------------------------------------------

def mel(pcm: np.ndarray, fb: np.ndarray) -> np.ndarray:
    x = np.asarray(pcm, np.float64)
    n_fft, hop, wl = 512, 160, 400
    nfr = len(x) // hop
    k = np.arange(wl)
    w = np.zeros(n_fft)
    w[:wl] = 0.5 - 0.5 * np.cos(2 * np.pi * k / (wl - 1))
    xp = np.pad(x, (200, 200 + n_fft))
    idx = np.arange(n_fft)[None] + hop * np.arange(nfr)[:, None]
    spec = np.abs(np.fft.rfft(xp[idx] * w, n=n_fft)) ** 2
    mp = spec @ fb
    m = np.log(mp + 1e-8 * mp.max() + 1e-30)
    sd = m.std(0)
    return (m - m.mean(0)) / np.where(sd > 0, sd, np.inf)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

def _conv_s2(x, k):
    """Depthwise 3x3, stride 2, pad 1. x [T, F, C], k [3, 3, C]."""
    xp = np.pad(x, ((1, 1), (1, 1), (0, 0)))
    To = (xp.shape[0] - 3) // 2 + 1
    Fo = (xp.shape[1] - 3) // 2 + 1
    y = np.zeros((To, Fo, x.shape[2]))
    for a in range(3):
        for b in range(3):
            y += xp[a:a + 2 * To:2, b:b + 2 * Fo:2] * k[a, b]
    return y


def stem(m: np.ndarray, W: Weights) -> np.ndarray:
    C = 128
    x = np.broadcast_to(m[:, :, None], m.shape + (C,))
    x = silu(_conv_s2(x, W("enc.stem.conv.w.weight").reshape(3, 3, C)))
    x = silu(_conv_s2(x, W("enc.stem.conv.dw_1.weight").reshape(3, 3, C)) @ W("enc.stem.pw_1.weight").T)
    x = silu(_conv_s2(x, W("enc.stem.conv.dw_2.weight").reshape(3, 3, C)) @ W("enc.stem.pw_2.weight").T)
    T, F, _ = x.shape
    return x.transpose(0, 2, 1).reshape(T, C * F) @ W("enc.stem.out.weight").T


def hmlp(x, W: Weights, p: str):
    g = lambda s: W(f"{p}.{s}")
    p1 = W("hada.perm1").astype(int)
    p2 = W("hada.perm2").astype(int)
    cond = 1 + softmax(x @ g("cond_v"), -1) @ g("cond_u")

    def kron(z, a, b):
        lead = z.shape[:-1]
        z = z.reshape(*lead, a.shape[0], b.shape[0])
        return np.einsum("...ij,ik,jl->...kl", z, a, b).reshape(*lead, -1)

    z = kron(g("d1") * x, g("w1a"), g("w1b"))[..., p1]
    z = kron(silu(g("d2") * cond * z + g("b2")), g("w2a"), g("w2b"))[..., p2]
    z = kron(g("d3") * z, g("w3a"), g("w3b"))
    return g("d4") * z


def attention(x, W: Weights, p: str, rope=None, kv_src=None, n_kv=N_KV, mask=None, probs=None):
    g = lambda s: W(f"{p}.{s}")
    kv_in = x if kv_src is None else kv_src
    T, S = x.shape[0], kv_in.shape[0]
    q = (x @ g("q.weight").T).reshape(T, N_HEADS, QK).transpose(1, 0, 2)
    k = (kv_in @ g("k.weight").T).reshape(S, n_kv, QK).transpose(1, 0, 2)
    v = (kv_in @ g("v.weight").T).reshape(S, n_kv, VD).transpose(1, 0, 2)
    q = zcrms(q, g("q_norm.weight"))
    k = zcrms(k, g("k_norm.weight"))
    if rope is not None:
        c, s = rope
        q = apply_rope(q, c[:T], s[:T])
        k = apply_rope(k, c[:S], s[:S])
    rep = N_HEADS // n_kv
    k = np.repeat(k, rep, 0)
    v = np.repeat(v, rep, 0)
    sc = q @ k.transpose(0, 2, 1) / np.sqrt(QK)
    if mask is not None:
        sc = np.where(mask, sc, -1e30)
    pr = softmax(sc, -1)
    if probs is not None:
        probs.append(pr)
    o = (pr @ v).transpose(1, 0, 2).reshape(T, N_HEADS * VD)
    o = o * sigmoid(x @ g("gate.weight").T)
    return o @ g("out.weight").T


def mhc_step(stream, block, W: Weights, pfx: str, layer: int):
    T, n, C = stream.shape
    nx = rms_unit(stream.reshape(T, n * C))
    a_pre, a_post, a_res = (W(f"{pfx}.{k}")[layer] for k in ("a_pre", "a_post", "a_res"))
    b_pre = W(f"{pfx}.b_pre").reshape(-1, n)[layer]
    b_post = W(f"{pfx}.b_post").reshape(-1, n)[layer]
    b_res = W(f"{pfx}.b_res").reshape(-1, n, n)[layer]
    phi_pre = W(f"{pfx}.phi_pre.weight")[layer * n:(layer + 1) * n].T
    phi_post = W(f"{pfx}.phi_post.weight")[layer * n:(layer + 1) * n].T
    phi_res = W(f"{pfx}.phi_res.weight")[layer * n * n:(layer + 1) * n * n].T
    lane = np.eye(n)[layer % n]
    hpre = sigmoid(a_pre * (nx @ phi_pre) + b_pre + 8 * lane - 4)
    u = np.einsum("tn,tnc->tc", hpre, stream)
    y = block(u) - u
    hpost = 2 * sigmoid(a_post * (nx @ phi_post) + b_post - 4 * (1 - lane))
    hres = sinkhorn(a_res * (nx @ phi_res).reshape(T, n, n) + b_res)
    return np.einsum("tij,tjc->tic", hres, stream) + hpost[..., None] * y[:, None, :]


def _dwconv_same(h, k):
    K = k.shape[0]
    p = (K - 1) // 2
    hp = np.pad(h, ((p, p), (0, 0)))
    return sum(hp[j:j + len(h)] * k[j] for j in range(K))


def encoder(x0: np.ndarray, W: Weights, blocks_out: list | None = None) -> np.ndarray:
    """Stem output [T, 512] -> final-normed encoder output [T, 512]."""
    T = len(x0)
    rope = rope_tables(QK, T)
    s = np.repeat(x0[:, None, :], LANES, 1)
    for i in range(8):
        p = f"enc.blocks.{i}"
        ag = sigmoid(W(f"{p}.attn_gate")[0])

        def block(u, p=p, ag=ag):
            u = u + 0.5 * hmlp(zcrms(u, W(f"{p}.norm_hmlp_0.weight")), W, f"{p}.hmlp_0")
            a = attention(zcrms(u, W(f"{p}.norm_in.weight")), W, f"{p}.attn", rope=rope)
            u = u + ag * zcrms(a, W(f"{p}.norm_post_attn.weight"))
            h = zcrms(u, W(f"{p}.norm_conv.weight")) @ W(f"{p}.conv_pw1.weight").T
            h = h[:, :D_MODEL] * sigmoid(h[:, D_MODEL:])
            h = silu(zcrms(_dwconv_same(h, W(f"{p}.conv.dw.weight")), W(f"{p}.norm_conv_out.weight")))
            u = u + h @ W(f"{p}.conv_pw2.weight").T
            u = u + 0.5 * hmlp(zcrms(u, W(f"{p}.norm_hmlp.weight")), W, f"{p}.hmlp")
            return u

        s = mhc_step(s, block, W, "enc.mhc", i)
        if blocks_out is not None:
            blocks_out.append(s.mean(1))
    return zcrms(s.mean(1), W("enc.final_norm.weight"))


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

def memory(enc: np.ndarray, W: Weights) -> np.ndarray:
    T = len(enc)
    i = np.arange(D_MODEL // 2)[None]
    ang = np.arange(T)[:, None] / 10000.0 ** (2 * i / D_MODEL)
    return enc + sigmoid(W("dec.pe_gate")[0]) * np.concatenate([np.sin(ang), np.cos(ang)], 1)


def _shift(x, j):
    if j == 0:
        return x
    if j >= len(x):
        return np.zeros_like(x)
    return np.concatenate([np.zeros((j,) + x.shape[1:]), x[:-j]], 0)


def engram_indices(tokens):
    u = np.asarray(tokens, np.uint64)
    idx = []
    for oi, order in enumerate(ENGRAM_ORDERS):
        for h in range(ENGRAM_HEADS):
            acc = np.full_like(u, (ENGRAM_SEED * (oi * ENGRAM_HEADS + h + 1)) & 0xFFFFFFFF)
            for j in range(order):
                sh = np.concatenate([np.zeros(min(j, len(u)), np.uint64), u[:max(len(u) - j, 0)]]) if j else u
                acc = ((acc ^ sh) * np.uint64(ENGRAM_PRIME)) & np.uint64(0xFFFFFFFF)
            acc = acc ^ (acc >> np.uint64(15))
            idx.append((acc % np.uint64(ENGRAM_SLOTS)).astype(np.int64))
    return np.stack(idx, -1)


def engram_kv(tokens, W: Weights, site: int):
    T = len(tokens)
    ind = engram_indices(tokens)
    tab = W(f"dec.engram.{site}.tables.weight")
    nt = ind.shape[1]
    fetched = np.stack([tab[t * ENGRAM_SLOTS + ind[:, t]] for t in range(nt)], 1)
    pos = np.arange(T)
    ok = np.stack([(pos >= o - 1) for o in ENGRAM_ORDERS for _ in range(ENGRAM_HEADS)], -1).astype(float)
    e = (fetched * ok[..., None]).reshape(T, -1)
    k = e @ W(f"dec.engram.{site}.key_proj.weight").T
    v = e @ W(f"dec.engram.{site}.value_proj.weight").T
    tp = W(f"dec.engram.{site}.conv_taps")
    dil = max(ENGRAM_ORDERS)
    return k, sum(tp[j] * _shift(v, j * dil) for j in range(ENGRAM_TAPS))


def _self_attn(x, W: Weights, p: str, rope):
    g = lambda s: W(f"{p}.{s}")
    T = x.shape[0]
    q, k, v = (sum(g(f"{n}_taps")[j] * _shift(x @ g(f"{n}.weight").T, j) for j in range(3)) for n in "qkv")
    q = zcrms(q.reshape(T, N_HEADS, QK).transpose(1, 0, 2), g("q_norm.weight"))
    k = zcrms(k.reshape(T, N_KV, QK).transpose(1, 0, 2), g("k_norm.weight"))
    v = v.reshape(T, N_KV, VD).transpose(1, 0, 2)
    c, s = rope
    q = apply_rope(q, c[:T], s[:T])
    k = apply_rope(k, c[:T], s[:T])
    rep = N_HEADS // N_KV
    k = np.repeat(k, rep, 0)
    v = np.repeat(v, rep, 0)
    sc = np.where(np.tril(np.ones((T, T), bool)), q @ k.transpose(0, 2, 1) / np.sqrt(QK), -1e30)
    o = (softmax(sc, -1) @ v).transpose(1, 0, 2).reshape(T, N_HEADS * VD)
    o = o * sigmoid(x @ g("gate.weight").T)
    return o @ g("out.weight").T


def decoder_logits(tokens, mem: np.ndarray, W: Weights) -> np.ndarray:
    """Teacher-forced logits [len(tokens), vocab]; mem = memory(enc)."""
    T = len(tokens)
    rope = rope_tables(QK, max(T, 8))
    emb = W("dec.token_embd.weight")
    x = emb[np.asarray(tokens)] * np.sqrt(D_MODEL)
    ekv = {s: engram_kv(tokens, W, s) for s in ENGRAM_SITES.values()}
    stream = np.repeat(x[:, None, :], LANES, 1)
    for i in range(8):
        p = f"dec.blocks.{i}"

        def block(u, i=i, p=p):
            if i in ENGRAM_SITES:
                ek, ev = ekv[ENGRAM_SITES[i]]
                alpha = sigmoid((rms_unit(u) * rms_unit(ek)).sum(-1) / np.sqrt(D_MODEL))
                u = u + alpha[:, None] * ev
            a = _self_attn(zcrms(u, W(f"{p}.norm_in.weight")), W, f"{p}.attn", rope)
            u = u + sigmoid(W(f"{p}.attn_gate")[0]) * zcrms(a, W(f"{p}.norm_post_attn.weight"))
            a = attention(zcrms(u, W(f"{p}.norm_cross.weight")), W, f"{p}.cross", kv_src=mem, n_kv=N_HEADS)
            u = u + sigmoid(W(f"{p}.cross_gate")[0]) * zcrms(a, W(f"{p}.norm_post_cross.weight"))
            return u + hmlp(zcrms(u, W(f"{p}.norm_hmlp.weight")), W, f"{p}.hmlp")

        stream = mhc_step(stream, block, W, "dec.mhc", i)
    return zcrms(stream.mean(1), W("dec.final_norm.weight")) @ emb.T


def greedy(mem: np.ndarray, W: Weights, prompt, max_new=200, eos=1):
    toks = list(prompt)
    for _ in range(max_new):
        t = int(np.argmax(decoder_logits(toks, mem, W)[-1]))
        if t == eos:
            break
        toks.append(t)
    return toks[len(prompt):]

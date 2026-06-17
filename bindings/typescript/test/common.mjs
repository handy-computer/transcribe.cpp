// Shared test plumbing. Two CI tiers (requirements §4): no-model tests run
// always; model-gated tests skip cleanly when their canary GGUF is absent
// (resolved from the TRANSCRIBE_SMOKE_* env vars fetch-canary exports).

import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { test } from "node:test";

const HERE = path.dirname(fileURLToPath(import.meta.url));

export const MODEL = process.env.TRANSCRIBE_SMOKE_MODEL || "";
export const STREAMING_MODEL = process.env.TRANSCRIBE_SMOKE_STREAMING_MODEL || "";
export const PARAKEET_STREAM_MODEL = process.env.TRANSCRIBE_SMOKE_PARAKEET_STREAM_MODEL || "";
export const PARAKEET_BUFFERED_MODEL = process.env.TRANSCRIBE_SMOKE_PARAKEET_BUFFERED_MODEL || "";
// Voxtral realtime is local-only (~2.5 GB+, too heavy for the CI canary set):
// its env var is NOT exported by fetch-canary, so this skips cleanly in CI.
export const VOXTRAL_MODEL = process.env.TRANSCRIBE_SMOKE_VOXTRAL_MODEL || "";

// jfk.wav ships in-repo; fetch-canary exports only the model paths.
export const AUDIO =
  process.env.TRANSCRIBE_SMOKE_AUDIO || path.resolve(HERE, "../../../samples/jfk.wav");

export const have = (p) => Boolean(p) && fs.existsSync(p);

/** A test that skips cleanly when its canary model is unavailable. */
export function modelTest(name, model, fn) {
  test(name, { skip: have(model) ? false : `canary model unavailable` }, fn);
}

/** Minimal 16-bit PCM WAV reader -> Float32Array (mono). */
export function readWav(file) {
  const b = fs.readFileSync(file);
  let o = 12,
    fmt = null,
    d = null;
  while (o + 8 <= b.length) {
    const id = b.toString("ascii", o, o + 4);
    const s = b.readUInt32LE(o + 4);
    if (id === "fmt ") fmt = { ch: b.readUInt16LE(o + 10) };
    else if (id === "data") d = b.subarray(o + 8, o + 8 + s);
    o += 8 + s + (s & 1);
  }
  const ch = fmt.ch;
  const n = Math.floor(d.length / 2 / ch);
  const p = new Float32Array(n);
  for (let i = 0; i < n; i++) p[i] = d.readInt16LE(i * 2 * ch) / 32768;
  return p;
}

export const jfk = () => readWav(AUDIO);

/** Feed `pcm` to a stream in ~1s chunks. */
export async function feedChunks(stream, pcm, chunk = 16000) {
  for (let i = 0; i < pcm.length; i += chunk) await stream.feed(pcm.subarray(i, i + chunk));
}

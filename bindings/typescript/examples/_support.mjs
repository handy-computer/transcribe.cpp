// Shared example plumbing (the analog of Rust's examples/common). Each example
// resolves its model/audio from argv or the TRANSCRIBE_SMOKE_* env vars, and
// skips cleanly (exit 0) when a required canary is absent — so the whole set is
// CI-executable headless under the same skip rules as the model test tier.

import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
export const DEFAULT_AUDIO = path.resolve(HERE, "../../../samples/jfk.wav");

export function model(envName) {
  return process.argv[2] || process.env[envName] || "";
}

export function audio() {
  return process.argv[3] || process.env.TRANSCRIBE_SMOKE_AUDIO || DEFAULT_AUDIO;
}

export function skip(msg) {
  console.log(`skip: ${msg}`);
  process.exit(0);
}

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
  if (!fmt || !d) throw new Error("bad WAV");
  const ch = fmt.ch;
  const n = Math.floor(d.length / 2 / ch);
  const p = new Float32Array(n);
  for (let i = 0; i < n; i++) p[i] = d.readInt16LE(i * 2 * ch) / 32768;
  return p;
}

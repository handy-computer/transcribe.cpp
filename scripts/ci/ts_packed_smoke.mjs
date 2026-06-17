// Verify the SHIPPED npm tarballs (the "test the shipped artifact" gate, §4):
// import the installed transcribe-cpp by absolute path — so node resolves it
// from the clean consumer's node_modules, not the repo tree — and transcribe
// the canary. No TRANSCRIBE_LIBRARY: the install must stand on the platform
// package alone.
//
// Env: TS_PKG_ENTRY (.../node_modules/transcribe-cpp/dist/index.js),
//      TRANSCRIBE_SMOKE_MODEL (canary GGUF), TRANSCRIBE_SMOKE_AUDIO (a WAV).

import * as fs from "node:fs";

const entry = process.env.TS_PKG_ENTRY;
const model = process.env.TRANSCRIBE_SMOKE_MODEL;
const audio = process.env.TRANSCRIBE_SMOKE_AUDIO;
if (!entry || !model || !audio) {
  console.log("skip: set TS_PKG_ENTRY + TRANSCRIBE_SMOKE_MODEL + TRANSCRIBE_SMOKE_AUDIO");
  process.exit(0);
}

function readWav(file) {
  const b = fs.readFileSync(file);
  let o = 12, fmt = null, d = null;
  while (o + 8 <= b.length) {
    const id = b.toString("ascii", o, o + 4), s = b.readUInt32LE(o + 4);
    if (id === "fmt ") fmt = { ch: b.readUInt16LE(o + 10) };
    else if (id === "data") d = b.subarray(o + 8, o + 8 + s);
    o += 8 + s + (s & 1);
  }
  const ch = fmt.ch, n = Math.floor(d.length / 2 / ch), p = new Float32Array(n);
  for (let i = 0; i < n; i++) p[i] = d.readInt16LE(i * 2 * ch) / 32768;
  return p;
}

const { TranscribeModel, libraryPath, version } = await import(entry);
console.log("version:", JSON.stringify(version()));
console.log("libraryPath:", libraryPath());
const m = await TranscribeModel.load(model);
const r = await m.transcribe(readWav(audio));
m.dispose();
const fromPackage = /node_modules\/@transcribe-cpp\//.test(libraryPath());
const ok = fromPackage && /ask not what your country/i.test(r.text);
console.log(`packed smoke: ${ok ? "PASS" : "FAIL"} (fromPackage=${fromPackage})`);
console.log(`  text: ${r.text.trim()}`);
process.exit(ok ? 0 : 1);

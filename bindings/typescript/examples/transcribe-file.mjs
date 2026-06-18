// transcribe-file — load a model, transcribe a WAV, print text + segments.
//
//   node examples/transcribe-file.mjs <model.gguf> <audio.wav>
//
// or set TRANSCRIBE_SMOKE_MODEL (+ optional TRANSCRIBE_SMOKE_AUDIO). Skips
// cleanly (exit 0) when no model is given.

import { TranscribeModel } from "../dist/index.js";
import { model, audio, readWav, skip } from "./_support.mjs";

const path = model("TRANSCRIBE_SMOKE_MODEL");
if (!path) skip("no model (pass a .gguf or set TRANSCRIBE_SMOKE_MODEL)");

const m = await TranscribeModel.load(path);
console.log(`model: ${m.arch}/${m.variant} on ${m.backend}`);

const result = await m.transcribe(readWav(audio()), { timestamps: "segment" });
console.log(`\ntext: ${result.text.trim()}`);
console.log(`language: ${result.language}`);
for (const s of result.segments) {
  console.log(`  [${s.t0Ms} - ${s.t1Ms} ms] ${s.text}`);
}
console.log(
  `\ntimings: load ${result.timings.loadMs.toFixed(0)}ms ` +
    `encode ${result.timings.encodeMs.toFixed(0)}ms decode ${result.timings.decodeMs.toFixed(0)}ms`,
);

m.dispose();

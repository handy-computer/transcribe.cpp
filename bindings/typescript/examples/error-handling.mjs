// error-handling — typed errors, cooperative cancellation, cleanup.
//
//   node examples/error-handling.mjs [model.gguf] [audio.wav]
//
// The typed-error demo needs no model; cancellation runs when a model is given.

import { TranscribeModel, ModelFileNotFound, Aborted } from "../dist/index.js";
import { model, audio, readWav, skip } from "./_support.mjs";

// 1) A missing file is a typed, catchable error — not a crash.
try {
  await TranscribeModel.load("/no/such/model.gguf");
} catch (e) {
  console.log(`missing file -> ${e.constructor.name} (status ${e.status}): ${e.message}`);
  console.log(`  instanceof ModelFileNotFound: ${e instanceof ModelFileNotFound}`);
}

const path = model("TRANSCRIBE_SMOKE_MODEL");
if (!path) skip("\nno model — typed-error demo only (set TRANSCRIBE_SMOKE_MODEL for cancellation)");

const m = await TranscribeModel.load(path);

// 2) Cooperative cancellation via AbortSignal; the partial transcript survives.
const controller = new AbortController();
controller.abort(); // abort up front for a deterministic demo
try {
  await m.transcribe(readWav(audio()), { signal: controller.signal });
} catch (e) {
  if (e instanceof Aborted) {
    console.log(`\ncancelled -> Aborted; partial text: "${(e.partialResult?.text ?? "").trim()}"`);
  } else {
    throw e;
  }
}

// 3) Deterministic cleanup.
m.dispose();
console.log("disposed cleanly");

// batch — transcribe multiple utterances in one call.
//
//   node examples/batch.mjs <model.gguf> <audio.wav>
//
// or set TRANSCRIBE_SMOKE_MODEL. Skips cleanly when absent.

import { TranscribeModel } from "../dist/index.js";
import { model, audio, readWav, skip } from "./_support.mjs";

const path = model("TRANSCRIBE_SMOKE_MODEL");
if (!path) skip("no model (set TRANSCRIBE_SMOKE_MODEL)");

const m = await TranscribeModel.load(path);
const session = m.createSession();

const pcm = readWav(audio());
const items = await session.runBatch([pcm, pcm.subarray(0, pcm.length / 2)]);

items.forEach((item, i) => {
  if (item.ok) console.log(`[${i}] ${item.result.text.trim()}`);
  else console.log(`[${i}] ERROR (${item.error.constructor.name}): ${item.error.message}`);
});

session.dispose();
m.dispose();

// streaming — feed chunks, show committed (stable) vs tentative (volatile) text.
//
//   node examples/streaming.mjs <streaming-model.gguf> <audio.wav>
//
// or set TRANSCRIBE_SMOKE_STREAMING_MODEL. Skips cleanly when absent.

import { TranscribeModel } from "../dist/index.js";
import { model, audio, readWav, skip } from "./_support.mjs";

const path = model("TRANSCRIBE_SMOKE_STREAMING_MODEL");
if (!path) skip("no streaming model (set TRANSCRIBE_SMOKE_STREAMING_MODEL)");

const m = await TranscribeModel.load(path);
const session = m.createSession();
const stream = await session.stream({ commitPolicy: "stable_prefix" });

const pcm = readWav(audio());
for (let i = 0; i < pcm.length; i += 16000) {
  await stream.feed(pcm.subarray(i, i + 16000));
  const t = stream.text;
  console.log(`rev ${stream.revision} | committed: "${t.committed.trim()}" | tentative: "${t.tentative.trim()}"`);
}
await stream.finalize();
console.log(`\nfinal: ${stream.text.committed.trim()}`);

stream.reset();
session.dispose();
m.dispose();

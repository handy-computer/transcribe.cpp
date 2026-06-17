import assert from "node:assert/strict";
import { modelTest, MODEL, jfk } from "./common.mjs";
import { TranscribeModel } from "../dist/index.js";

modelTest("batch returns one result per utterance", MODEL, async () => {
  const m = await TranscribeModel.load(MODEL);
  try {
    const s = m.createSession();
    const pcm = jfk();
    const items = await s.runBatch([pcm, pcm.subarray(0, pcm.length / 2)]);
    assert.equal(items.length, 2);
    assert.ok(items[0].ok && /ask not/i.test(items[0].result.text));
    assert.ok(items[1].ok);
    s.dispose();
  } finally {
    m.dispose();
  }
});

modelTest("empty batch is rejected", MODEL, async () => {
  const m = await TranscribeModel.load(MODEL);
  try {
    const s = m.createSession();
    await assert.rejects(() => s.runBatch([]));
    s.dispose();
  } finally {
    m.dispose();
  }
});

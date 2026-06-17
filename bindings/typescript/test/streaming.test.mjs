import assert from "node:assert/strict";
import { modelTest, MODEL, STREAMING_MODEL, jfk, feedChunks } from "./common.mjs";
import { TranscribeModel } from "../dist/index.js";

modelTest("streaming commits text and finalizes", STREAMING_MODEL, async () => {
  const m = await TranscribeModel.load(STREAMING_MODEL);
  try {
    assert.equal(m.capabilities.supportsStreaming, true);
    const s = m.createSession();
    const stream = await s.stream({ commitPolicy: "stable_prefix" });
    assert.equal(stream.state, "active");
    await feedChunks(stream, jfk());
    const fin = await stream.finalize();
    assert.equal(fin.isFinal, true);
    assert.ok(stream.revision > 0);
    const t = stream.text;
    assert.ok((t.committed + t.full).trim().length > 0, "expected non-empty streamed text");
    stream.reset();
    assert.equal(stream.state, "idle");
    s.dispose();
  } finally {
    m.dispose();
  }
});

modelTest("a non-streaming model rejects stream begin", MODEL, async () => {
  const m = await TranscribeModel.load(MODEL);
  try {
    const s = m.createSession();
    await assert.rejects(() => s.stream());
    s.dispose();
  } finally {
    m.dispose();
  }
});

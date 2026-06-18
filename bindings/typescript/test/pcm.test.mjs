import assert from "node:assert/strict";
import { modelTest, MODEL, jfk } from "./common.mjs";
import { TranscribeModel } from "../dist/index.js";

modelTest("empty PCM is rejected before any native call", MODEL, async () => {
  const m = await TranscribeModel.load(MODEL);
  try {
    const s = m.createSession();
    await assert.rejects(() => s.run(new Float32Array(0)), /empty/i);
    s.dispose();
  } finally {
    m.dispose();
  }
});

modelTest("number[] PCM is accepted (converted to float32)", MODEL, async () => {
  const m = await TranscribeModel.load(MODEL);
  try {
    const r = await m.transcribe(Array.from(jfk()));
    assert.match(r.text, /ask not/i);
  } finally {
    m.dispose();
  }
});

modelTest("a Buffer of float32 bytes is accepted", MODEL, async () => {
  const m = await TranscribeModel.load(MODEL);
  try {
    const f = jfk();
    const buf = Buffer.from(f.buffer, f.byteOffset, f.byteLength);
    const r = await m.transcribe(buf);
    assert.match(r.text, /ask not/i);
  } finally {
    m.dispose();
  }
});

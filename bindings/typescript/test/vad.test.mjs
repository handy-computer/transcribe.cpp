// VAD role on the toy silero_vad fixture: run, and stream parity with run.

import assert from "node:assert/strict";
import { modelTest, VAD_TOY_MODEL } from "./common.mjs";
import { TranscribeModel } from "../dist/index.js";

modelTest("toy: run, and stream equals run", VAD_TOY_MODEL, async () => {
  const m = await TranscribeModel.load(VAD_TOY_MODEL, { backend: "cpu" });
  const vad = m.createVadSession({ nThreads: 1 });
  try {
    assert.deepEqual(m.roles, ["vad"]);
    assert.deepEqual(m.vadInfo, { sampleRate: 16000, frameSamples: 512 });
    const pcm = new Float32Array(16000 + 100);
    let s = 7;
    for (let i = 0; i < pcm.length; i++) {
      s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
      pcm[i] = ((s >>> 8) & 0xffffff) / 16777216 - 0.5;
    }
    const r = await vad.run(pcm);
    assert.equal(r.probs.length, 32);

    const got = [];
    for (let off = 0; off < pcm.length; off += 777) {
      const f = await vad.streamFeed(pcm.subarray(off, off + 777));
      assert.equal(f.firstFrame, Math.floor(off / 512));
      got.push(...f.probs);
    }
    got.push(...(await vad.streamFlush()).probs);
    assert.deepEqual(got, r.probs);

    // Empty feed scores nothing and keeps the position; reset / flush rewind to 0.
    await vad.streamFeed(pcm.subarray(0, 1000));
    const empty = await vad.streamFeed(new Float32Array(0));
    assert.deepEqual([empty.probs.length, empty.firstFrame], [0, 1]);
    await vad.streamReset();
    assert.equal((await vad.streamFeed(pcm.subarray(0, 1000))).firstFrame, 0);
    await vad.streamFlush();
    assert.equal((await vad.streamFeed(pcm.subarray(0, 1000))).firstFrame, 0);
  } finally {
    vad.dispose();
    m.dispose();
  }
});

// VAD role: model.roles, vadInfo, VadSession wiring (run / stream / abort) on
// the toy silero_vad fixture, and the model-free VadIterator policy.

import assert from "node:assert/strict";
import { test } from "node:test";
import { modelTest, VAD_TOY_MODEL } from "./common.mjs";
import {
  TranscribeModel,
  VadIterator,
  InvalidArgument,
  UnsupportedRole,
  Aborted,
} from "../dist/index.js";

function noise(n, seed = 7) {
  const out = new Float32Array(n);
  let s = (seed | 1) >>> 0;
  for (let i = 0; i < n; i++) {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    out[i] = ((s >>> 8) & 0xffffff) / 16777216 - 0.5;
  }
  return out;
}

modelTest("toy: roles, info; other roles are refused", VAD_TOY_MODEL, async () => {
  const m = await TranscribeModel.load(VAD_TOY_MODEL, { backend: "cpu" });
  try {
    assert.deepEqual(m.roles, ["vad"]);
    assert.deepEqual(m.vadInfo, { sampleRate: 16000, frameSamples: 512 });
    assert.throws(() => m.createSession(), UnsupportedRole);
    assert.throws(() => m.createLangIdSession(), UnsupportedRole);
  } finally {
    m.dispose();
  }
});

modelTest("toy: run, stream equals run, flush, timings", VAD_TOY_MODEL, async () => {
  const m = await TranscribeModel.load(VAD_TOY_MODEL, { backend: "cpu" });
  const vad = m.createVadSession({ nThreads: 1 });
  try {
    const pcm = noise(16000 + 100);
    const r = await vad.run(pcm);
    assert.equal(r.probs.length, Math.ceil(pcm.length / 512));
    assert.equal(r.firstFrame, 0);
    assert.ok(r.probs.every((p) => Number.isFinite(p) && p >= 0 && p <= 1));
    for (const s of r.segments) assert.ok(0 <= s.startSample && s.startSample < s.endSample && s.endSample <= pcm.length);
    // threshold 0 marks every frame speech: one segment over the whole clip.
    const all = await vad.run(pcm, { threshold: 0, minSpeechMs: 0 });
    assert.deepEqual(all.segments, [{ startSample: 0, endSample: pcm.length }]);
    await assert.rejects(vad.run(pcm, { threshold: 2 }), InvalidArgument);

    // Streaming in odd chunks reproduces the offline probabilities.
    const got = [];
    for (let off = 0; off < pcm.length; off += 777) {
      const u = await vad.streamFeed(pcm.subarray(off, Math.min(off + 777, pcm.length)));
      assert.equal(u.firstFrame, got.length);
      assert.deepEqual(u.segments, []);
      got.push(...u.probs);
    }
    const tail = await vad.streamFlush();
    assert.equal(tail.firstFrame, got.length);
    got.push(...tail.probs);
    assert.deepEqual(Float32Array.from(got), r.probs);

    // Reset drops buffered audio: the next stream starts at frame 0.
    await vad.streamFeed(pcm.subarray(0, 1000));
    await vad.streamReset();
    assert.equal((await vad.streamFeed(pcm.subarray(0, 512))).firstFrame, 0);
    assert.ok(vad.timings.encodeMs > 0);
  } finally {
    vad.dispose();
    m.dispose();
  }
});

modelTest("toy: an aborted signal cancels the run", VAD_TOY_MODEL, async () => {
  const m = await TranscribeModel.load(VAD_TOY_MODEL, { backend: "cpu" });
  const vad = m.createVadSession();
  try {
    const ac = new AbortController();
    ac.abort();
    await assert.rejects(vad.run(noise(16000), { signal: ac.signal }), Aborted);
    assert.equal((await vad.run(noise(16000))).probs.length, 32);
  } finally {
    vad.dispose();
    m.dispose();
  }
});

test("VadIterator: START/END events and input checks", () => {
  const it = new VadIterator(512);
  try {
    // Speech in frames 2..4, then silence; END fires once it lasts minSilenceMs (100).
    const r = it.feed([0, 0, 1, 1, 1, 0, 0, 0, 0, 0]);
    assert.deepEqual(r.events, [
      { type: "start", sample: 3 * 512 - 480 - 512 },
      { type: "end", sample: 6 * 512 + 480 - 512 },
    ]);
    assert.equal(r.currentSample, 10 * 512);
    assert.equal(r.triggered, false);
    // Probabilities must be in [0, 1].
    assert.throws(() => it.feed([1.5]), InvalidArgument);
    // EOF does not emit END: active speech just stays triggered.
    const open = it.feed(new Float32Array([1]));
    assert.equal(open.triggered, true);
    assert.equal(open.events[0].type, "start");
    it.reset();
    assert.deepEqual(it.feed([]), { events: [], currentSample: 0, triggered: false });
  } finally {
    it.dispose();
  }
  assert.throws(() => new VadIterator(0), InvalidArgument);
});

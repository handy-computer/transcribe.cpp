// LANGID role: model.roles, langidInfo / labels and LangIdSession wiring.
// Contract checks run on the toy ecapa_tdnn fixture; top-1 on the real
// VoxLingua107 model.

import assert from "node:assert/strict";
import * as path from "node:path";
import { modelTest, LANGID_MODEL, LANGID_TOY_MODEL, SAMPLES, readWav } from "./common.mjs";
import {
  TranscribeModel,
  InvalidArgument,
  InputTooShort,
  UnsupportedRequest,
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

modelTest("toy: roles, info, labels; ASR calls are refused", LANGID_TOY_MODEL, async () => {
  const m = await TranscribeModel.load(LANGID_TOY_MODEL, { backend: "cpu" });
  try {
    assert.deepEqual(m.roles, ["langid"]);
    assert.deepEqual(m.langidInfo, { sampleRate: 16000, nLabels: 5, minAudioMs: 500, maxAudioMs: 30000 });
    assert.deepEqual(m.langidLabels[2], ["cc", "Charlie"]);
    assert.equal(m.langidLabelIndex("xx"), 0);
    assert.equal(m.langidLabelIndex("zz"), null);
    assert.throws(() => m.capabilities, UnsupportedRole);
    assert.throws(() => m.createSession(), UnsupportedRole);
  } finally {
    m.dispose();
  }
});

modelTest("toy: run contract (allowed, crop, minimum)", LANGID_TOY_MODEL, async () => {
  const m = await TranscribeModel.load(LANGID_TOY_MODEL, { backend: "cpu" });
  const lid = m.createLangIdSession({ nThreads: 1 });
  try {
    const pcm = noise(16000);
    const r = await lid.run(pcm);
    assert.equal(r.candidates.length, 5);
    assert.equal(r.allowedMass, 1);
    assert.equal(r.code, r.candidates[0].code);
    const sum = r.candidates.reduce((a, c) => a + c.p, 0);
    assert.ok(Math.abs(sum - 1) < 1e-5);

    const restricted = await lid.run(pcm, { allowed: ["bb", "dd", "bb"] });
    assert.equal(restricted.candidates.length, 2);
    assert.deepEqual(new Set(restricted.candidates.map((c) => c.code)), new Set(["bb", "dd"]));
    assert.ok(restricted.allowedMass < 1);

    assert.equal((await lid.run(pcm, { allowed: null })).candidates.length, 5);
    // An empty list must not silently mean "every label".
    await assert.rejects(lid.run(pcm, { allowed: [] }), InvalidArgument);
    await assert.rejects(lid.run(pcm, { allowed: ["zz"] }), UnsupportedRequest);
    await assert.rejects(lid.run(pcm.subarray(0, 6400)), InputTooShort);
    // Longer input is scored on its first maxAudioMs.
    const long = noise(16000 * 31);
    assert.deepEqual(
      (await lid.run(long)).candidates,
      (await lid.run(long.subarray(0, 16000 * 30))).candidates,
    );
    assert.ok(lid.timings.encodeMs > 0);
  } finally {
    lid.dispose();
    m.dispose();
  }
});

modelTest("toy: an aborted signal cancels the run", LANGID_TOY_MODEL, async () => {
  const m = await TranscribeModel.load(LANGID_TOY_MODEL, { backend: "cpu" });
  const lid = m.createLangIdSession();
  try {
    const ac = new AbortController();
    ac.abort();
    await assert.rejects(lid.run(noise(16000), { signal: ac.signal }), Aborted);
    assert.ok((await lid.run(noise(16000))).candidates.length > 0);
  } finally {
    lid.dispose();
    m.dispose();
  }
});

modelTest("real model: top-1 on the FLEURS clips", LANGID_MODEL, async () => {
  const m = await TranscribeModel.load(LANGID_MODEL, { backend: "cpu" });
  const lid = m.createLangIdSession();
  try {
    assert.equal(m.langidInfo.nLabels, 107);
    assert.equal(m.langidInfo.maxAudioMs, 30000);
    assert.equal(m.langidLabelIndex("he"), m.langidLabelIndex("iw"));
    for (const code of ["en", "de", "fr", "es", "ja", "zh", "ru", "id"]) {
      const r = await lid.run(readWav(path.join(SAMPLES, `fleurs-${code}.wav`)));
      assert.equal(r.candidates.length, 107);
      assert.equal(r.code, code);
      assert.ok(r.candidates[0].p >= 0.5);
    }
  } finally {
    lid.dispose();
    m.dispose();
  }
});

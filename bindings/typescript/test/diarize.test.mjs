// DIARIZE role: model.roles, diarizeInfo and DiarizeSession wiring. Segment
// values are pinned by the C test (sortformer_diarize_unit); here we check the
// binding returns well-formed rows and passes the preset through.

import assert from "node:assert/strict";
import { modelTest, MODEL, SORTFORMER_MODEL, SORTFORMER_AUDIO, readWav } from "./common.mjs";
import { TranscribeModel, InvalidArgument, TranscribeError, UnsupportedRole } from "../dist/index.js";

const mix = () => readWav(SORTFORMER_AUDIO);

modelTest("an ASR-only model refuses the diarize role with UnsupportedRole", MODEL, async () => {
  const m = await TranscribeModel.load(MODEL);
  try {
    assert.deepEqual(m.roles, ["asr"]);
    assert.throws(() => m.diarizeInfo, UnsupportedRole);
    assert.throws(() => m.createDiarizeSession(), UnsupportedRole);
    assert.equal(m.accepts({ kind: "sortformer_diarize" }), false);
  } finally {
    m.dispose();
  }
});

modelTest("sortformer serves only diarize; diarizeInfo; ASR calls are refused", SORTFORMER_MODEL, async () => {
  const m = await TranscribeModel.load(SORTFORMER_MODEL);
  try {
    assert.deepEqual(m.roles, ["diarize"]);
    assert.deepEqual(m.diarizeInfo, { sampleRate: 16000, maxSpeakers: 4 });
    assert.equal(m.accepts({ kind: "sortformer_diarize" }), true);
    assert.throws(() => m.capabilities, UnsupportedRole);
    assert.throws(() => m.createSession(), UnsupportedRole);
  } finally {
    m.dispose();
  }
});

modelTest("diarize run returns speaker turns and honors the preset", SORTFORMER_MODEL, async () => {
  const m = await TranscribeModel.load(SORTFORMER_MODEL, { backend: "cpu" });
  try {
    const pcm = mix();
    const d = m.createDiarizeSession({ nThreads: 4 });
    const turns = async (preset) =>
      (await d.run(pcm, { family: { kind: "sortformer_diarize", preset } })).map((r) => [
        r.t0Ms,
        r.t1Ms,
        r.speakerId,
      ]);
    const def = await turns("default");
    assert.ok(def.length > 0);
    for (const [t0, t1, spk] of def) {
      assert.ok(t0 < t1, `${t0} < ${t1}`);
      assert.ok(spk >= 1 && spk <= m.diarizeInfo.maxSpeakers, `speaker ${spk}`);
    }
    assert.notDeepEqual(await turns("low_latency"), def, "preset reaches the native run");
    assert.ok(d.timings.encodeMs > 0);
    d.dispose();
  } finally {
    m.dispose();
  }
});

modelTest("a bad preset or wrong-slot extension is rejected", SORTFORMER_MODEL, async () => {
  const m = await TranscribeModel.load(SORTFORMER_MODEL);
  try {
    const d = m.createDiarizeSession();
    const pcm = mix();
    await assert.rejects(
      () => d.run(pcm, { family: { kind: "sortformer_diarize", preset: "ultra_low_latency" } }),
      (e) => e instanceof TranscribeError && /invalid sortformer preset/.test(e.message),
    );
    await assert.rejects(() => d.run(pcm, { family: { kind: "whisper" } }), InvalidArgument);
    await assert.rejects(
      () => d.run(pcm, { family: { kind: "sortformer" } }),
      (e) => e instanceof InvalidArgument && /unknown family extension kind/.test(e.message),
    );
    assert.ok((await d.run(pcm)).length > 0);
    d.dispose();
  } finally {
    m.dispose();
  }
});

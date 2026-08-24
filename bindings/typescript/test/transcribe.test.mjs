import assert from "node:assert/strict";
import { modelTest, ITN_MODEL, MODEL, PNC_MODEL, jfk } from "./common.mjs";
import { TranscribeModel } from "../dist/index.js";

modelTest("offline transcription returns text + detected language", MODEL, async () => {
  const m = await TranscribeModel.load(MODEL);
  try {
    const r = await m.transcribe(jfk());
    assert.match(r.text, /ask not what your country/i);
    assert.equal(r.language, "en");
    assert.equal(r.aborted, false);
    assert.equal(r.truncated, false);
  } finally {
    m.dispose();
  }
});

modelTest("segment timestamps populate", MODEL, async () => {
  const m = await TranscribeModel.load(MODEL);
  try {
    const r = await m.transcribe(jfk(), { timestamps: "segment" });
    assert.equal(r.timestampKind, "segment");
    assert.ok(r.segments.length >= 1);
    const s = r.segments[0];
    assert.ok(s.t1Ms > s.t0Ms, "segment should span time");
    assert.ok(s.text.length > 0);
  } finally {
    m.dispose();
  }
});

modelTest("capabilities + identity", MODEL, async () => {
  const m = await TranscribeModel.load(MODEL);
  try {
    const c = m.capabilities;
    assert.equal(c.nativeSampleRate, 16000);
    assert.ok(c.languages.length > 0);
    assert.equal(typeof m.arch, "string");
    assert.ok(m.arch.length > 0);
    assert.ok(m.backend.length > 0);
  } finally {
    m.dispose();
  }
});

modelTest("tokenize returns a non-empty token array", MODEL, async () => {
  const m = await TranscribeModel.load(MODEL);
  try {
    const toks = m.tokenize("ask not what your country can do for you");
    assert.ok(toks instanceof Int32Array);
    assert.ok(toks.length > 0);
  } finally {
    m.dispose();
  }
});

modelTest("PNC changes the Canary prompt", PNC_MODEL, async () => {
  const m = await TranscribeModel.load(PNC_MODEL, { backend: "cpu" });
  try {
    assert.equal(m.supports("pnc"), true);
    const s = m.createSession();
    try {
      await assert.rejects(
        // @ts-expect-error exercised from JavaScript to prove runtime validation
        () => s.run(jfk(), { pnc: "maybe" }),
        /invalid pnc/,
      );
      const base = await s.run(jfk(), { language: "en", pnc: "default" });
      const on = await s.run(jfk(), { language: "en", pnc: "on" });
      const off = await s.run(jfk(), { language: "en", pnc: "off" });
      assert.equal(base.text, on.text);
      assert.notEqual(off.text, on.text);
      assert.equal(off.text, off.text.toLowerCase());
    } finally {
      s.dispose();
    }
  } finally {
    m.dispose();
  }
});

modelTest("ITN changes SenseVoice text normalization", ITN_MODEL, async () => {
  const m = await TranscribeModel.load(ITN_MODEL, { backend: "cpu" });
  try {
    assert.equal(m.supports("itn"), true);
    const s = m.createSession();
    try {
      await assert.rejects(
        // @ts-expect-error exercised from JavaScript to prove runtime validation
        () => s.run(jfk(), { itn: "maybe" }),
        /invalid itn/,
      );
      const base = await s.run(jfk(), { language: "en", itn: "default" });
      const off = await s.run(jfk(), { language: "en", itn: "off" });
      const on = await s.run(jfk(), { language: "en", itn: "on" });
      assert.deepEqual([base.text, base.rawText], [off.text, off.rawText]);
      assert.notEqual(on.text, off.text);
      assert.match(off.rawText, /<\|woitn\|>/);
      assert.match(on.rawText, /<\|withitn\|>/);
    } finally {
      s.dispose();
    }
  } finally {
    m.dispose();
  }
});

modelTest("one model serves many sessions", MODEL, async () => {
  const m = await TranscribeModel.load(MODEL);
  try {
    const a = m.createSession();
    const b = m.createSession();
    const [ra, rb] = await Promise.all([a.run(jfk()), b.run(jfk())]);
    assert.match(ra.text, /ask not/i);
    assert.match(rb.text, /ask not/i);
    a.dispose();
    b.dispose();
  } finally {
    m.dispose();
  }
});

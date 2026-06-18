import assert from "node:assert/strict";
import { modelTest, MODEL, jfk } from "./common.mjs";
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

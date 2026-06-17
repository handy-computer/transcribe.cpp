import assert from "node:assert/strict";
import {
  modelTest,
  MODEL,
  PARAKEET_STREAM_MODEL,
  PARAKEET_BUFFERED_MODEL,
  VOXTRAL_MODEL,
  jfk,
  feedChunks,
} from "./common.mjs";
import { TranscribeModel } from "../dist/index.js";

modelTest("whisper run extension applies; accepts() discriminates", MODEL, async () => {
  const m = await TranscribeModel.load(MODEL);
  try {
    assert.equal(m.accepts({ kind: "whisper" }), true);
    assert.equal(m.accepts({ kind: "moonshine" }), false);
    const r = await m.transcribe(jfk(), {
      family: { kind: "whisper", initialPrompt: "A JFK speech." },
    });
    assert.match(r.text, /ask not/i);
  } finally {
    m.dispose();
  }
});

modelTest("run() rejects a stream-slot family with InvalidArgument", MODEL, async () => {
  const m = await TranscribeModel.load(MODEL);
  try {
    await assert.rejects(
      () => m.transcribe(jfk(), { family: { kind: "moonshine" } }),
      (e) => e.constructor.name === "InvalidArgument",
    );
  } finally {
    m.dispose();
  }
});

modelTest("parakeet cache-aware stream extension", PARAKEET_STREAM_MODEL, async () => {
  const m = await TranscribeModel.load(PARAKEET_STREAM_MODEL);
  try {
    assert.equal(m.accepts({ kind: "parakeet" }), true);
    const s = m.createSession();
    const stream = await s.stream({ family: { kind: "parakeet", attContextRight: 13 } });
    await feedChunks(stream, jfk());
    await stream.finalize();
    assert.ok((stream.text.committed + stream.text.full).trim().length > 0);
    stream.reset();
    s.dispose();
  } finally {
    m.dispose();
  }
});

modelTest("parakeet buffered stream extension", PARAKEET_BUFFERED_MODEL, async () => {
  const m = await TranscribeModel.load(PARAKEET_BUFFERED_MODEL);
  try {
    assert.equal(m.accepts({ kind: "parakeet_buffered" }), true);
    const s = m.createSession();
    const stream = await s.stream({ family: { kind: "parakeet_buffered", chunkMs: 2000 } });
    await feedChunks(stream, jfk());
    await stream.finalize();
    assert.ok((stream.text.committed + stream.text.full).trim().length > 0);
    stream.reset();
    s.dispose();
  } finally {
    m.dispose();
  }
});

// Local-only (CI never sets TRANSCRIBE_SMOKE_VOXTRAL_MODEL — the GGUF is too
// heavy for the canary set), matching the other bindings' voxtral footnote.
modelTest("voxtral realtime stream extension", VOXTRAL_MODEL, async () => {
  const m = await TranscribeModel.load(VOXTRAL_MODEL);
  try {
    assert.equal(m.accepts({ kind: "voxtral" }), true);
    const s = m.createSession();
    const stream = await s.stream({
      family: { kind: "voxtral", numDelayTokens: 4, minDecodeIntervalMs: 0 },
    });
    await feedChunks(stream, jfk());
    await stream.finalize();
    assert.ok((stream.text.committed + stream.text.full).trim().length > 0);
    stream.reset();
    s.dispose();
  } finally {
    m.dispose();
  }
});

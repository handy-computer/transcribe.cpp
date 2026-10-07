// Generic prompting (vocabulary / prompt / prefix) against the whisper and
// streaming canaries. Mirrors bindings/python/tests/test_prompting.py.

import assert from "node:assert/strict";
import koffi from "koffi";
import { modelTest, MODEL, STREAMING_MODEL, jfk, feedChunks } from "./common.mjs";
import { TranscribeModel, InvalidArgument, UnsupportedRequest } from "../dist/index.js";

const TERMS = ["Kennedy", "Americans", "GGUF"];

async function withSession(path, fn) {
  const m = await TranscribeModel.load(path);
  try {
    const s = m.createSession();
    try {
      await fn(m, s);
    } finally {
      s.dispose();
    }
  } finally {
    m.dispose();
  }
}

modelTest("vocabulary and prompt reach run and runBatch", MODEL, async () => {
  await withSession(MODEL, async (m, s) => {
    for (const f of ["vocabulary", "context_prompt", "instruct", "transcript_prefix"]) {
      assert.equal(typeof m.supports(f), "boolean");
    }
    const r = await s.run(jfk(), { vocabulary: TERMS, prompt: "A speech." });
    assert.match(r.text, /ask not/i);
    const items = await s.runBatch([jfk(), jfk()], { vocabulary: TERMS });
    assert.ok(items.every((i) => i.ok));
  });
});

modelTest("a rejected family option frees the vocabulary it followed", MODEL, async () => {
  await withSession(MODEL, async (_m, s) => {
    const { alloc, free } = koffi;
    let live = 0;
    koffi.alloc = (...a) => (live++, alloc(...a));
    koffi.free = (...a) => (live--, free(...a));
    try {
      await assert.rejects(s.run(jfk(), { vocabulary: TERMS, family: { kind: "nope" } }));
      await assert.rejects(s.stream({ vocabulary: TERMS, commitPolicy: "nope" }));
    } finally {
      koffi.alloc = alloc;
      koffi.free = free;
    }
    assert.equal(live, 0, "koffi allocations leaked");
  });
});

modelTest("prefix: text is the continuation, rawText leads with it", MODEL, async () => {
  await withSession(MODEL, async (m, s) => {
    if (!m.supports("transcript_prefix")) return;
    const r = await s.run(jfk(), { prefix: "And so my fellow Americans", timestamps: "none" });
    assert.match(r.rawText, /^\s*And so my fellow Americans/);
    assert.doesNotMatch(r.text, /fellow Americans/);
    assert.match(r.text, /ask not/i);
  });
});

modelTest("prompting input errors", MODEL, async () => {
  await withSession(MODEL, async (_m, s) => {
    await assert.rejects(() => s.run(jfk(), { vocabulary: "Kennedy" }), InvalidArgument);
    await assert.rejects(() => s.run(jfk(), { prompt: "a\0b" }), InvalidArgument);
    await assert.rejects(() => s.run(jfk(), { prompt: "hi <|endoftext|>" }), InvalidArgument);
    await assert.rejects(() => s.runBatch([jfk()], { prefix: "And so" }), InvalidArgument);
  });
});

modelTest("stream accepts vocabulary, rejects instruct", STREAMING_MODEL, async () => {
  await withSession(STREAMING_MODEL, async (_m, s) => {
    await assert.rejects(() => s.stream({ task: "instruct", prompt: "Summarize." }), UnsupportedRequest);
    const stream = await s.stream({ vocabulary: TERMS, prompt: "A speech." });
    await feedChunks(stream, jfk());
    const fin = await stream.finalize();
    assert.equal(fin.isFinal, true);
    stream.reset();
  });
});

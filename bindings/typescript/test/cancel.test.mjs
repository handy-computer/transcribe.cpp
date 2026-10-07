import assert from "node:assert/strict";
import { getEventListeners } from "node:events";
import { modelTest, MODEL, jfk } from "./common.mjs";
import { TranscribeModel, Aborted } from "../dist/index.js";

modelTest("an uncancelled run is not aborted", MODEL, async () => {
  const m = await TranscribeModel.load(MODEL);
  try {
    const r = await m.transcribe(jfk());
    assert.equal(r.aborted, false);
  } finally {
    m.dispose();
  }
});

modelTest("a pre-aborted signal raises Aborted with a partial result", MODEL, async () => {
  const m = await TranscribeModel.load(MODEL);
  try {
    const s = m.createSession();
    const ac = new AbortController();
    ac.abort();
    await assert.rejects(
      () => s.run(jfk(), { signal: ac.signal }),
      (e) => e instanceof Aborted && e.partialResult !== undefined,
    );
    s.dispose();
  } finally {
    m.dispose();
  }
});

modelTest("dispose during a run with a signal returns the result and removes the listener", MODEL, async () => {
  const m = await TranscribeModel.load(MODEL);
  try {
    const s = m.createSession();
    const ac = new AbortController();
    const p = s.run(jfk(), { signal: ac.signal });
    await Promise.resolve();
    s.dispose();
    assert.match((await p).text, /ask not what your country/i);
    assert.equal(getEventListeners(ac.signal, "abort").length, 0);
  } finally {
    m.dispose();
  }
});

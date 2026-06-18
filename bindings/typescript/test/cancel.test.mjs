import assert from "node:assert/strict";
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

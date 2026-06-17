import assert from "node:assert/strict";
import { modelTest, MODEL } from "./common.mjs";
import { TranscribeModel } from "../dist/index.js";

modelTest("double dispose is safe; disposing a model closes its sessions", MODEL, async () => {
  const m = await TranscribeModel.load(MODEL);
  const s = m.createSession();
  m.dispose();
  m.dispose(); // idempotent
  assert.throws(() => s.limits, /disposed/); // closed by the model
});

modelTest("dispose ordering is safe either way", MODEL, async () => {
  const m = await TranscribeModel.load(MODEL);
  const s = m.createSession();
  s.dispose();
  s.dispose(); // idempotent
  m.dispose(); // model after session
});

modelTest("Symbol.dispose releases the model", MODEL, async () => {
  const m = await TranscribeModel.load(MODEL);
  m.createSession();
  m[Symbol.dispose](); // what a `using` declaration calls at block exit
  assert.throws(() => m.capabilities, /disposed/);
});

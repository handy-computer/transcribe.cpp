// backend-select — discover devices, request an explicit backend, fall back.
//
//   node examples/backend-select.mjs [model.gguf]
//
// Device discovery needs no model; the explicit-backend load is skipped when no
// model is given.

import { getAvailableBackends, backendAvailable, TranscribeModel } from "../dist/index.js";
import { model, skip } from "./_support.mjs";

console.log("discovered devices:");
for (const d of getAvailableBackends()) {
  console.log(`  ${d.kind.padEnd(7)} ${d.name} — ${d.description}`);
}
console.log("\nbackend availability:");
for (const b of ["cpu", "metal", "vulkan", "cuda"]) {
  console.log(`  ${b.padEnd(7)} ${backendAvailable(b)}`);
}

const path = model("TRANSCRIBE_SMOKE_MODEL");
if (!path) skip("\nno model — device discovery only (set TRANSCRIBE_SMOKE_MODEL to load)");

// Prefer an accelerator, fall back to CPU on a clean failure.
const preferred = backendAvailable("metal") ? "metal" : backendAvailable("cuda") ? "cuda" : "cpu";
console.log(`\nrequesting backend: ${preferred}`);
let m;
try {
  m = await TranscribeModel.load(path, { backend: preferred });
} catch (e) {
  console.log(`  ${preferred} unavailable (${e.constructor.name}); retrying on cpu`);
  m = await TranscribeModel.load(path, { backend: "cpu" });
}
console.log(`loaded on backend: ${m.backend}`);
m.dispose();

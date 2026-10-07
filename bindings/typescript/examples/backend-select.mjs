// backend-select — discover devices, select exactly, show policy failure.
//
//   node examples/backend-select.mjs [model.gguf]
//
// Device discovery needs no model; model loads are skipped when none is given.

import { getAvailableBackends, backendAvailable, TranscribeModel } from "../dist/index.js";
import { model, skip } from "./_support.mjs";

const devices = getAvailableBackends();
console.log("discovered devices:");
for (const d of devices) {
  console.log(`  ${d.kind.padEnd(7)} ${d.name} — ${d.description}`);
}
console.log("\nbackend availability:");
for (const b of ["cpu", "metal", "vulkan", "cuda", "rocm"]) {
  console.log(`  ${b.padEnd(7)} ${backendAvailable(b)}`);
}

const path = model("TRANSCRIBE_SMOKE_MODEL");
if (!path) skip("\nno model — device discovery only (set TRANSCRIBE_SMOKE_MODEL to load)");

// Exact selection passes an object returned by getAvailableBackends(), not its
// display index. CPU provides a deterministic example on every platform.
const cpu = devices.find((device) => device.deviceType === "cpu");
if (!cpu) skip("no selectable CPU device");
const exact = await TranscribeModel.load(path, { device: cpu });
console.log(`\nselected exact device ${exact.device.name}, bound to: ${exact.backend}`);
exact.dispose();

// An explicit backend policy that cannot be satisfied fails cleanly.
const unavailable = ["cuda", "rocm", "vulkan", "metal"].find(
  (backend) => !backendAvailable(backend),
);
if (unavailable) {
  try {
    const unexpected = await TranscribeModel.load(path, { backend: unavailable });
    unexpected.dispose();
    console.log(`requesting ${unavailable} unexpectedly succeeded`);
  } catch (error) {
    console.log(`requesting ${unavailable} failed cleanly: ${error.message}`);
  }
} else {
  console.log("every optional backend is available on this build");
}

// Model-gated tier: device selection on load. Skips cleanly when the canary
// GGUF is absent (modelTest), otherwise loads it and exercises model.device
// plus the gpuDevice/backend validation surface.
import assert from "node:assert/strict";
import { modelTest, MODEL } from "./common.mjs";
import { TranscribeModel, getAvailableBackends, InvalidArgument } from "../dist/index.js";

modelTest("model.device reports an index-less device that matches a registry entry", MODEL, async () => {
  const m = await TranscribeModel.load(MODEL);
  try {
    const dev = m.device;
    assert.equal(typeof dev, "object");
    assert.notEqual(dev, null);
    // model.device comes from transcribe_model_get_device, which does not expose
    // a registry index — the binding reports it as null (see types.ts).
    assert.equal(dev.index, null);

    // It must correspond to a device the registry enumerates: match by name, and
    // by deviceId too when the backend reports a stable hardware id.
    const backends = getAvailableBackends();
    const match = backends.find(
      (b) =>
        b.name === dev.name &&
        (dev.deviceId === null || b.deviceId === dev.deviceId),
    );
    assert.ok(
      match,
      `model.device (${JSON.stringify({ name: dev.name, deviceId: dev.deviceId })}) ` +
        `should match a getAvailableBackends() entry`,
    );
  } finally {
    m.dispose();
  }
});

modelTest("negative gpuDevice is rejected with InvalidArgument", MODEL, async () => {
  await assert.rejects(
    () => TranscribeModel.load(MODEL, { gpuDevice: -1 }),
    (e) => e instanceof InvalidArgument,
  );
});

modelTest("out-of-range gpuDevice is rejected with InvalidArgument", MODEL, async () => {
  const outOfRange = getAvailableBackends().length + 1000;
  await assert.rejects(
    () => TranscribeModel.load(MODEL, { gpuDevice: outOfRange }),
    (e) => e instanceof InvalidArgument,
  );
});

modelTest("selecting a GPU index under the cpu backend is rejected", MODEL, async () => {
  // Hardware-independent: the cpu backend has no GPU device 1 to select, so the
  // pairing must be rejected regardless of what GPUs the host actually has.
  await assert.rejects(
    () => TranscribeModel.load(MODEL, { backend: "cpu", gpuDevice: 1 }),
    (e) => e instanceof InvalidArgument,
  );
});

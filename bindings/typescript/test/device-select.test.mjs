// Exact opaque-device selection tests.

import assert from "node:assert/strict";
import test from "node:test";
import {
  backendAvailable,
  getAvailableBackends,
  InvalidArgument,
  TranscribeModel,
} from "../dist/index.js";
import { MODEL, modelTest } from "./common.mjs";

modelTest("an enumerated device can be selected exactly", MODEL, async () => {
  const device = getAvailableBackends().find((d) => d.deviceType !== "accel");
  if (!device) return;
  try {
    const model = await TranscribeModel.load(MODEL, { device });
    try {
      assert.equal(model.device.name, device.name);
      assert.equal(model.device.deviceId, device.deviceId);
    } finally {
      model.dispose();
    }
  } catch (error) {
    // A registered device can still fail driver initialization. The important
    // contract is that native selection fails rather than moving elsewhere.
    assert.match(String(error), /backend/i);
  }
});

modelTest("explicit backend must match exact device", MODEL, async () => {
  const gpu = getAvailableBackends().find(
    (d) => d.deviceType === "gpu" || d.deviceType === "igpu",
  );
  if (!gpu) return;
  await assert.rejects(
    () => TranscribeModel.load(MODEL, { backend: "cpu", device: gpu }),
    InvalidArgument,
  );
});

test("automatic backend probing remains available", () => {
  assert.equal(typeof backendAvailable("auto"), "boolean");
});

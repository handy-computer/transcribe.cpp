// No-model tier: always runs (needs the native library loadable, no GGUF).
import { test } from "node:test";
import assert from "node:assert/strict";
import * as os from "node:os";
import * as fs from "node:fs";
import * as path from "node:path";
import {
  version,
  getAvailableBackends,
  backendAvailable,
  setLogHandler,
  TranscribeModel,
  Stream,
  ModelFileNotFound,
  ModelLoadError,
} from "../dist/index.js";

test("Stream does not expose internal lease/teardown controls", () => {
  // releaseLease()/deactivate() must NOT be reachable from user code — calling
  // the lease release on a live stream would reopen the cross-session race.
  // They live in a module-private map, not on the class.
  for (const name of ["releaseLease", "deactivate", "invalidate"]) {
    assert.equal(name in Stream.prototype, false, `${name} must not be on Stream.prototype`);
  }
});

test("setLogHandler works before and after native bootstrap", () => {
  const seen = [];
  setLogHandler((level, message) => seen.push([level, message]));
  const v = version();
  assert.match(v.version, /^\d+\.\d+\.\d+/);

  setLogHandler(null);
  setLogHandler(() => {});
  setLogHandler(null);
});

test("version exposes base version, commit, and header hash", () => {
  const v = version();
  assert.match(v.version, /^\d+\.\d+\.\d+/);
  assert.match(v.headerHash, /^[0-9a-f]{16}$/);
  assert.equal(typeof v.commit, "string");
});

test("at least one CPU backend is registered", () => {
  const devices = getAvailableBackends();
  assert.ok(devices.length >= 1, "expected >= 1 device");
  assert.ok(
    devices.some((d) => d.kind === "cpu"),
    "expected a cpu device",
  );
});

test("getAvailableBackends returns a non-empty array", () => {
  const devices = getAvailableBackends();
  assert.ok(Array.isArray(devices), "expected an array");
  assert.ok(devices.length >= 1, "expected >= 1 device");
});

test("each backend device has a well-formed, index-aligned shape", () => {
  const devices = getAvailableBackends();
  devices.forEach((dev, i) => {
    assert.equal(dev.index, i, `device ${i} index should equal its position`);
    assert.ok(
      ["cpu", "gpu", "igpu", "accel", "unknown"].includes(dev.deviceType),
      `device ${i} deviceType ${JSON.stringify(dev.deviceType)} is not a known class`,
    );
    assert.ok(dev.memoryTotal >= 0, `device ${i} memoryTotal must be >= 0`);
    assert.ok(dev.memoryFree >= 0, `device ${i} memoryFree must be >= 0`);
    assert.ok(
      dev.deviceId === null || typeof dev.deviceId === "string",
      `device ${i} deviceId must be null or a string`,
    );
    assert.equal(typeof dev.name, "string");
    assert.ok(dev.name.length > 0, `device ${i} name must be non-empty`);
    assert.equal(typeof dev.kind, "string");
    assert.ok(dev.kind.length > 0, `device ${i} kind must be non-empty`);
  });
});

test("backendAvailable is a clean boolean probe", () => {
  assert.equal(backendAvailable("cpu"), true);
  assert.equal(typeof backendAvailable("cuda"), "boolean");
});

test("invalid backend string is rejected, not silently coerced", () => {
  // @ts-expect-error intentionally bogus
  assert.throws(() => backendAvailable("nope"));
});

test("missing model file maps to ModelFileNotFound", async () => {
  await assert.rejects(
    () => TranscribeModel.load("/no/such/model.gguf"),
    (e) => e instanceof ModelFileNotFound,
  );
});

test("junk file maps to ModelLoadError", async () => {
  const f = path.join(os.tmpdir(), `transcribe-junk-${process.pid}.gguf`);
  fs.writeFileSync(f, Buffer.from("definitely not a gguf"));
  try {
    await assert.rejects(
      () => TranscribeModel.load(f),
      (e) => e instanceof ModelLoadError,
    );
  } finally {
    fs.rmSync(f, { force: true });
  }
});

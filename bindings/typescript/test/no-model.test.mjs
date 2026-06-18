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

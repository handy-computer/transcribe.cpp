// Two-phase native bootstrap: library load vs. backend initialization.
//
// Backend-init state is process-global and one-shot, so every scenario runs in
// a fresh child process. The timeout doubles as the deadlock check: a sync call
// that waited on the init worker instead of failing fast would hang the child.

import { test } from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const ENTRY = pathToFileURL(
  path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../dist/index.js"),
).href;

/** Run `body` (an ES module body with `t` = the package) in a fresh Node. */
function scenario(body, env = {}) {
  const src = `import * as t from ${JSON.stringify(ENTRY)};\n${body}`;
  const r = spawnSync(process.execPath, ["--input-type=module", "-e", src], {
    encoding: "utf8",
    timeout: 120_000,
    env: { ...process.env, ...env },
  });
  assert.equal(r.error, undefined, `child failed to finish (hang?): ${r.error}`);
  assert.equal(r.status, 0, `child exited ${r.status}\nstdout:\n${r.stdout}\nstderr:\n${r.stderr}`);
  return r.stdout;
}

test("version(), libraryPath(), and argument validation do not initialize backends", () => {
  scenario(`
    const v = t.version();
    if (!/^\\d+\\.\\d+\\.\\d+/.test(v.version)) throw new Error("bad version " + v.version);
    t.libraryPath();
    await t.backendAvailableAsync("nope").then(
      () => { throw new Error("expected rejection"); },
      (e) => { if (!(e instanceof t.TranscribeError)) throw e; },
    );
    if (t.backendState() !== "uninitialized") throw new Error("state " + t.backendState());
  `);
});

test("initialize() is shared, and sync backend queries fail fast while it runs", () => {
  scenario(`
    const a = t.initialize();
    const b = t.initialize();
    if (a !== b) throw new Error("concurrent initialize() calls must share one promise");
    if (t.backendState() !== "initializing") throw new Error("state " + t.backendState());
    for (const call of [() => t.getAvailableBackends(), () => t.backendAvailable("cpu")]) {
      try { call(); throw new Error("sync query during init did not throw"); }
      catch (e) { if (!(e instanceof t.BackendInitializing)) throw e; }
    }
    // Metadata stays available during init.
    t.version();
    await a;
    if (t.backendState() !== "ready") throw new Error("state " + t.backendState());
    await t.initialize(); // idempotent once ready
    if (t.getAvailableBackends().length < 1) throw new Error("no devices after init");
  `);
});

test("async discovery initializes on demand and matches the sync view", () => {
  scenario(`
    const pending = t.getAvailableBackendsAsync();
    if (t.backendState() !== "initializing") throw new Error("state " + t.backendState());
    const devs = await pending;
    if (!devs.some((d) => d.kind === "cpu")) throw new Error("no cpu device");
    if (await t.backendAvailableAsync("cpu") !== true) throw new Error("cpu unavailable");
    const sync = t.getAvailableBackends().map((d) => d.name);
    if (JSON.stringify(sync) !== JSON.stringify(devs.map((d) => d.name))) {
      throw new Error("async/sync device lists differ");
    }
    // Devices from the async path are valid exact-selection handles.
    const cpu = devs.find((d) => d.deviceType === "cpu");
    try { await t.TranscribeModel.load("/nonexistent/model.gguf", { device: cpu }); }
    catch (e) { if (!(e instanceof t.ModelFileNotFound)) throw e; }
  `);
});

test("TranscribeModel.load awaits initialization instead of blocking", () => {
  scenario(`
    const loads = [
      t.TranscribeModel.load("/nonexistent/a.gguf"),
      t.TranscribeModel.load("/nonexistent/b.gguf"),
    ];
    // load() must not have run backend init synchronously.
    if (t.backendState() !== "initializing") throw new Error("state " + t.backendState());
    for (const r of await Promise.allSettled(loads)) {
      if (r.status !== "rejected" || !(r.reason instanceof t.ModelFileNotFound)) {
        throw new Error("expected ModelFileNotFound, got " + (r.reason ?? r.value));
      }
    }
    // A failed model load is local to that call: backends stay ready.
    if (t.backendState() !== "ready") throw new Error("state " + t.backendState());
    await t.initialize();
    await t.TranscribeModel.load("/nonexistent/c.gguf").catch((e) => {
      if (!(e instanceof t.ModelFileNotFound)) throw e;
    });
  `);
});

// The init summary line always reaches the handler, so both re-entry tests are
// guaranteed to exercise a sync query from inside initialization.
for (const [mode, start] of [
  ["async", "await t.initialize();"],
  ["inline", "t.backendAvailable('cpu');"],
]) {
  test(`a log handler re-entering a sync query during ${mode} init fails fast`, () => {
    scenario(`
      const seen = [];
      t.setLogHandler((level, msg) => {
        let outcome = "no-throw";
        try { t.getAvailableBackends(); } catch (e) { outcome = e.name; }
        seen.push({ msg, outcome });
      });
      ${start}
      if (!seen.some((s) => s.msg.includes("transcribe_init_backends"))) {
        throw new Error("init summary not logged: " + JSON.stringify(seen));
      }
      const bad = seen.filter((s) => s.outcome !== "BackendInitializing");
      if (bad.length) throw new Error("re-entrant query did not fail fast: " + JSON.stringify(bad));
    `);
  });
}

// A backend-init failure is permanent and cached. It needs a dynamic-backend
// libtranscribe copied into a directory with no backend modules, so it is a
// local-only check (CI builds compile backends in): set
// TRANSCRIBE_TEST_NO_MODULES_LIBRARY to such a library to run it.
const NO_MODULES = process.env.TRANSCRIBE_TEST_NO_MODULES_LIBRARY || "";
test(
  "a failed backend initialization is cached for the process",
  { skip: NO_MODULES && fs.existsSync(NO_MODULES) ? false : "TRANSCRIBE_TEST_NO_MODULES_LIBRARY unset" },
  () => {
    scenario(
      `
      const first = await t.initialize().then(() => null, (e) => e);
      if (!(first instanceof t.BackendError)) throw new Error("expected BackendError: " + first);
      if (t.backendState() !== "failed") throw new Error("state " + t.backendState());
      const again = await t.initialize().then(() => null, (e) => e);
      if (again !== first) throw new Error("expected the cached error");
      try { t.getAvailableBackends(); throw new Error("no throw"); }
      catch (e) { if (e !== first) throw e; }
      const load = await t.TranscribeModel.load("/nonexistent.gguf").then(() => null, (e) => e);
      if (load !== first) throw new Error("model load should surface the bootstrap failure");
      t.version(); // metadata still works
    `,
      { TRANSCRIBE_LIBRARY: NO_MODULES },
    );
  },
);

test("a library-load failure is not cached", () => {
  const missing = path.join(os.tmpdir(), `no-such-transcribe-${process.pid}.so`);
  scenario(
    `
    for (let i = 0; i < 2; i++) {
      const e = await t.initialize().then(() => null, (err) => err);
      if (!(e instanceof t.TranscribeError)) throw new Error("expected a load error: " + e);
      if (t.backendState() !== "uninitialized") throw new Error("state " + t.backendState());
    }
  `,
    { TRANSCRIBE_LIBRARY: missing },
  );
});

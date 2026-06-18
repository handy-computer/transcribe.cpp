import assert from "node:assert/strict";
import { modelTest, MODEL, STREAMING_MODEL, jfk, feedChunks } from "./common.mjs";
import { TranscribeModel, Busy } from "../dist/index.js";

modelTest("streaming commits text and finalizes", STREAMING_MODEL, async () => {
  const m = await TranscribeModel.load(STREAMING_MODEL);
  try {
    assert.equal(m.capabilities.supportsStreaming, true);
    const s = m.createSession();
    const stream = await s.stream({ commitPolicy: "stable_prefix" });
    assert.equal(stream.state, "active");
    await feedChunks(stream, jfk());
    const fin = await stream.finalize();
    assert.equal(fin.isFinal, true);
    assert.ok(stream.revision > 0);
    assert.equal(stream.lastStatus, null, "a healthy finished stream has no failure status");
    const t = stream.text;
    assert.ok((t.committed + t.full).trim().length > 0, "expected non-empty streamed text");
    stream.reset();
    assert.equal(stream.state, "idle");
    s.dispose();
  } finally {
    m.dispose();
  }
});

modelTest("stream reads reject while a feed is in flight", STREAMING_MODEL, async () => {
  const m = await TranscribeModel.load(STREAMING_MODEL);
  try {
    const s = m.createSession();
    const stream = await s.stream({ commitPolicy: "stable_prefix" });
    const chunk = jfk().subarray(0, 16000);

    const pending = stream.feed(chunk); // do NOT await — leave it in flight
    await Promise.resolve(); // let the feed's native call reach the worker
    assert.throws(() => stream.text, /in flight/i, "reading .text mid-feed must throw");
    assert.throws(() => stream.state, /in flight/i);
    assert.throws(() => stream.revision, /in flight/i);
    assert.throws(() => s.limits, /in flight/i);
    assert.throws(() => s.wasAborted, /in flight/i);

    await pending; // once awaited, reads are fine again
    assert.doesNotThrow(() => stream.text);

    stream.reset();
    s.dispose();
  } finally {
    m.dispose();
  }
});

modelTest("an active stream holds a model-wide compute lease", STREAMING_MODEL, async () => {
  const m = await TranscribeModel.load(STREAMING_MODEL);
  try {
    const a = m.createSession();
    const b = m.createSession();
    const stream = await a.stream({ commitPolicy: "stable_prefix" });
    await stream.feed(jfk().subarray(0, 16000));

    // While the stream is active, a sibling run/stream on the same model is
    // refused with Busy rather than allowed to race the C library.
    await assert.rejects(() => b.run(jfk().subarray(0, 16000)), Busy);
    await assert.rejects(() => b.stream(), Busy);

    // Finalizing releases the lease; a sibling stream may then begin.
    await stream.finalize();
    const b2 = await b.stream({ commitPolicy: "stable_prefix" });
    b2.reset();

    a.dispose();
    b.dispose();
  } finally {
    m.dispose();
  }
});

modelTest("reset() and dispose() are safe during an un-awaited feed", STREAMING_MODEL, async () => {
  const m = await TranscribeModel.load(STREAMING_MODEL);
  try {
    const s = m.createSession();
    const stream = await s.stream({ commitPolicy: "stable_prefix" });

    // Tear down while a feed is still in flight: the native reset/free must be
    // deferred behind the worker call, never run concurrently with it.
    const pending = stream.feed(jfk().subarray(0, 16000));
    stream.reset();
    assert.equal(stream.state, "idle", "reset() returns the stream to idle synchronously");
    await pending; // the in-flight feed still settles cleanly
    s.dispose();
  } finally {
    m.dispose();
  }
  assert.ok(true, "no native crash: teardown ordered behind the worker");
});

modelTest("disposing a session with an active stream frees the lease", STREAMING_MODEL, async () => {
  const m = await TranscribeModel.load(STREAMING_MODEL);
  try {
    const a = m.createSession();
    const b = m.createSession();
    const stream = await a.stream({ commitPolicy: "stable_prefix" });
    await stream.feed(jfk().subarray(0, 16000));

    // Dispose the session WITHOUT finalizing/resetting the stream.
    a.dispose();

    // The model lease must be released — a sibling session can stream again.
    const bs = await b.stream({ commitPolicy: "stable_prefix" });
    bs.reset();

    // The orphaned stream must fail fast, never touch the freed handle.
    assert.throws(() => stream.revision, /disposed/i);
    await assert.rejects(() => stream.feed(jfk().subarray(0, 16000)), /disposed/i);
    stream.reset(); // no-op, no crash

    b.dispose();
  } finally {
    m.dispose();
  }
});

modelTest("disposing a model with an active stream is crash-safe", STREAMING_MODEL, async () => {
  const m = await TranscribeModel.load(STREAMING_MODEL);
  const s = m.createSession();
  const stream = await s.stream({ commitPolicy: "stable_prefix" });
  await stream.feed(jfk().subarray(0, 16000));

  m.dispose(); // disposes the session, invalidates the stream, defers the frees

  // The orphaned stream must throw, not use a freed session/model handle.
  assert.throws(() => stream.text, /disposed/i);
  await assert.rejects(() => stream.finalize(), /disposed/i);
  stream.reset(); // no-op
  assert.ok(true, "no native crash after model dispose with an active stream");
});

modelTest("a stream queued before another's reset is refused, not overlapped", STREAMING_MODEL, async () => {
  const m = await TranscribeModel.load(STREAMING_MODEL);
  try {
    const a = m.createSession();
    const b = m.createSession();
    const sa = await a.stream({ commitPolicy: "stable_prefix" });
    await sa.feed(jfk().subarray(0, 16000));

    // Queue b.stream() while a's stream is still active, THEN reset a. The lease
    // is released only after a's native reset runs (FIFO), so at b's turn the
    // slot is still held — b must see Busy, not begin and overlap a's stream.
    const pb = b.stream({ commitPolicy: "stable_prefix" });
    sa.reset();
    await assert.rejects(pb, Busy);

    a.dispose();
    b.dispose();
  } finally {
    m.dispose();
  }
});

modelTest("a stream queued before another's dispose is refused, not overlapped", STREAMING_MODEL, async () => {
  const m = await TranscribeModel.load(STREAMING_MODEL);
  try {
    const a = m.createSession();
    const b = m.createSession();
    const sa = await a.stream({ commitPolicy: "stable_prefix" });
    await sa.feed(jfk().subarray(0, 16000));

    // Same ordering hazard via dispose: the lease releases only after a's
    // sessionFree runs, so b is refused rather than overlapping the live stream.
    const pb = b.stream({ commitPolicy: "stable_prefix" });
    a.dispose();
    await assert.rejects(pb, Busy);

    // Once a is actually gone, the slot is free again.
    const sb = await b.stream({ commitPolicy: "stable_prefix" });
    sb.reset();
    b.dispose();
  } finally {
    m.dispose();
  }
});

modelTest("a stream begin queued before its own dispose is rejected, not leaked", STREAMING_MODEL, async () => {
  const m = await TranscribeModel.load(STREAMING_MODEL);
  try {
    const a = m.createSession();
    const b = m.createSession();

    // Queue a stream begin, then dispose the session before the queued body
    // runs. The body must recheck disposal and abort — not begin a stream on a
    // dead session and leak the lease.
    const p = a.stream({ commitPolicy: "stable_prefix" });
    a.dispose();
    await assert.rejects(p, /disposed/i);

    // The lease never got claimed: another session can still stream.
    const sb = await b.stream({ commitPolicy: "stable_prefix" });
    sb.reset();
    b.dispose();
  } finally {
    m.dispose();
  }
});

modelTest("a stale stream wrapper cannot reset a newer same-session stream", STREAMING_MODEL, async () => {
  const m = await TranscribeModel.load(STREAMING_MODEL);
  try {
    const s = m.createSession();
    const first = await s.stream({ commitPolicy: "stable_prefix" });
    await first.feed(jfk().subarray(0, 16000));
    await first.finalize();

    // Queue a new begin, then reset the old finished wrapper before that begin
    // runs. The old reset must not clear the newer stream once it owns the
    // session's single native stream slot.
    const secondBegin = s.stream({ commitPolicy: "stable_prefix" });
    first.reset();
    const second = await secondBegin;
    assert.equal(second.state, "active");
    assert.throws(() => first.revision, /no longer current/i);

    first.reset(); // stale reset is a no-op, not a reset of `second`
    assert.equal(second.state, "active");
    await second.feed(jfk().subarray(0, 16000));
    assert.equal(second.state, "active");

    second.reset();
    s.dispose();
  } finally {
    m.dispose();
  }
});

modelTest("a non-streaming model rejects stream begin", MODEL, async () => {
  const m = await TranscribeModel.load(MODEL);
  try {
    const s = m.createSession();
    await assert.rejects(() => s.stream());
    s.dispose();
  } finally {
    m.dispose();
  }
});

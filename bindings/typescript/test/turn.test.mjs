// TURN role on the toy smart_turn (clip only) and sava (streaming, VAD + TURN)
// fixtures: info, run, stream bookkeeping and parity with run, the batched
// feed, error mapping and cancellation. The real models only smoke-test.

import assert from "node:assert/strict";
import {
  modelTest,
  jfk,
  SAVA_MODEL,
  SAVA_TOY_MODEL,
  SMART_TURN_MODEL,
  SMART_TURN_TOY_MODEL,
} from "./common.mjs";
import {
  Aborted,
  InvalidArgument,
  NotImplementedByModel,
  TranscribeError,
  TranscribeModel,
  TurnSession,
  UnsupportedRole,
} from "../dist/index.js";

function noise(n, seed = 7) {
  const out = new Float32Array(n);
  let s = (seed | 1) >>> 0;
  for (let i = 0; i < n; i++) {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    out[i] = ((s >>> 8) & 0xffffff) / 16777216 - 0.5;
  }
  return out;
}

const inUnit = (p) => p >= 0 && p <= 1;

modelTest("toy smart_turn: info, run, threshold, no streaming", SMART_TURN_TOY_MODEL, async () => {
  const m = await TranscribeModel.load(SMART_TURN_TOY_MODEL, { backend: "cpu" });
  const t = m.createTurnSession({ nThreads: 1 });
  try {
    assert.deepEqual(m.roles, ["turn"]);
    assert.deepEqual(m.turnInfo, {
      sampleRate: 16000,
      maxAudioMs: 200,
      frameSamples: 0,
      hasSpeech: false,
      threshold: 0.5,
    });
    assert.throws(() => m.vadInfo, UnsupportedRole);
    assert.throws(() => m.createVadSession(), UnsupportedRole);

    const pcm = noise(4000);
    const r = await t.run(pcm);
    assert.ok(inUnit(r.p));
    assert.equal(r.ended, r.p >= 0.5);
    assert.deepEqual(
      [r.probs, r.firstFrame, r.speechProbs, r.speechFirstFrame],
      [[], 0, [], 0],
    );
    // Longer audio is scored on its last maxAudioMs (3200 samples).
    assert.equal((await t.run(pcm.subarray(800))).p, r.p);

    // The threshold override decides `ended`; omitted = the model's 0.5.
    assert.equal((await t.run(pcm, { threshold: 0 })).ended, true);
    assert.equal((await t.run(pcm, { threshold: 1 })).ended, r.p >= 1);
    assert.equal((await t.run(pcm, { threshold: -1 })).ended, r.ended);
    assert.ok(t.timings.encodeMs > 0);

    // A clip model cannot stream.
    await assert.rejects(t.streamFeed(pcm), NotImplementedByModel);
    await assert.rejects(t.streamFlush(), NotImplementedByModel);
    await assert.rejects(TurnSession.streamFeedBatch([{ session: t, pcm }]), NotImplementedByModel);
    await t.streamReset();

    // Invalid input.
    await assert.rejects(t.run(new Float32Array(0)), TranscribeError);
    await assert.rejects(t.run(pcm, { threshold: 1.5 }), InvalidArgument);
    await assert.rejects(t.run(pcm, { threshold: NaN }), InvalidArgument);
    const bad = pcm.slice();
    bad[10] = NaN;
    await assert.rejects(t.run(bad), InvalidArgument);
    // ...and leaves the last good result readable on the next run.
    assert.equal((await t.run(pcm)).p, r.p);
  } finally {
    t.dispose();
    m.dispose();
  }
});

modelTest("toy sava: info, run, stream equals run", SAVA_TOY_MODEL, async () => {
  const m = await TranscribeModel.load(SAVA_TOY_MODEL, { backend: "cpu" });
  const t = m.createTurnSession({ nThreads: 1 });
  try {
    assert.deepEqual(m.roles, ["vad", "turn"]);
    const info = m.turnInfo;
    assert.deepEqual(info, {
      sampleRate: 16000,
      maxAudioMs: 0,
      frameSamples: 640,
      hasSpeech: true,
      threshold: info.threshold,
    });
    assert.ok(inUnit(info.threshold));

    const pcm = noise(16000 + 100);
    const nFrames = Math.ceil(pcm.length / 640);
    const r = await t.run(pcm);
    assert.equal(r.probs.length, nFrames);
    assert.equal(r.speechProbs.length, nFrames);
    assert.deepEqual([r.firstFrame, r.speechFirstFrame], [0, 0]);
    assert.equal(r.p, r.probs[nFrames - 1]);
    assert.equal(r.ended, r.p >= info.threshold);
    assert.ok([...r.probs, ...r.speechProbs].every(inUnit));
    assert.equal((await t.run(pcm, { threshold: 0 })).ended, true);
    assert.equal((await t.run(pcm, { threshold: 1 })).ended, r.p >= 1);

    // Odd chunks reproduce the run; each call continues where the last ended.
    const probs = [];
    const speech = [];
    const push = (f) => {
      assert.deepEqual([f.firstFrame, f.speechFirstFrame], [probs.length, speech.length]);
      probs.push(...f.probs);
      speech.push(...f.speechProbs);
    };
    for (let off = 0; off < pcm.length; off += 777) push(await t.streamFeed(pcm.subarray(off, off + 777)));
    push(await t.streamFlush());
    assert.deepEqual(probs, r.probs);
    assert.deepEqual(speech, r.speechProbs);

    // Empty feed scores nothing and keeps the position; reset / flush rewind to 0.
    await t.streamFeed(pcm.subarray(0, 1000));
    const empty = await t.streamFeed(new Float32Array(0));
    assert.deepEqual([empty.probs.length, empty.firstFrame], [0, 1]);
    await t.streamReset();
    assert.equal((await t.streamFeed(pcm.subarray(0, 1000))).firstFrame, 0);
    await t.streamFlush();
    assert.equal((await t.streamFeed(pcm.subarray(0, 1000))).firstFrame, 0);

    // Invalid input leaves the stream unchanged.
    const bad = pcm.slice(0, 640);
    bad[3] = Infinity;
    await assert.rejects(t.streamFeed(bad), InvalidArgument);
    assert.equal((await t.streamFeed(pcm.subarray(0, 640))).firstFrame, 1);
    await t.streamReset();
    assert.ok(t.timings.encodeMs >= 0);
  } finally {
    t.dispose();
    m.dispose();
  }
});

modelTest("toy sava: a batched feed equals feeding each session alone", SAVA_TOY_MODEL, async () => {
  const m = await TranscribeModel.load(SAVA_TOY_MODEL, { backend: "cpu" });
  const batched = [0, 1, 2].map(() => m.createTurnSession({ nThreads: 1 }));
  const alone = [0, 1, 2].map(() => m.createTurnSession({ nThreads: 1 }));
  try {
    const audio = [noise(9000, 3), noise(9000, 5), noise(9000, 9)];
    // Uneven chunks per session, including empty ones.
    const chunks = [
      [700, 0, 1300],
      [640, 2000, 1],
      [0, 333, 1919],
      [3000, 640, 0],
    ];
    const offs = [0, 0, 0];
    for (const round of chunks) {
      const items = batched.map((session, i) => ({
        session,
        pcm: audio[i].subarray(offs[i], offs[i] + round[i]),
      }));
      const got = await TurnSession.streamFeedBatch(items);
      assert.equal(got.length, 3);
      for (let i = 0; i < 3; i++) {
        const want = await alone[i].streamFeed(audio[i].subarray(offs[i], offs[i] + round[i]));
        assert.deepEqual(got[i], want, `round ${chunks.indexOf(round)} session ${i}`);
        offs[i] += round[i];
      }
    }
    // Flushing afterwards also matches.
    for (let i = 0; i < 3; i++) assert.deepEqual(await batched[i].streamFlush(), await alone[i].streamFlush());

    // Invalid batches (a disposed session: see the queued-dispose test).
    const [a, b] = batched;
    const one = noise(640);
    await assert.rejects(TurnSession.streamFeedBatch([]), InvalidArgument);
    await assert.rejects(
      TurnSession.streamFeedBatch([{ session: a, pcm: one }, { session: a, pcm: one }]),
      InvalidArgument,
    );
    await assert.rejects(TurnSession.streamFeedBatch([{ session: {}, pcm: one }]), InvalidArgument);
    const other = await TranscribeModel.load(SAVA_TOY_MODEL, { backend: "cpu" });
    const foreign = other.createTurnSession();
    try {
      await assert.rejects(
        TurnSession.streamFeedBatch([{ session: a, pcm: one }, { session: foreign, pcm: one }]),
        InvalidArgument,
      );
    } finally {
      other.dispose();
    }
    // A NaN in any chunk is rejected by the library and leaves every stream unchanged.
    const bad = one.slice();
    bad[0] = NaN;
    await assert.rejects(
      TurnSession.streamFeedBatch([{ session: a, pcm: one }, { session: b, pcm: bad }]),
      InvalidArgument,
    );
    assert.equal((await a.streamFeed(one)).firstFrame, 0);
    assert.equal((await b.streamFeed(one)).firstFrame, 0);
  } finally {
    for (const s of [...batched, ...alone]) s.dispose();
    m.dispose();
  }
});

modelTest("toy sava: an aborted signal cancels and resets the stream", SAVA_TOY_MODEL, async () => {
  const m = await TranscribeModel.load(SAVA_TOY_MODEL, { backend: "cpu" });
  const t = m.createTurnSession();
  const u = m.createTurnSession();
  try {
    const pcm = noise(3200);
    const ac = new AbortController();
    ac.abort();
    await assert.rejects(t.run(pcm, { signal: ac.signal }), Aborted);
    await t.streamFeed(pcm);
    await assert.rejects(t.streamFeed(pcm, { signal: ac.signal }), Aborted);
    // An abort resets the stream: the next feed starts at frame 0.
    assert.equal((await t.streamFeed(pcm)).firstFrame, 0);
    await u.streamFeed(pcm);
    await assert.rejects(
      TurnSession.streamFeedBatch([{ session: t, pcm }, { session: u, pcm }], { signal: ac.signal }),
      Aborted,
    );
    assert.equal((await t.streamFeed(pcm)).firstFrame, 0);
    assert.equal((await u.streamFeed(pcm)).firstFrame, 0);
    await assert.rejects(t.streamFlush({ signal: ac.signal }), Aborted);
  } finally {
    t.dispose();
    u.dispose();
    m.dispose();
  }
});

modelTest("toy sava: dispose defers the free; disposed sessions refuse", SAVA_TOY_MODEL, async () => {
  const m = await TranscribeModel.load(SAVA_TOY_MODEL, { backend: "cpu" });
  const t = m.createTurnSession();
  const u = m.createTurnSession();
  const busy = u.run(noise(64000)); // holds the model lock
  const queued = t.run(noise(6400)); // queued behind it
  t.dispose(); // the native free queues behind both; the queued run is refused
  await assert.rejects(queued, /disposed/);
  assert.equal((await busy).probs.length, 100);
  await assert.rejects(() => t.run(noise(640)), TranscribeError);
  await assert.rejects(() => t.streamFeed(noise(640)), TranscribeError);
  assert.throws(() => t.timings, TranscribeError);
  t.dispose(); // idempotent
  m.dispose(); // frees u, then the model, behind the lock
  await assert.rejects(() => u.run(noise(640)), TranscribeError);
});

modelTest("toy sava: a later session disposed while the batch waits", SAVA_TOY_MODEL, async () => {
  const m = await TranscribeModel.load(SAVA_TOY_MODEL, { backend: "cpu" });
  const [a, b, holder] = [0, 1, 2].map(() => m.createTurnSession({ nThreads: 1 }));
  try {
    const one = noise(640);
    await a.streamFeed(one.subarray(0, 300));
    const busy = holder.run(noise(64000)); // holds the model lock
    const queued = TurnSession.streamFeedBatch([
      { session: a, pcm: one },
      { session: b, pcm: one },
    ]);
    b.dispose(); // while the batch is queued behind the lock
    await assert.rejects(queued, (e) => e.constructor === TranscribeError && /disposed/.test(e.message));
    await busy;
    // Nothing was fed: a still has its 300 buffered samples.
    const r = await a.streamFeed(one.subarray(300, 640));
    assert.deepEqual([r.probs.length, r.firstFrame], [1, 0]);
  } finally {
    for (const s of [a, b, holder]) s.dispose();
    m.dispose();
  }
});

modelTest("real smart-turn v3.2: run on jfk", SMART_TURN_MODEL, async () => {
  const m = await TranscribeModel.load(SMART_TURN_MODEL, { backend: "cpu" });
  const t = m.createTurnSession();
  try {
    const info = m.turnInfo;
    assert.deepEqual([info.maxAudioMs, info.frameSamples, info.hasSpeech], [8000, 0, false]);
    const r = await t.run(jfk());
    assert.ok(inUnit(r.p));
    assert.equal(r.ended, r.p >= info.threshold);
    await assert.rejects(t.streamFeed(jfk().subarray(0, 640)), NotImplementedByModel);
  } finally {
    t.dispose();
    m.dispose();
  }
});

modelTest("real sava: stream and batch equal run on jfk", SAVA_MODEL, async () => {
  const m = await TranscribeModel.load(SAVA_MODEL, { backend: "cpu" });
  const t = m.createTurnSession();
  const u = m.createTurnSession();
  try {
    const pcm = jfk().subarray(0, 16000 * 3);
    const r = await t.run(pcm);
    assert.equal(r.probs.length, Math.ceil(pcm.length / m.turnInfo.frameSamples));
    const probs = [];
    for (let off = 0; off < pcm.length; off += 320) {
      const [a, b] = await TurnSession.streamFeedBatch([
        { session: t, pcm: pcm.subarray(off, off + 320) },
        { session: u, pcm: pcm.subarray(off, off + 320) },
      ]);
      assert.deepEqual(a, b);
      probs.push(...a.probs);
    }
    probs.push(...(await t.streamFlush()).probs);
    assert.equal(probs.length, r.probs.length);
    for (let i = 0; i < probs.length; i++) assert.ok(Math.abs(probs[i] - r.probs[i]) < 1e-3);
  } finally {
    t.dispose();
    u.dispose();
    m.dispose();
  }
});

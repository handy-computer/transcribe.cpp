//! Cancellation: a clean baseline run, and a cross-thread timer cancel of an
//! in-flight run on long (tiled) audio. Ports the Python C5 case. Skips the
//! abort assertion if the machine wins the race (run finishes before cancel).

mod common;

use std::thread;
use std::time::Duration;

use transcribe_cpp::{CancelToken, Error, Feature, Model, RunOptions};

#[test]
fn uncancelled_run_is_not_aborted() {
    // Even an uncancelled run crashes on Windows/MSVC if a cancel token is
    // installed — so the native whisper ÷0 (docs/known-issues.md "B") is
    // triggered by an abort callback being present, not by the abort firing.
    // Skip on Windows; covered on linux/macos.
    if cfg!(target_os = "windows") {
        eprintln!("skip uncancelled_run_is_not_aborted: native whisper ÷0 with a cancel token on Windows (known-issues B)");
        return;
    }
    let Some((model_path, pcm)) = common::smoke_fixtures("uncancelled_run_is_not_aborted") else {
        return;
    };
    let mut session = Model::load(&model_path).unwrap().session().unwrap();
    let token = CancelToken::new();
    session.set_cancel_token(&token);
    let result = session.run(&pcm, &RunOptions::default()).unwrap();
    assert!(result.text.to_lowercase().contains("country"));
    assert!(!session.was_aborted());
}

#[test]
fn cross_thread_cancel_of_in_flight_run() {
    // Cancelling an in-flight whisper run hits a native integer divide-by-zero
    // in whisper's abort/partial path on Windows/MSVC specifically (it faults as
    // STATUS_INTEGER_DIVIDE_BY_ZERO; the same abort works on linux/macos x86 and
    // is masked on arm64). It is a whisper-compute bug, not a binding bug — the
    // Rust layer maps the abort correctly — and reaches the abort path the same
    // way for short- or long-form audio. Skip on Windows until it is fixed;
    // tracked in docs/known-issues.md "B". The contract is still covered on
    // linux/macos.
    if cfg!(target_os = "windows") {
        eprintln!("skip cross_thread_cancel: native whisper abort ÷0 on Windows (known-issues B)");
        return;
    }
    let Some((model_path, pcm)) = common::smoke_fixtures("cross_thread_cancel_of_in_flight_run")
    else {
        return;
    };
    let model = Model::load(&model_path).unwrap();
    if !model.supports(Feature::Cancellation) {
        eprintln!("skip: model does not support cancellation");
        return;
    }
    // Tile the clip enough to be reliably mid-flight when we cancel.
    let long: Vec<f32> = std::iter::repeat(pcm.iter().copied())
        .take(4)
        .flatten()
        .collect();

    let mut session = model.session().unwrap();
    let token = CancelToken::new();
    session.set_cancel_token(&token);

    // Fire the cancel shortly after the run starts, from another thread.
    let canceller = {
        let token = token.clone();
        thread::spawn(move || {
            thread::sleep(Duration::from_millis(40));
            token.cancel();
        })
    };

    let outcome = session.run(&long, &RunOptions::default());
    canceller.join().unwrap();

    match outcome {
        Err(Error::Aborted { partial, .. }) => {
            // The abort fired mid-run: partial transcript preserved, flag set.
            assert!(session.was_aborted());
            assert!(partial.is_some(), "aborted run should carry a partial");
        }
        Ok(_) => {
            // The machine won the race (run completed before the 40 ms timer).
            assert!(!session.was_aborted());
            eprintln!("note: run completed before cancel fired (fast machine)");
        }
        Err(other) => panic!("unexpected error: {other:?}"),
    }
}

//! backend-select — device discovery, exact selection, backend-policy failure.
//!
//!     cargo run --example backend-select -- [model.gguf]
//!
//! Device discovery needs no model and always runs. With a model it selects the
//! enumerated CPU device exactly, then demonstrates that an unavailable backend
//! policy fails cleanly from the request path rather than crashing.

#[path = "common/mod.rs"]
mod common;

use transcribe_cpp::{
    backend_available, devices, init_backends_default, Backend, Model, ModelOptions,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Register backend modules before anything else. A no-op in a compiled-in
    // build; in a `dynamic-backends` build it loads the per-ISA CPU / GPU
    // modules so the device list below reflects what actually registered. Must
    // happen once, before the first model load.
    init_backends_default()?;

    let registered = devices();
    println!("registered compute devices:");
    for d in &registered {
        println!("  {} [{}] — {}", d.name, d.kind, d.description);
    }
    println!("\nbackend availability:");
    for b in [
        Backend::Cpu,
        Backend::Metal,
        Backend::Vulkan,
        Backend::Cuda,
        Backend::Rocm,
    ] {
        println!("  {b:?} = {}", backend_available(b));
    }

    // The first optional backend this build does NOT have — used below to show
    // a request that cannot be satisfied. (CPU is always present, so it is not
    // a candidate.)
    let unavailable = [
        Backend::Cuda,
        Backend::Rocm,
        Backend::Vulkan,
        Backend::Metal,
    ]
    .into_iter()
    .find(|&b| !backend_available(b));

    let mut args = std::env::args().skip(1);
    let Some(model_path) = common::model_path(args.next().as_deref()) else {
        eprintln!("\nskip the model half of backend-select: no model found");
        return Ok(());
    };

    // Exact selection uses a process-local Device handle, not its display
    // index. CPU is always registered and provides a deterministic example.
    let cpu = registered
        .iter()
        .find(|device| device.kind == "cpu")
        .expect("the CPU device must be registered")
        .clone();
    let model = Model::load_with(
        &model_path,
        &ModelOptions {
            device: Some(cpu.clone()),
            ..Default::default()
        },
    )?;
    println!(
        "\nselected exact device {} -> bound backend: {}",
        model.device()?.name,
        model.backend()
    );
    assert_eq!(model.device()?, cpu);
    drop(model);

    // Graceful failure: requesting an unavailable backend must error cleanly.
    match unavailable {
        Some(b) => match Model::load_with(
            &model_path,
            &ModelOptions {
                backend: b,
                ..Default::default()
            },
        ) {
            Ok(_) => println!("requesting {b:?} unexpectedly succeeded"),
            Err(e) => println!("requesting {b:?} failed cleanly: {e}"),
        },
        None => println!("every optional backend is available on this build"),
    }
    Ok(())
}

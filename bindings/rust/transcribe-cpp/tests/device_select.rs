//! Model-gated exact-device selection tests.

mod common;

use transcribe_cpp::{devices, Backend, Error, Model, ModelOptions};

#[test]
fn every_primary_device_can_be_selected_exactly() {
    let Some((model_path, _)) =
        common::smoke_fixtures("every_primary_device_can_be_selected_exactly")
    else {
        return;
    };

    for device in devices()
        .into_iter()
        .filter(|d| d.device_type != transcribe_cpp::DeviceType::Accel)
    {
        let result = Model::load_with(
            &model_path,
            &ModelOptions {
                backend: Backend::Auto,
                device: Some(device.clone()),
            },
        );
        match result {
            Ok(model) => assert_eq!(model.device().unwrap(), device),
            Err(Error::Backend(_)) => {
                // A registered device may still fail initialization on the
                // current driver. Exact selection must fail rather than move.
            }
            Err(error) => panic!("unexpected exact-device error: {error}"),
        }
    }
}

#[test]
fn explicit_backend_must_match_device() {
    let Some((model_path, _)) = common::smoke_fixtures("explicit_backend_must_match_device") else {
        return;
    };
    let Some(gpu) = devices().into_iter().find(|d| {
        matches!(
            d.device_type,
            transcribe_cpp::DeviceType::Gpu | transcribe_cpp::DeviceType::Igpu
        )
    }) else {
        return;
    };

    let error = Model::load_with(
        &model_path,
        &ModelOptions {
            backend: Backend::Cpu,
            device: Some(gpu),
        },
    )
    .unwrap_err();
    assert!(matches!(error, Error::InvalidArgument(_)));
}

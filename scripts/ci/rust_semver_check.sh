#!/usr/bin/env bash
# cargo semver-checks for transcribe-cpp-types and transcribe-cpp against their
# latest release on crates.io. The tool judges a break against the version in
# the tree: on a PR (version = last release) any break fails; at release, after
# prepare.py, a minor bump (0.5 -> 0.6) allows breaks and a patch bump does not.
#
#   scripts/ci/rust_semver_check.sh
#
# Runs informational in rust-ci and blocking in publish.yml's rust-verify /
# rust-rehearsal. Features: serde on (the derives are API), no backend features
# (they don't change the Rust API and need SDKs to build).
set -euo pipefail

# Skip until transcribe-cpp-types has a release to compare against (the first
# is 0.5.1). Before that, cargo-semver-checks, which does not follow re-exports
# from another crate, reports every type moved there as removed from
# transcribe-cpp. Starts checking on its own once 0.5.1 is published.
code=$(curl -s -o /dev/null -w '%{http_code}' \
  -A 'transcribe-cpp-ci (+https://github.com/handy-computer/transcribe.cpp)' \
  https://crates.io/api/v1/crates/transcribe-cpp-types || true)
case "$code" in
  200) ;;
  404)
    echo "::notice::transcribe-cpp-types is not on crates.io yet; semver checks start with its first release"
    exit 0 ;;
  *)
    echo "::warning::crates.io unreachable (HTTP $code); semver checks skipped"
    exit 0 ;;
esac

cargo semver-checks -p transcribe-cpp-types --only-explicit-features --features serde
cargo semver-checks -p transcribe-cpp --only-explicit-features --features serde

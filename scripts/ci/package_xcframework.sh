#!/usr/bin/env bash
#
# package_xcframework.sh — zip the built TranscribeCpp.xcframework for release
# and print its SwiftPM binaryTarget checksum.
#
# The release flow (publish.yml, tag-gated) uses this after
# build_xcframework.sh: the zip is uploaded as a GitHub release asset and the
# printed checksum goes into the mirror repo's Package.swift
# `binaryTarget(url:checksum:)`. "Releases are cut from CI, never a laptop"
# (requirements §5).
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${REPO_ROOT}/bindings/swift/build-apple"
XCFRAMEWORK="${OUT_DIR}/TranscribeCpp.xcframework"
ZIP="${OUT_DIR}/TranscribeCpp.xcframework.zip"

if [[ ! -d "$XCFRAMEWORK" ]]; then
    echo "error: $XCFRAMEWORK not found — run scripts/ci/build_xcframework.sh first" >&2
    exit 1
fi

# Bundle the third-party + project license texts inside the artifact so they
# travel with the binaryTarget zip (requirements §5: vendored native code ships
# its license texts).
cp "${REPO_ROOT}/LICENSE" "${XCFRAMEWORK}/LICENSE"
cp "${REPO_ROOT}/ggml/LICENSE" "${XCFRAMEWORK}/LICENSE.ggml"
cp "${REPO_ROOT}/src/third_party/miniz/LICENSE" "${XCFRAMEWORK}/LICENSE.miniz"

# Deterministic zip from the output dir (store the path as
# "TranscribeCpp.xcframework/..." so SwiftPM unpacks it correctly).
rm -f "$ZIP"
( cd "$OUT_DIR" && /usr/bin/zip -qr -X "$(basename "$ZIP")" "$(basename "$XCFRAMEWORK")" )

echo "zip:      $ZIP"
echo "size:     $(du -h "$ZIP" | cut -f1)"
echo -n "checksum: "
swift package --package-path "${REPO_ROOT}/bindings/swift" compute-checksum "$ZIP"

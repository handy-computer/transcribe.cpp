#!/usr/bin/env bash
#
# build_xcframework.sh — produce bindings/swift/build-apple/TranscribeCpp.xcframework
#
# The Swift binding consumes the native library as a prebuilt static
# `.xcframework` binaryTarget (notes/swift-bindings-plan.md; requirements §5).
# This script is the lane that produces it. Adapted from whisper.cpp's
# build-xcframework.sh, retargeted to transcribe.cpp's CMake tree and the
# project's per-slice backend posture.
#
# Slices and backends (decision: Metal only where ggml-metal is reliable):
#   macos        arm64(Metal) + x86_64(CPU-only)        -> universal
#   ios-device   arm64(Metal)
#   ios-sim      arm64(CPU-only) + x86_64(CPU-only)     -> universal
#
# Per (slice,arch) we build a static libtranscribe + ggml, MERGE all the
# archives into one (collision-safe — see below), then `lipo` arches within a
# slice and hand each slice to `xcodebuild -create-xcframework -library`.
#
# Collision-safe merge: our per-family CMake target emits many same-basename
# objects (model.cpp.o, encoder.cpp.o, ...). `libtool -static` DEDUPES archive
# members by basename and silently drops objects. We instead partial-link
# (`ld -r -all_load`) every archive into ONE relocatable object, then wrap that
# single object — no basename collisions possible.
#
# Usage:
#   scripts/ci/build_xcframework.sh
#   TRANSCRIBE_XCFRAMEWORK_SLICES="macos" scripts/ci/build_xcframework.sh   # subset
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_ROOT="${REPO_ROOT}/tmp/xcframework"
OUT_DIR="${REPO_ROOT}/bindings/swift/build-apple"
XCFRAMEWORK="${OUT_DIR}/TranscribeCpp.xcframework"

MACOS_MIN="${MACOS_MIN_OS_VERSION:-13.0}"
IOS_MIN="${IOS_MIN_OS_VERSION:-16.0}"

SLICES="${TRANSCRIBE_XCFRAMEWORK_SLICES:-macos ios-device ios-sim}"

# Prefer Ninja, fall back to Unix Makefiles.
if command -v ninja >/dev/null 2>&1; then
    GENERATOR="Ninja"
else
    GENERATOR="Unix Makefiles"
fi

log() { printf '\n=== %s ===\n' "$*" >&2; }

# Deterministic build directory for a (tag, arch) pair. Both build_arch and the
# slice loop derive the path from this — never capture it via stdout (cmake
# writes to stdout and would pollute the value).
arch_bdir() { printf '%s' "${BUILD_ROOT}/$1-$2"; }

# build_arch <tag> <system_name> <sdk> <arch> <metal:ON|OFF> <min>
#   system_name="" for macOS (host), "iOS" for iOS slices.
build_arch() {
    local tag="$1" system_name="$2" sdk="$3" arch="$4" metal="$5" min="$6"
    local bdir; bdir="$(arch_bdir "$tag" "$arch")"
    rm -rf "$bdir"

    local extra=()
    [[ -n "$system_name" ]] && extra+=(-DCMAKE_SYSTEM_NAME="$system_name")
    if [[ "$metal" == "ON" ]]; then
        extra+=(-DTRANSCRIBE_METAL=ON -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON)
    else
        extra+=(-DTRANSCRIBE_METAL=OFF -DGGML_METAL=OFF)
    fi

    log "configure ${tag}/${arch} (metal=${metal})"
    cmake -B "$bdir" -S "$REPO_ROOT" -G "$GENERATOR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_OSX_SYSROOT="$sdk" \
        -DCMAKE_OSX_ARCHITECTURES="$arch" \
        -DCMAKE_OSX_DEPLOYMENT_TARGET="$min" \
        -DTRANSCRIBE_BUILD_SHARED=OFF \
        -DTRANSCRIBE_BUILD_TESTS=OFF \
        -DTRANSCRIBE_BUILD_EXAMPLES=OFF \
        -DTRANSCRIBE_BUILD_TOOLS=OFF \
        -DTRANSCRIBE_INSTALL=OFF \
        -DTRANSCRIBE_USE_OPENMP=OFF \
        -DGGML_OPENMP=OFF \
        -DGGML_NATIVE=OFF \
        "${extra[@]}"

    log "build ${tag}/${arch}"
    cmake --build "$bdir" --target transcribe --config Release --parallel
}

# merge_arch <bdir> <arch> <ld_platform> <sdk> <min> <out.a>
#   ld_platform: macos | ios | ios-simulator
merge_arch() {
    local bdir="$1" arch="$2" ld_platform="$3" sdk="$4" min="$5" out="$6"
    local sdkver
    sdkver="$(xcrun --sdk "$sdk" --show-sdk-version)"

    # Every static archive the build produced (ggml-metal only when Metal was on).
    local archives
    archives="$(find "$bdir" \( -name 'libtranscribe.a' -o -name 'libggml*.a' \) | sort)"

    log "merge ${arch} (${ld_platform})"
    # shellcheck disable=SC2086
    xcrun ld -r -arch "$arch" \
        -platform_version "$ld_platform" "$min" "$sdkver" \
        -all_load $archives \
        -o "${bdir}/combined.o"
    libtool -static -o "$out" "${bdir}/combined.o"
}

stage_headers() {
    local hdr="$1"
    rm -rf "$hdr" && mkdir -p "$hdr/transcribe"
    cp "${REPO_ROOT}/include/transcribe.h" "$hdr/"
    cp "${REPO_ROOT}/include/transcribe/"*.h "$hdr/transcribe/"
    cat > "${hdr}/module.modulemap" <<'EOF'
module CTranscribe {
    header "transcribe/extensions.h"
    export *
}
EOF
}

# ---- Build each requested slice into a single (possibly fat) static lib ----
rm -rf "$BUILD_ROOT" "$XCFRAMEWORK"
mkdir -p "$BUILD_ROOT" "$OUT_DIR"
HEADERS="${BUILD_ROOT}/Headers"
stage_headers "$HEADERS"

XCARGS=()

for slice in $SLICES; do
    case "$slice" in
        # Each slice's final library MUST be named `libtranscribe.a` (SwiftPM
        # derives a `-ltranscribe` flag from the static binaryTarget's filename)
        # and live in its own directory so create-xcframework can take several.
        macos)
            sdir="${BUILD_ROOT}/slice-macos"; mkdir -p "$sdir"
            build_arch macos "" macosx arm64  ON  "$MACOS_MIN"
            merge_arch "$(arch_bdir macos arm64)"  arm64  macos macosx "$MACOS_MIN" "${BUILD_ROOT}/macos-arm64.a"
            build_arch macos "" macosx x86_64 OFF "$MACOS_MIN"
            merge_arch "$(arch_bdir macos x86_64)" x86_64 macos macosx "$MACOS_MIN" "${BUILD_ROOT}/macos-x86_64.a"
            lipo -create "${BUILD_ROOT}/macos-arm64.a" "${BUILD_ROOT}/macos-x86_64.a" \
                -output "${sdir}/libtranscribe.a"
            XCARGS+=(-library "${sdir}/libtranscribe.a" -headers "$HEADERS")
            ;;
        ios-device)
            sdir="${BUILD_ROOT}/slice-ios-device"; mkdir -p "$sdir"
            build_arch ios-device iOS iphoneos arm64 ON "$IOS_MIN"
            merge_arch "$(arch_bdir ios-device arm64)" arm64 ios iphoneos "$IOS_MIN" "${sdir}/libtranscribe.a"
            XCARGS+=(-library "${sdir}/libtranscribe.a" -headers "$HEADERS")
            ;;
        ios-sim)
            sdir="${BUILD_ROOT}/slice-ios-sim"; mkdir -p "$sdir"
            build_arch ios-sim iOS iphonesimulator arm64  OFF "$IOS_MIN"
            merge_arch "$(arch_bdir ios-sim arm64)"  arm64  ios-simulator iphonesimulator "$IOS_MIN" "${BUILD_ROOT}/ios-sim-arm64.a"
            build_arch ios-sim iOS iphonesimulator x86_64 OFF "$IOS_MIN"
            merge_arch "$(arch_bdir ios-sim x86_64)" x86_64 ios-simulator iphonesimulator "$IOS_MIN" "${BUILD_ROOT}/ios-sim-x86_64.a"
            lipo -create "${BUILD_ROOT}/ios-sim-arm64.a" "${BUILD_ROOT}/ios-sim-x86_64.a" \
                -output "${sdir}/libtranscribe.a"
            XCARGS+=(-library "${sdir}/libtranscribe.a" -headers "$HEADERS")
            ;;
        *)
            echo "unknown slice: $slice" >&2; exit 2 ;;
    esac
done

log "create xcframework"
xcodebuild -create-xcframework "${XCARGS[@]}" -output "$XCFRAMEWORK"

log "done -> $XCFRAMEWORK"
find "$XCFRAMEWORK" -maxdepth 1 -mindepth 1 -type d | sort >&2

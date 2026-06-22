#!/usr/bin/env bash
#
# build_xcframework.sh — produce bindings/swift/build-apple/TranscribeCpp.xcframework
#
# The Swift binding consumes the native library as a prebuilt `.xcframework`
# binaryTarget (notes/swift-bindings-plan.md; requirements §5). This script is
# the lane that produces it.
#
# We ship a DYNAMIC `.framework` per slice (the same posture as
# llama.cpp/whisper.cpp), NOT a bare static `.a`. Why: ggml-metal is
# Objective-C and creates libdispatch objects during Metal device-init. A
# static-lib xcframework pushes the burden of correctly realizing that
# Objective-C / libdispatch runtime metadata onto every consumer app's final
# link — and when an app links it without `-ObjC`/`-force_load`, the dispatch
# objects come up with the wrong runtime class and the app aborts on launch
# with "-[OS_dispatch_mach_msg _setContext:]: unrecognized selector". A dynamic
# framework links those dependencies into the dylib image itself, so dyld
# registers the Objective-C image normally at load and consumers need NO special
# flags. This also lets us drop the old `ld -r -all_load` merge entirely.
#
# Slices and backends (Metal only where ggml-metal is reliable):
#   macos        arm64(Metal) + x86_64(CPU-only)        -> universal framework
#   ios-device   arm64(Metal)
#   ios-sim      arm64(CPU-only) + x86_64(CPU-only)     -> universal framework
#
# Per (slice,arch) we build static libtranscribe + ggml, then link a dylib with
# `-Wl,-force_load` of EVERY archive — force_load (not `libtool -static`) so the
# per-family same-basename objects (model.cpp.o, encoder.cpp.o, ...) are all
# kept; `libtool -static` dedups archive members by basename and would silently
# drop them. lipo the per-arch dylibs into the slice's framework binary.
#
# Usage:
#   scripts/ci/build_xcframework.sh
#   TRANSCRIBE_XCFRAMEWORK_SLICES="ios-sim" scripts/ci/build_xcframework.sh   # subset
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_ROOT="${REPO_ROOT}/tmp/xcframework"
OUT_DIR="${REPO_ROOT}/bindings/swift/build-apple"
XCFRAMEWORK="${OUT_DIR}/TranscribeCpp.xcframework"
FRAMEWORK_NAME="CTranscribe"   # framework + Clang module name (Swift wrapper does `import CTranscribe`)

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

# Deterministic build directory for a (tag, arch) pair.
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

# link_dylib <bdir> <arch> <sdk> <min_flag> <install_name> <out.dylib>
#   Links one architecture's dynamic library, force-loading every static
#   archive cmake produced so all members (incl. same-basename objects and the
#   embedded metallib) and their Objective-C metadata land in the image.
link_dylib() {
    local bdir="$1" arch="$2" sdk="$3" min_flag="$4" install_name="$5" out="$6"
    local force_args=()
    local a
    while IFS= read -r a; do
        [[ -n "$a" ]] && force_args+=(-Wl,-force_load,"$a")
    done < <(find "$bdir" \( -name 'libtranscribe.a' -o -name 'libggml*.a' \) | sort)

    log "link dylib ${arch} (${sdk})"
    xcrun -sdk "$sdk" clang++ -dynamiclib \
        -isysroot "$(xcrun --sdk "$sdk" --show-sdk-path)" \
        -arch "$arch" \
        "$min_flag" \
        "${force_args[@]}" \
        -framework Foundation -framework Metal -framework MetalKit -framework Accelerate \
        -lz \
        -install_name "$install_name" \
        -o "$out"
}

# assemble_framework <framework_root> <slice> <min> <fat_dylib>
#   slice ∈ {macos, ios-device, ios-sim}. macOS uses the versioned bundle
#   layout; iOS device/sim use the flat layout.
assemble_framework() {
    local fw="$1" slice="$2" min="$3" dylib="$4"
    rm -rf "$fw"

    local hpath mpath plist binpath supported dtplatform sdkname device_family=""
    case "$slice" in
        macos)
            mkdir -p "$fw/Versions/A/Headers" "$fw/Versions/A/Modules" "$fw/Versions/A/Resources"
            ln -sf A "$fw/Versions/Current"
            ln -sf Versions/Current/Headers "$fw/Headers"
            ln -sf Versions/Current/Modules "$fw/Modules"
            ln -sf Versions/Current/Resources "$fw/Resources"
            ln -sf "Versions/Current/${FRAMEWORK_NAME}" "$fw/${FRAMEWORK_NAME}"
            hpath="$fw/Versions/A/Headers"; mpath="$fw/Versions/A/Modules"
            plist="$fw/Versions/A/Resources/Info.plist"; binpath="$fw/Versions/A/${FRAMEWORK_NAME}"
            supported="MacOSX"; dtplatform="macosx"; sdkname="macosx${min}" ;;
        ios-device|ios-sim)
            mkdir -p "$fw/Headers" "$fw/Modules"
            hpath="$fw/Headers"; mpath="$fw/Modules"
            plist="$fw/Info.plist"; binpath="$fw/${FRAMEWORK_NAME}"
            device_family='    <key>UIDeviceFamily</key>
    <array>
        <integer>1</integer>
        <integer>2</integer>
    </array>
'
            if [[ "$slice" == "ios-sim" ]]; then
                supported="iPhoneSimulator"; dtplatform="iphonesimulator"; sdkname="iphonesimulator${min}"
            else
                supported="iPhoneOS"; dtplatform="iphoneos"; sdkname="iphoneos${min}"
            fi ;;
    esac

    cp "$dylib" "$binpath"

    # Headers: a framework's Headers dir is not on the quoted-include search
    # path for subdirectories, so the public headers' `#include "transcribe/x.h"`
    # would not resolve. Flatten every header into Headers/ and strip the
    # "transcribe/" include prefix in the STAGED copies (the source headers are
    # untouched) so the cross-includes become same-directory quoted includes —
    # the layout llama.cpp/whisper.cpp use. The Swift wrapper imports the module,
    # not individual header paths, so flattening is transparent to consumers.
    cp "${REPO_ROOT}/include/transcribe.h" "$hpath/transcribe.h"
    cp "${REPO_ROOT}/include/transcribe/"*.h "$hpath/"
    /usr/bin/sed -i '' 's|#include "transcribe/|#include "|g' "$hpath"/*.h

    {
        echo "framework module ${FRAMEWORK_NAME} {"
        local h
        for h in "$hpath"/*.h; do
            echo "    header \"$(basename "$h")\""
        done
        cat <<EOF

    link "c++"
    link "z"
    link framework "Accelerate"
    link framework "Metal"
    link framework "Foundation"

    export *
}
EOF
    } > "$mpath/module.modulemap"

    cat > "$plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleExecutable</key>
    <string>${FRAMEWORK_NAME}</string>
    <key>CFBundleIdentifier</key>
    <string>com.transcribe.${FRAMEWORK_NAME}</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>${FRAMEWORK_NAME}</string>
    <key>CFBundlePackageType</key>
    <string>FMWK</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>MinimumOSVersion</key>
    <string>${min}</string>
    <key>CFBundleSupportedPlatforms</key>
    <array>
        <string>${supported}</string>
    </array>
${device_family}    <key>DTPlatformName</key>
    <string>${dtplatform}</string>
    <key>DTSDKName</key>
    <string>${sdkname}</string>
</dict>
</plist>
EOF

    # Ad-hoc sign so the framework is valid in the xcframework; consumers
    # re-sign on "Embed & Sign".
    codesign --force --sign - --timestamp=none "$fw" >/dev/null 2>&1 || true
}

# ---- Build each requested slice into a CTranscribe.framework ----
rm -rf "$BUILD_ROOT" "$XCFRAMEWORK"
mkdir -p "$BUILD_ROOT" "$OUT_DIR"

INSTALL_NAME_FLAT="@rpath/${FRAMEWORK_NAME}.framework/${FRAMEWORK_NAME}"
INSTALL_NAME_MACOS="@rpath/${FRAMEWORK_NAME}.framework/Versions/Current/${FRAMEWORK_NAME}"

XCARGS=()

for slice in $SLICES; do
    case "$slice" in
        macos)
            build_arch macos "" macosx arm64  ON  "$MACOS_MIN"
            link_dylib "$(arch_bdir macos arm64)"  arm64  macosx "-mmacosx-version-min=$MACOS_MIN" "$INSTALL_NAME_MACOS" "${BUILD_ROOT}/macos-arm64.dylib"
            build_arch macos "" macosx x86_64 OFF "$MACOS_MIN"
            link_dylib "$(arch_bdir macos x86_64)" x86_64 macosx "-mmacosx-version-min=$MACOS_MIN" "$INSTALL_NAME_MACOS" "${BUILD_ROOT}/macos-x86_64.dylib"
            lipo -create "${BUILD_ROOT}/macos-arm64.dylib" "${BUILD_ROOT}/macos-x86_64.dylib" -output "${BUILD_ROOT}/macos.dylib"
            assemble_framework "${BUILD_ROOT}/fw-macos/${FRAMEWORK_NAME}.framework" macos "$MACOS_MIN" "${BUILD_ROOT}/macos.dylib"
            XCARGS+=(-framework "${BUILD_ROOT}/fw-macos/${FRAMEWORK_NAME}.framework")
            ;;
        ios-device)
            build_arch ios-device iOS iphoneos arm64 ON "$IOS_MIN"
            link_dylib "$(arch_bdir ios-device arm64)" arm64 iphoneos "-mios-version-min=$IOS_MIN" "$INSTALL_NAME_FLAT" "${BUILD_ROOT}/ios-device-arm64.dylib"
            # NOTE: do NOT run `vtool -set-build-version` here. The clang link
            # above already stamps a correct LC_BUILD_VERSION (platform iOS,
            # minos $IOS_MIN, sdk = the real SDK). `vtool -set-build-version ios
            # $IOS_MIN $IOS_MIN` overwrites the *sdk* field with the min-OS
            # version (e.g. 16.0); dyld on newer iOS (e.g. 27) keys runtime/ABI
            # init off that sdk field and a stale "sdk 16" framework takes a
            # legacy path that aborts in libxpc's launch initializer.
            assemble_framework "${BUILD_ROOT}/fw-ios-device/${FRAMEWORK_NAME}.framework" ios-device "$IOS_MIN" "${BUILD_ROOT}/ios-device-arm64.dylib"
            XCARGS+=(-framework "${BUILD_ROOT}/fw-ios-device/${FRAMEWORK_NAME}.framework")
            ;;
        ios-sim)
            build_arch ios-sim iOS iphonesimulator arm64  OFF "$IOS_MIN"
            link_dylib "$(arch_bdir ios-sim arm64)"  arm64  iphonesimulator "-mios-simulator-version-min=$IOS_MIN" "$INSTALL_NAME_FLAT" "${BUILD_ROOT}/ios-sim-arm64.dylib"
            build_arch ios-sim iOS iphonesimulator x86_64 OFF "$IOS_MIN"
            link_dylib "$(arch_bdir ios-sim x86_64)" x86_64 iphonesimulator "-mios-simulator-version-min=$IOS_MIN" "$INSTALL_NAME_FLAT" "${BUILD_ROOT}/ios-sim-x86_64.dylib"
            lipo -create "${BUILD_ROOT}/ios-sim-arm64.dylib" "${BUILD_ROOT}/ios-sim-x86_64.dylib" -output "${BUILD_ROOT}/ios-sim.dylib"
            assemble_framework "${BUILD_ROOT}/fw-ios-sim/${FRAMEWORK_NAME}.framework" ios-sim "$IOS_MIN" "${BUILD_ROOT}/ios-sim.dylib"
            XCARGS+=(-framework "${BUILD_ROOT}/fw-ios-sim/${FRAMEWORK_NAME}.framework")
            ;;
        *)
            echo "unknown slice: $slice" >&2; exit 2 ;;
    esac
done

log "create xcframework"
xcodebuild -create-xcframework "${XCARGS[@]}" -output "$XCFRAMEWORK"

log "done -> $XCFRAMEWORK"
find "$XCFRAMEWORK" -maxdepth 1 -mindepth 1 -type d | sort >&2

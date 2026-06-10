#!/usr/bin/env bash
# Build the Vulkan toolchain (headers + loader + glslc) from pinned Khronos
# tags inside the manylinux_2_28 container, for the Linux cpu+vulkan provider
# wheel (cibuildwheel before-all hook; see [tool.cibuildwheel.linux] in the
# repo-root pyproject.toml).
#
# Why from source: LunarG's prebuilt SDK binaries require glibc 2.34+ (checked
# 2026-06: glslc 1.4.350.1 imports GLIBC_2.34), which rules them out inside
# manylinux_2_28 (glibc 2.28) — and LunarG prunes old SDK downloads, so pinned
# URLs rot. Khronos/Google git tags are permanent and the build is small.
# This is also what forced the manylinux2014 -> manylinux_2_28 baseline move
# (the plan's documented "revisit if Vulkan toolchains force it" carve-out:
# manylinux2014 images are EOL and their toolchain is too old regardless).
#
# The result is cached across CI runs via a host-mounted volume
# ($VK_TOOLCHAIN_CACHE, wired up in .github/workflows/python-wheels.yml):
# first run ~8 min, cached runs seconds.

set -euo pipefail

VULKAN_SDK_TAG="vulkan-sdk-1.4.350.0"   # Vulkan-Headers + Vulkan-Loader
SHADERC_TAG="v2026.2"                   # google/shaderc (glslc)

PREFIX=/usr/local
CACHE_ROOT="${VK_TOOLCHAIN_CACHE:-}"
STAMP="${VULKAN_SDK_TAG}_shaderc-${SHADERC_TAG}"

# Restore from the mounted cache if this exact toolchain was built before.
if [[ -n "$CACHE_ROOT" && -d "$CACHE_ROOT/$STAMP" ]]; then
    echo "vulkan-toolchain: restoring $STAMP from cache"
    cp -a "$CACHE_ROOT/$STAMP/." "$PREFIX/"
    glslc --version
    exit 0
fi

# cmake/ninja: take them from a container python (the manylinux image always
# has /opt/python; system cmake may be absent or ancient).
PYBIN=$(ls -d /opt/python/cp312-cp312/bin 2>/dev/null || ls -d /opt/python/cp3*/bin | head -1)
"$PYBIN/pip" install --quiet cmake ninja
export PATH="$PYBIN:$PATH"

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
STAGE="$WORK/stage"   # install everything here, then copy to prefix + cache
mkdir -p "$STAGE"

echo "vulkan-toolchain: building $STAMP (no cache hit)"

# 1. Vulkan-Headers: header-only install, seconds.
git clone --quiet --depth 1 --branch "$VULKAN_SDK_TAG" \
    https://github.com/KhronosGroup/Vulkan-Headers "$WORK/headers"
cmake -S "$WORK/headers" -B "$WORK/headers/build" -G Ninja \
    -DCMAKE_INSTALL_PREFIX="$STAGE" >/dev/null
cmake --install "$WORK/headers/build" >/dev/null

# 2. Vulkan-Loader: link-time libvulkan.so for the ggml-vulkan module. WSI
#    support off — compute-only linking, no X11/Wayland build deps. At
#    runtime the wheel resolves the *system* loader (auditwheel --exclude
#    libvulkan.so.1), so this library never ships.
git clone --quiet --depth 1 --branch "$VULKAN_SDK_TAG" \
    https://github.com/KhronosGroup/Vulkan-Loader "$WORK/loader"
cmake -S "$WORK/loader" -B "$WORK/loader/build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$STAGE" \
    -DCMAKE_PREFIX_PATH="$STAGE" \
    -DBUILD_TESTS=OFF \
    -DBUILD_WSI_XCB_SUPPORT=OFF \
    -DBUILD_WSI_XLIB_SUPPORT=OFF \
    -DBUILD_WSI_WAYLAND_SUPPORT=OFF >/dev/null
cmake --build "$WORK/loader/build" -j >/dev/null
cmake --install "$WORK/loader/build" >/dev/null

# 3. shaderc -> glslc. git-sync-deps pins glslang/spirv-tools/spirv-headers to
#    the revisions in shaderc's DEPS file, so the tag fully determines the
#    toolchain. Only the glslc executable is needed.
git clone --quiet --depth 1 --branch "$SHADERC_TAG" \
    https://github.com/google/shaderc "$WORK/shaderc"
(cd "$WORK/shaderc" && ./utils/git-sync-deps >/dev/null 2>&1)
cmake -S "$WORK/shaderc" -B "$WORK/shaderc/build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DSHADERC_SKIP_TESTS=ON \
    -DSHADERC_SKIP_EXAMPLES=ON \
    -DSHADERC_SKIP_COPYRIGHT_CHECK=ON >/dev/null
cmake --build "$WORK/shaderc/build" --target glslc_exe -j >/dev/null
install -D -m 0755 "$WORK/shaderc/build/glslc/glslc" "$STAGE/bin/glslc"

# Install into the build prefix and persist to the cache volume.
cp -a "$STAGE/." "$PREFIX/"
if [[ -n "$CACHE_ROOT" ]]; then
    mkdir -p "$CACHE_ROOT/$STAMP"
    cp -a "$STAGE/." "$CACHE_ROOT/$STAMP/"
fi

glslc --version
echo "vulkan-toolchain: done ($STAMP)"

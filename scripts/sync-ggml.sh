#!/usr/bin/env bash
#
# sync-ggml.sh — re-vendor ggml/ from upstream ggml-org/ggml at a given ref.
#
# ggml is vendored (not a submodule): ggml/ is the upstream tracked tree at the
# SHA recorded in ggml/UPSTREAM, plus the ordered downstream patches in
# patches/ggml/. The SHA and patch directory are the complete vendor recipe.
#
# What it does:
#   1. Fetches the upstream tracked tree at <ref> (a SHA, tag, or branch).
#   2. Materializes it via `git archive` (tracked files only — no .git, no
#      build cruft), minus the paths in EXCLUDES below.
#   3. Applies patches/ggml/*.patch in filename order.
#   4. Swaps the result into ggml/ and rewrites ggml/UPSTREAM.
#
# Upstream examples and tests are kept (they are not built — TRANSCRIBE_*/
# GGML_BUILD_* leave them off — but keeping them makes the vendor diff
# reviewable). Only .github/ is dropped: upstream CI is irrelevant here.
#
# Usage:
#   scripts/sync-ggml.sh                 # re-vendor the CURRENT pinned SHA (repair / verify)
#   scripts/sync-ggml.sh master          # bump to upstream default-branch HEAD
#   scripts/sync-ggml.sh <sha|tag>       # pin to a specific commit or release tag
#   scripts/sync-ggml.sh master --dry-run   # show what would change, write nothing
#
# Flags:
#   --dry-run        Resolve the ref and report the file-level diff; do not touch ggml/.
#   --force          Proceed even if ggml/ has uncommitted local changes (default: abort,
#                    so an accidental hand-edit is never silently clobbered).
#   --repo <url>     Override the upstream URL (default: the repo: line in ggml/UPSTREAM).
#
# Exit-code driven, non-interactive. After a real sync, build and let native-ci
# (path filter ggml/**) certify the C/C++ contracts the bindings depend on:
#   cmake --build build --target transcribe-cli

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GGML_DIR="${REPO_ROOT}/ggml"
UPSTREAM_FILE="${GGML_DIR}/UPSTREAM"
PATCH_DIR="${REPO_ROOT}/patches/ggml"

# Upstream paths to drop from the snapshot (relative to the ggml tree root).
EXCLUDES=( ".github" )

shopt -s nullglob
PATCHES=( "${PATCH_DIR}"/*.patch )
shopt -u nullglob

# ---- parse args -------------------------------------------------------------
REF=""
REPO=""
DRY_RUN=0
FORCE=0

die() { echo "sync-ggml: $*" >&2; exit 1; }

while [ "$#" -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --force)   FORCE=1; shift ;;
        --repo)    REPO="${2:-}"; [ -n "$REPO" ] || die "--repo needs a URL"; shift 2 ;;
        -h|--help) sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        --*)       die "unknown flag: $1" ;;
        *)         [ -z "$REF" ] || die "more than one ref given ($REF, $1)"; REF="$1"; shift ;;
    esac
done

[ -f "$UPSTREAM_FILE" ] || die "missing $UPSTREAM_FILE — run from a checkout with vendored ggml"

# ---- read current pin -------------------------------------------------------
CUR_REPO="$(sed -n 's/^repo:[[:space:]]*//p' "$UPSTREAM_FILE" | head -1)"
CUR_SHA="$(sed -n 's/^sha:[[:space:]]*//p'  "$UPSTREAM_FILE" | head -1)"
[ -n "$CUR_REPO" ] || die "no 'repo:' line in $UPSTREAM_FILE"
[ -n "$CUR_SHA"  ] || die "no 'sha:' line in $UPSTREAM_FILE"

REPO="${REPO:-$CUR_REPO}"
REF="${REF:-$CUR_SHA}"   # default: re-vendor the current pin (idempotent repair)

command -v git >/dev/null || die "git not found"
command -v tar >/dev/null || die "tar not found"

# ---- guard against clobbering local edits -----------------------------------
if [ "$DRY_RUN" -eq 0 ] && [ "$FORCE" -eq 0 ]; then
    if [ -n "$(git -C "$REPO_ROOT" status --porcelain -- ggml 2>/dev/null)" ]; then
        die "ggml/ has uncommitted changes. Commit/stash them, or pass --force to overwrite."
    fi
fi

# ---- scratch dirs (self-cleaning) -------------------------------------------
CLONE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/sync-ggml.XXXXXX")"
STAGE_DIR="${REPO_ROOT}/ggml.sync-stage.$$"   # sibling of ggml/ → same FS → atomic mv
cleanup() { rm -rf "$CLONE_DIR" "$STAGE_DIR"; }
trap cleanup EXIT

# ---- fetch just the requested commit ----------------------------------------
echo "sync-ggml: fetching $REF from $REPO ..."
git init  --quiet "$CLONE_DIR"
git -C "$CLONE_DIR" remote add origin "$REPO"
# `git fetch <ref>` resolves a SHA, tag, or branch in one shot; --depth 1 keeps
# it to a single commit's complete tree+blobs (GitHub allows arbitrary-SHA fetch).
git -C "$CLONE_DIR" fetch --quiet --depth 1 origin "$REF" \
    || die "could not fetch '$REF' (try a branch/tag, or check the SHA is reachable)"
# Peel to the commit: for an annotated tag, FETCH_HEAD is the tag *object* —
# ^{commit} resolves it to the commit SHA (a no-op for branch/commit refs), so
# UPSTREAM records a real commit, matching the existing convention.
RESOLVED="$(git -C "$CLONE_DIR" rev-parse "FETCH_HEAD^{commit}")"

# ---- materialize tracked tree, minus excludes -------------------------------
mkdir -p "$STAGE_DIR"
git -C "$CLONE_DIR" archive --format=tar "$RESOLVED" | tar -x -C "$STAGE_DIR"
for ex in "${EXCLUDES[@]}"; do
    rm -rf "${STAGE_DIR:?}/${ex}"
done

# ---- apply downstream patches ----------------------------------------------
STAGE_NAME="$(basename "$STAGE_DIR")"
for patch in "${PATCHES[@]}"; do
    echo "sync-ggml: applying patches/ggml/$(basename "$patch")"
    git -C "$REPO_ROOT" apply --check --directory="$STAGE_NAME" "$patch" \
        || die "patch does not apply: patches/ggml/$(basename "$patch")"
    git -C "$REPO_ROOT" apply --directory="$STAGE_NAME" "$patch"
done

# ---- regenerate UPSTREAM ----------------------------------------------------
{
    cat <<EOF
repo: ${REPO}
sha:  ${RESOLVED}
patches:
EOF
    if [ "${#PATCHES[@]}" -eq 0 ]; then
        echo "  (none)"
    else
        for patch in "${PATCHES[@]}"; do
            echo "  patches/ggml/$(basename "$patch")"
        done
    fi
    cat <<'EOF'

This directory is generated from the upstream ggml tree at the SHA above, minus
.github/, with the listed downstream patches applied in order. Do not edit it by
hand. Run scripts/sync-ggml.sh <ref> from the repo root to reproduce or upgrade
it; the script rewrites this file.
EOF
} > "${STAGE_DIR}/UPSTREAM"

# git archive preserves upstream commit times, which can be older than objects
# in an existing build tree. Refresh source mtimes so an incremental build after
# an upgrade cannot accidentally link stale ggml objects.
find "$STAGE_DIR" -type f -exec touch {} +

# ---- dry-run: report and stop ----------------------------------------------
if [ "$DRY_RUN" -eq 1 ]; then
    echo "[dry-run] would re-vendor: ${CUR_SHA:0:12} -> ${RESOLVED:0:12}"
    if diff -rq "$GGML_DIR" "$STAGE_DIR" >/tmp/sync-ggml.diff.$$ 2>/dev/null; then
        echo "[dry-run] no differences — ggml/ already matches this ref."
    else
        echo "[dry-run] path-level differences vs current ggml/:"
        sed "s#${STAGE_DIR}#ggml(new)#g; s#${GGML_DIR}#ggml(cur)#g" /tmp/sync-ggml.diff.$$ | sed 's/^/  /'
    fi
    rm -f /tmp/sync-ggml.diff.$$
    echo "[dry-run] nothing written."
    exit 0
fi

# ---- swap into place --------------------------------------------------------
rm -rf "$GGML_DIR"
mv "$STAGE_DIR" "$GGML_DIR"

CHANGED="$(git -C "$REPO_ROOT" status --porcelain -- ggml 2>/dev/null | wc -l | tr -d ' ')"
echo "sync-ggml: ggml re-vendored ${CUR_SHA:0:12} -> ${RESOLVED:0:12}"
echo "sync-ggml: ${CHANGED} path(s) changed under ggml/  (review: git diff --stat -- ggml)"
echo "sync-ggml: next — cmake --build build --target transcribe-cli, then push to run native-ci (ggml/**)"

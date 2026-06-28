#!/usr/bin/env bash
# clang-format.sh - canonical formatter for transcribe.cpp's own C/C++.
#
# Single source of truth for three things, so local dev and CI can never drift:
#   1. the pinned clang-format version (fetched via uvx, not the system binary);
#   2. which files are "ours" vs vendored / verbatim-upstream;
#   3. the exact check the CI gate runs.
#
# Usage:
#   scripts/ci/clang-format.sh [--fix]            format our tree in place (default)
#   scripts/ci/clang-format.sh --check            fail if our whole tree is unformatted
#   scripts/ci/clang-format.sh --check-diff [REF] fail only on files changed vs REF
#                                                 (default origin/main)
set -euo pipefail

# Pinned clang-format. Bump here and reformat the tree in the SAME commit.
# Fetched through uvx so the result never depends on a locally installed binary.
CF_VERSION="22.1.5"

# Paths we never format: vendored upstream trees, and files copied verbatim from
# upstream (kept byte-identical so future re-syncs stay clean).
EXCLUDE_RE='^(ggml/|src/third_party/|src/transcribe-unicode-data\.cpp$)'

cd "$(git rev-parse --show-toplevel)"

CF=(uvx "clang-format@${CF_VERSION}")
EXTS=('*.c' '*.cc' '*.cpp' '*.cxx' '*.h' '*.hh' '*.hpp' '*.hxx')

# Our tracked C/C++, vendored/verbatim paths removed.
ours_all() {
    git ls-files -- "${EXTS[@]}" | grep -vE "$EXCLUDE_RE" || true
}

# Same, restricted to files added/copied/modified/renamed vs a ref's merge base.
ours_changed() {
    local base
    base="$(git merge-base "$1" HEAD)"
    git diff --name-only --diff-filter=ACMR "$base" -- "${EXTS[@]}" \
        | grep -vE "$EXCLUDE_RE" || true
}

# Read a newline list from stdin into the global SELECTED array (bash 3.2 safe).
SELECTED=()
collect() {
    SELECTED=()
    local line
    while IFS= read -r line; do
        [ -n "$line" ] && SELECTED+=("$line")
    done
}

check() {
    if [ "${#SELECTED[@]}" -eq 0 ]; then
        echo "clang-format: no files to check."
        return 0
    fi
    echo "clang-format ${CF_VERSION}: checking ${#SELECTED[@]} file(s)."
    "${CF[@]}" --dry-run --Werror "${SELECTED[@]}"
}

fix() {
    if [ "${#SELECTED[@]}" -eq 0 ]; then
        echo "clang-format: no files to format."
        return 0
    fi
    echo "clang-format ${CF_VERSION}: formatting ${#SELECTED[@]} file(s)."
    # clang-format -i is not always idempotent: trailing-comment alignment can
    # need a second pass to settle. Iterate until the result already satisfies
    # --check, so what we write can never fail the CI gate.
    local pass
    for pass in 1 2 3; do
        "${CF[@]}" -i "${SELECTED[@]}"
        if "${CF[@]}" --dry-run --Werror "${SELECTED[@]}" >/dev/null 2>&1; then
            return 0
        fi
    done
    echo "error: clang-format did not converge after 3 passes." >&2
    return 1
}

case "${1:---fix}" in
    --fix)
        collect < <(ours_all)
        fix
        ;;
    --check)
        collect < <(ours_all)
        check
        ;;
    --check-diff)
        collect < <(ours_changed "${2:-origin/main}")
        check
        ;;
    -h | --help)
        echo "usage: $0 [--fix | --check | --check-diff [REF]]"
        ;;
    *)
        echo "unknown option: $1" >&2
        exit 2
        ;;
esac

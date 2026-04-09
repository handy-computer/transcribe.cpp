#!/usr/bin/env bash
# Download and extract LibriSpeech test-clean (~350 MB).
#
# After running:
#   samples/wer/raw/LibriSpeech/test-clean/<spk>/<chap>/*.flac + *.trans.txt
#
# Idempotent: skips download if archive exists, skips extract if dir exists.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RAW_DIR="${REPO_ROOT}/samples/wer/raw"
ARCHIVE="${RAW_DIR}/test-clean.tar.gz"
EXTRACT_DIR="${RAW_DIR}/LibriSpeech/test-clean"
URL="https://www.openslr.org/resources/12/test-clean.tar.gz"

mkdir -p "${RAW_DIR}"

if [ -d "${EXTRACT_DIR}" ]; then
    echo "Already extracted: ${EXTRACT_DIR}"
    echo "$(find "${EXTRACT_DIR}" -name '*.flac' | wc -l | tr -d ' ') flac files"
    exit 0
fi

if [ ! -f "${ARCHIVE}" ]; then
    echo "Downloading test-clean.tar.gz (~350 MB)..."
    curl -L --progress-bar "${URL}" -o "${ARCHIVE}"
else
    echo "Archive already exists: ${ARCHIVE}"
fi

echo "Extracting..."
tar -xzf "${ARCHIVE}" -C "${RAW_DIR}"

n_flac=$(find "${EXTRACT_DIR}" -name '*.flac' | wc -l | tr -d ' ')
echo "Done. ${n_flac} flac files in ${EXTRACT_DIR}"

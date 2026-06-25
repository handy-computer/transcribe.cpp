#!/usr/bin/env bash
# Quiet-box benchmark: run with the Claude agent stopped so all 4 cores
# are free. Results land in /tmp/quiet_bench.txt.
#
#   bash scripts/bench/quiet_4t.sh
#
# Takes ~6 minutes. Each section logs SoC temperature before/after so
# thermal state is interpretable.
set -u
cd "$(dirname "$0")/../.."
OUT=/tmp/quiet_bench.txt
MODEL=models/nemotron-3.5-asr-streaming-0.6b/nemotron-3.5-asr-streaming-0.6b-Q4_0.gguf
MODEL_Q8=models/nemotron-3.5-asr-streaming-0.6b/nemotron-3.5-asr-streaming-0.6b-Q8_0.gguf

temp() { cat /sys/class/thermal/thermal_zone0/temp; }

{
  echo "=== quiet-box bench $(date) ==="
  echo "idle temp: $(temp)"

  if [ -x /tmp/microbench/gemm_bench ]; then
    for t in 1 2 3 4; do
      echo "--- gemm_bench ${t}T (temp $(temp)) ---"
      /tmp/microbench/gemm_bench "$t" 2>/dev/null
    done
  fi

  for t in 3 4; do
    echo "--- streaming Q4_0 ${t}T (temp $(temp)) ---"
    TRANSCRIBE_PROFILE_STREAM=1 build/bin/transcribe-cli -m "$MODEL" \
      --language en-US --stream-chunk-ms 1120 --stream-att-right 13 \
      --threads "$t" samples/jfk.wav 2>&1 |
      grep -E "per-chunk mean|realtime|text:"
    echo "post temp: $(temp)"
  done

  echo "--- streaming Q4_0 4T chunk=560 R=6 (480ms lookahead) (temp $(temp)) ---"
  TRANSCRIBE_PROFILE_STREAM=1 build/bin/transcribe-cli -m "$MODEL" \
    --language en-US --stream-chunk-ms 560 --stream-att-right 6 \
    --threads 4 samples/jfk.wav 2>&1 |
    grep -E "per-chunk mean|realtime"
  echo "post temp: $(temp)"

  echo "--- streaming Q8_0 4T (temp $(temp)) ---"
  TRANSCRIBE_PROFILE_STREAM=1 build/bin/transcribe-cli -m "$MODEL_Q8" \
    --language en-US --stream-chunk-ms 1120 --stream-att-right 13 \
    --threads 4 samples/jfk.wav 2>&1 |
    grep -E "per-chunk mean|realtime"
  echo "post temp: $(temp)"

  echo "--- offline Q4_0 4T (temp $(temp)) ---"
  build/bin/transcribe-cli -m "$MODEL" --language en-US --threads 4 \
    samples/jfk.wav 2>/dev/null | grep -E "realtime"
  echo "final temp: $(temp)"
} | tee "$OUT"
echo "results saved to $OUT"

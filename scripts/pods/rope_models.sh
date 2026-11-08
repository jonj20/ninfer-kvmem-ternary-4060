#!/usr/bin/env bash
# The artifacts the rope-scaling perplexity runs score, one directory each: Qwen3.8-27B (IQ3_S)
# and Qwen3.8-Flash-Next (IQ3_S) with its n-gram table.
# Arguments select the models (qwen38-27b, flash-next); none selects both.
set -Eeuo pipefail
root=/workspace/ninfer-work
hf="$root/py311/bin/hf"
fetch() { # repo file directory
    mkdir -p "$root/models/$3"
    [ -f "$root/models/$3/$2" ] || "$hf" download "$1" "$2" --local-dir "$root/models/$3"
    ls -l "$root/models/$3/$2"
}
wanted=" ${*:-qwen38-27b flash-next} "
case "$wanted" in *" qwen38-27b "*)
    fetch WaveCut/Qwen3.8-27B-GSQ-RCO-IQ3_S-NInfer-v3 Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer qwen38-27b ;;
esac
case "$wanted" in *" flash-next "*)
    fetch WaveCut/Qwen3.8-Flash-Next-ngram-table-NInfer-v3 Qwen3.8-Flash-Next-ngram-table-IQ4_NL-ninfer-v3.ninfer flash-next
    fetch WaveCut/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-NInfer-v3 Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-ninfer-v3.ninfer flash-next ;;
esac

#!/usr/bin/env bash
# The rope-scaling tests on a GPU: the rope Op and its frequency schedules, Flash-Next's indexer
# under YaRN, the runtime resolver and the server_start record.
set -Eeuo pipefail
root=/workspace/ninfer-work
build="$root/build"
ctest --test-dir "$build" --output-on-failure -j2 \
    -R 'ninfer_(rope|qsa_indexer|rope_scaling|request_log|serve_options)_test'

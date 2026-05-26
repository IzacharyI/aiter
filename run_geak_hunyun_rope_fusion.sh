#!/usr/bin/env bash
set -euo pipefail

GEAK_REPO="${GEAK_REPO:-/opt/GEAK}"
REPO="${REPO:-/opt/hunyunFusionTicket/aiter}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
NUM_PARALLEL="${NUM_PARALLEL:-4}"
MODEL="${MODEL:-claude-sonnet-4.6}"

cd "$GEAK_REPO"

geak --repo "$REPO" \
  --model "$MODEL" \
  --kernel-url "$REPO/op_tests/bench_rope_norm_store_kv.py" \
  --test-command 'python3 ./op_tests/run_hunyun_rope_geak_eval.py' \
  --task "$REPO/geak_hunyun_rope_fusion_task.md" \
  --num-parallel "$NUM_PARALLEL" \
  --gpu-ids "$GPU_IDS" \
  --yolo --exit-immediately

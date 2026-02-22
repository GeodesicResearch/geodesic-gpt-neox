#!/bin/bash
# Submit all midtraining unlearning experiment configs
# Usage: bash submit_all.sh [post_training|midtraining|all]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_DIR="$SCRIPT_DIR/configs/midtraining_unlearning"

submit_post_training() {
    echo "=== Submitting Post-Training GDiff Runs ==="
    for model in deep_ignorance sfm_baseline; do
        for alpha in 10 20 40 80; do
            config="$CONFIG_DIR/post_training/$model/gdiff_post_training_alpha_${alpha}.yml"
            echo "Submitting $config ..."
            sbatch "$SCRIPT_DIR/pretrain_neox.sbatch" "$config"
        done
    done
}

submit_midtraining() {
    echo "=== Submitting Midtraining GDiff Runs ==="
    for model in deep_ignorance sfm_baseline; do
        for N in 2 4 8 16 32 128 256; do
            config="$CONFIG_DIR/midtraining/$model/gdiff_midtraining_N_${N}.yml"
            echo "Submitting $config ..."
            sbatch "$SCRIPT_DIR/pretrain_neox.sbatch" "$config"
        done
    done
}

case "${1:-all}" in
    post_training) submit_post_training ;;
    midtraining)   submit_midtraining ;;
    all)           submit_post_training; submit_midtraining ;;
    *)             echo "Usage: $0 [post_training|midtraining|all]"; exit 1 ;;
esac

echo ""
echo "=== Monitor with: ==="
echo "squeue -u \$USER | grep neox"

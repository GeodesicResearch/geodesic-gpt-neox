#!/bin/bash
# Submit all 8 post-training GDiff runs
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_DIR="$(pwd)/configs/midtraining_unlearning/post_training"

for alpha in 10 20 40 80; do
  echo "Submitting deep-ignorance alpha=$alpha..."
  sbatch pretrain_neox.sbatch "$CONFIG_DIR/deep_ignorance/gdiff_post_training_alpha_${alpha}.yml"
  echo "Submitting sfm_baseline alpha=$alpha..."
  sbatch pretrain_neox.sbatch "$CONFIG_DIR/sfm_baseline/gdiff_post_training_alpha_${alpha}.yml"
done

echo "All 8 post-training GDiff jobs submitted."
echo "Monitor: squeue -u \$USER | grep neox"

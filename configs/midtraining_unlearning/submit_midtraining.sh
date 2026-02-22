#!/bin/bash
# Submit all 14 midtraining GDiff runs
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_DIR="$(pwd)/configs/midtraining_unlearning/midtraining"

for N in 2 4 8 16 32 128 256; do
  echo "Submitting deep-ignorance N=$N..."
  sbatch pretrain_neox.sbatch "$CONFIG_DIR/deep_ignorance/gdiff_midtraining_N_${N}.yml"
  echo "Submitting sfm_baseline N=$N..."
  sbatch pretrain_neox.sbatch "$CONFIG_DIR/sfm_baseline/gdiff_midtraining_N_${N}.yml"
done

echo "All 14 midtraining GDiff jobs submitted."
echo "Monitor: squeue -u \$USER | grep neox"

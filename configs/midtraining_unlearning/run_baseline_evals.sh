#!/bin/bash
# Run baseline evaluations on both unmodified HF models
# Usage: bash run_baseline_evals.sh

set -euo pipefail

SCRIPT_DIR="/home/a5k/kyleobrien.a5k/geodesic-gpt-neox"
INCLUDE_PATH="$SCRIPT_DIR/lm_eval_tasks/deep_ignorance"
# Expand mmlu_bio to individual tasks since standalone lm_eval doesn't know our custom groups
TASKS="wmdp_bio_aisi_robust,wmdp_bio_cloze_verified,mmlu_college_biology,mmlu_high_school_biology"

echo "=== Submitting baseline evaluations ==="

# deep-ignorance baseline
echo "Submitting deep-ignorance baseline eval..."
JOB1=$(sbatch --parsable --time=04:00:00 "$SCRIPT_DIR/run_on_compute.sbatch" \
    uv run lm_eval \
    --model hf \
    --model_args pretrained=EleutherAI/deep-ignorance-unfiltered,dtype=bfloat16 \
    --tasks "$TASKS" \
    --include_path "$INCLUDE_PATH" \
    --batch_size auto)
echo "  Job ID: $JOB1"
echo "  Log: /projects/a5k/public/logs/neox-training/run_on_compute_${JOB1}.out"

# sfm_baseline baseline
echo "Submitting sfm_baseline baseline eval..."
JOB2=$(sbatch --parsable --time=04:00:00 "$SCRIPT_DIR/run_on_compute.sbatch" \
    uv run lm_eval \
    --model hf \
    --model_args pretrained=geodesic-research/sfm_baseline_unfiltered_base,dtype=bfloat16 \
    --tasks "$TASKS" \
    --include_path "$INCLUDE_PATH" \
    --batch_size auto)
echo "  Job ID: $JOB2"
echo "  Log: /projects/a5k/public/logs/neox-training/run_on_compute_${JOB2}.out"

echo ""
echo "=== Monitor with: ==="
echo "tail -f /projects/a5k/public/logs/neox-training/run_on_compute_${JOB1}.out"
echo "tail -f /projects/a5k/public/logs/neox-training/run_on_compute_${JOB2}.out"

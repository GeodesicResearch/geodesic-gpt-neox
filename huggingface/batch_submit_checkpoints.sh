#!/bin/bash

# Batch submission script for uploading and evaluating all checkpoints in an experiment
# Submits one SLURM job per checkpoint to upload NeoX format, convert to HuggingFace, and evaluate
#
# Uses SLURM singleton dependencies to ensure only one checkpoint conversion runs at a time
# per experiment, preventing HuggingFace rate limiting. Different experiments can still run
# checkpoints in parallel.
#
# Checkpoints are processed in DESCENDING order (most recent first), so the latest
# checkpoints are converted and uploaded first.
#
# Usage: ./batch_submit_checkpoints.sh <experiment_name> [hf_org] [options]
# Options:
#   --delay-between-jobs <seconds>  : Delay between job submissions (default: 0)
#   --upload-delay <seconds|random> : Initial delay before uploads - use 'random' for 0-12hr jitter (default: 0)
#   --max-retries <number>          : Maximum retry attempts on 429 errors (default: 5)
#   --retry-min-wait <seconds>      : Minimum wait for exponential backoff (default: 60)
#   --retry-max-wait <seconds>      : Maximum wait for exponential backoff (default: 600)
#   --checkpoints-base-dir <path>   : Base directory for checkpoints (default: /projects/a5k/public/checkpoints/)
#   --upload-neox                   : Upload raw NeoX checkpoints to HuggingFace (disabled by default)
#   --upload-neox-only              : Upload ONLY NeoX checkpoints, skip HF conversion and evaluation
#   --no-singleton                  : Disable singleton dependency (run all checkpoints in parallel)
#   --skip-eval                     : Skip evaluation (default)
#   --eval                          : Run evaluation on checkpoints
#   --eval-tasks <tasks>            : Comma-separated evaluation tasks (only used with --eval)
#   --task-include-path <path>      : Path for custom evaluation tasks (only used with --eval)
#   --wandb-entity <entity>         : W&B entity for logging (only used with --eval)
#   --wandb-project <project>       : W&B project for logging (only used with --eval)

EXPERIMENT_NAME=$1
HF_ORG=${2:-"geodesic-research"}

# Shift past positional arguments
shift 2

# Default values for optional parameters
DELAY_BETWEEN_JOBS=0
UPLOAD_DELAY=0
MAX_RETRIES=20
RETRY_MIN_WAIT=120
RETRY_MAX_WAIT=3600
CHECKPOINTS_BASE_DIR="/projects/a5k/public/checkpoints/sf_model_organisms/"
UPLOAD_NEOX_ONLY=""
UPLOAD_NEOX=""
USE_SINGLETON=true
SKIP_EVAL="--skip-eval"
EVAL_TASKS=""
TASK_INCLUDE_PATH=""
WANDB_ENTITY=""
WANDB_PROJECT=""

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --delay-between-jobs)
            DELAY_BETWEEN_JOBS="$2"
            shift 2
            ;;
        --upload-delay)
            UPLOAD_DELAY="$2"
            shift 2
            ;;
        --max-retries)
            MAX_RETRIES="$2"
            shift 2
            ;;
        --retry-min-wait)
            RETRY_MIN_WAIT="$2"
            shift 2
            ;;
        --retry-max-wait)
            RETRY_MAX_WAIT="$2"
            shift 2
            ;;
        --checkpoints-base-dir)
            CHECKPOINTS_BASE_DIR="$2"
            shift 2
            ;;
        --upload-neox-only)
            UPLOAD_NEOX_ONLY="--upload-neox-only"
            shift 1
            ;;
        --upload-neox)
            UPLOAD_NEOX="1"
            shift 1
            ;;
        --no-singleton)
            USE_SINGLETON=false
            shift 1
            ;;
        --skip-eval)
            SKIP_EVAL="--skip-eval"
            shift 1
            ;;
        --eval)
            SKIP_EVAL=""
            shift 1
            ;;
        --eval-tasks)
            EVAL_TASKS="$2"
            shift 2
            ;;
        --task-include-path)
            TASK_INCLUDE_PATH="$2"
            shift 2
            ;;
        --wandb-entity)
            WANDB_ENTITY="$2"
            shift 2
            ;;
        --wandb-project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$EXPERIMENT_NAME" ]; then
    echo "Error: Missing experiment name"
    echo "Usage: ./batch_submit_checkpoints.sh <experiment_name> [hf_org] [options]"
    echo "Options:"
    echo "  --delay-between-jobs <seconds>  : Delay between job submissions (default: 0)"
    echo "  --upload-delay <seconds|random> : Initial delay before uploads - 'random' for 0-12hr jitter (default: 0)"
    echo "  --max-retries <number>          : Maximum retry attempts (default: 5)"
    echo "  --retry-min-wait <seconds>      : Min wait for backoff (default: 60)"
    echo "  --retry-max-wait <seconds>      : Max wait for backoff (default: 600)"
    echo "  --checkpoints-base-dir <path>   : Base directory for checkpoints (default: /projects/a5k/public/checkpoints/)"
    echo "  --upload-neox                   : Upload raw NeoX checkpoints to HuggingFace (disabled by default)"
    echo "  --upload-neox-only              : Upload ONLY NeoX checkpoints, skip HF conversion and evaluation"
    echo "  --no-singleton                  : Disable singleton dependency (run all checkpoints in parallel)"
    echo "  --skip-eval                     : Skip evaluation (default)"
    echo "  --eval                          : Run evaluation on checkpoints"
    echo "  --eval-tasks <tasks>            : Comma-separated evaluation tasks (only with --eval)"
    echo "  --task-include-path <path>      : Path for custom evaluation tasks (only with --eval)"
    echo "  --wandb-entity <entity>         : W&B entity for logging (only with --eval)"
    echo "  --wandb-project <project>       : W&B project for logging (only with --eval)"
    exit 1
fi

CHECKPOINT_DIR="${CHECKPOINTS_BASE_DIR}${EXPERIMENT_NAME}"

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Error: Checkpoint directory does not exist: $CHECKPOINT_DIR"
    exit 1
fi

echo "========================================="
echo "CHECKPOINT UPLOAD AND EVALUATION BATCH"
echo "========================================="
echo "Experiment: $EXPERIMENT_NAME"
echo "HuggingFace org: $HF_ORG"
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo "Timestamp: $(date)"
if [ $DELAY_BETWEEN_JOBS -gt 0 ] || [ "$UPLOAD_DELAY" = "random" ] || [ $UPLOAD_DELAY -gt 0 ] 2>/dev/null; then
    echo ""
    echo "Delay configuration:"
    echo "  Between jobs: ${DELAY_BETWEEN_JOBS}s"
    if [ "$UPLOAD_DELAY" = "random" ]; then
        echo "  Upload delay: random (0-12 hours per job)"
    else
        echo "  Upload delay: ${UPLOAD_DELAY}s"
    fi
    echo "  Retry: max=$MAX_RETRIES, min=${RETRY_MIN_WAIT}s, max=${RETRY_MAX_WAIT}s"
fi
echo "========================================="

# Get all checkpoint directories in descending order (most recent first)
# This ensures recent checkpoints are processed first with singleton dependencies
CHECKPOINTS=$(ls -d $CHECKPOINT_DIR/global_step* 2>/dev/null | sort -Vr)

if [ -z "$CHECKPOINTS" ]; then
    echo "No checkpoints found in $CHECKPOINT_DIR"
    exit 1
fi

# Count total checkpoints
TOTAL=$(echo "$CHECKPOINTS" | wc -l)
echo ""
echo "CHECKPOINT DISCOVERY:"
echo "Found $TOTAL checkpoints to process"
echo "Processing order: DESCENDING (most recent first)"
echo ""

# Show checkpoint list for verification
echo "Checkpoint list (most recent first):"
PREVIEW_COUNT=0
for CHECKPOINT_PATH in $CHECKPOINTS; do
    PREVIEW_COUNT=$((PREVIEW_COUNT + 1))
    CHECKPOINT_NUM=$(basename $CHECKPOINT_PATH | sed 's/global_step//')
    echo "  [$PREVIEW_COUNT] global_step$CHECKPOINT_NUM"
    # Show first 5 and last 5 if more than 10 total
    if [ $PREVIEW_COUNT -eq 5 ] && [ $TOTAL -gt 10 ]; then
        REMAINING=$((TOTAL - 10))
        if [ $REMAINING -gt 0 ]; then
            echo "  ... [$REMAINING more checkpoints] ..."
        fi
        # Skip to last 5
        break
    fi
done

# Show last 5 if we have more than 10 total
if [ $TOTAL -gt 10 ]; then
    LAST_5_START=$((TOTAL - 4))
    CURRENT_COUNT=0
    for CHECKPOINT_PATH in $CHECKPOINTS; do
        CURRENT_COUNT=$((CURRENT_COUNT + 1))
        if [ $CURRENT_COUNT -ge $LAST_5_START ]; then
            CHECKPOINT_NUM=$(basename $CHECKPOINT_PATH | sed 's/global_step//')
            echo "  [$CURRENT_COUNT] global_step$CHECKPOINT_NUM"
        fi
    done
fi
echo ""

# Submit job for each checkpoint
echo "========================================="
echo "STARTING JOB SUBMISSIONS"
echo "========================================="
echo ""

# Get the directory where this script is located (for sbatch file path)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

COUNT=0
SUCCESSFUL_SUBMISSIONS=0
FAILED_SUBMISSIONS=0

for CHECKPOINT_PATH in $CHECKPOINTS; do
    COUNT=$((COUNT + 1))
    CHECKPOINT_NUM=$(basename $CHECKPOINT_PATH | sed 's/global_step//')

    echo "----------------------------------------"
    echo "Processing checkpoint $COUNT of $TOTAL"
    echo "Checkpoint: global_step$CHECKPOINT_NUM"
    echo "Path: $CHECKPOINT_PATH"
    echo "Time: $(date '+%H:%M:%S')"

    # Calculate upload delay for this job
    # For 'random', pass through to each job to generate its own random delay
    # For numeric delays, use incremental staggering (each job gets base + 30s * job_number)
    if [ "$UPLOAD_DELAY" = "random" ]; then
        JOB_UPLOAD_DELAY="random"
        echo "Upload delay for this job: random (0-12 hours, generated per job)"
    elif [ $UPLOAD_DELAY -gt 0 ] 2>/dev/null; then
        JOB_UPLOAD_DELAY=$((UPLOAD_DELAY + (COUNT - 1) * 30))
        echo "Upload delay for this job: ${JOB_UPLOAD_DELAY}s (staggered)"
    else
        JOB_UPLOAD_DELAY=0
        echo "Upload delay: None"
    fi

    # Build the sbatch command with job name and optional singleton dependency
    # Singleton ensures only one checkpoint conversion runs at a time per experiment
    JOB_NAME="convert_${EXPERIMENT_NAME}"
    if [ "$USE_SINGLETON" = true ]; then
        SBATCH_CMD="sbatch --parsable --job-name=$JOB_NAME --dependency=singleton"
        echo "SLURM command: $SBATCH_CMD"
        echo "Job name: $JOB_NAME (singleton dependency ensures sequential execution per experiment)"
    else
        SBATCH_CMD="sbatch --parsable --job-name=$JOB_NAME"
        echo "SLURM command: $SBATCH_CMD"
        echo "Job name: $JOB_NAME (NO singleton - jobs run in parallel)"
    fi

    # Create environment variables for the job
    # Note: SLURM uses commas to separate env vars, so we need to escape commas in values with %2C
    EVAL_TASKS_ESCAPED="${EVAL_TASKS//,/%2C}"

    EXPORT_VARS="--export=ALL"
    EXPORT_VARS="${EXPORT_VARS},UPLOAD_DELAY=$JOB_UPLOAD_DELAY"
    EXPORT_VARS="${EXPORT_VARS},MAX_RETRIES=$MAX_RETRIES"
    EXPORT_VARS="${EXPORT_VARS},RETRY_MIN_WAIT=$RETRY_MIN_WAIT"
    EXPORT_VARS="${EXPORT_VARS},RETRY_MAX_WAIT=$RETRY_MAX_WAIT"
    EXPORT_VARS="${EXPORT_VARS},CHECKPOINTS_BASE_DIR=$CHECKPOINTS_BASE_DIR"

    # Add NeoX-only mode flag if set
    if [ -n "$UPLOAD_NEOX_ONLY" ]; then
        EXPORT_VARS="${EXPORT_VARS},UPLOAD_NEOX_ONLY=1"
    fi

    # Add NeoX upload flag if set
    if [ -n "$UPLOAD_NEOX" ]; then
        EXPORT_VARS="${EXPORT_VARS},UPLOAD_NEOX=1"
    fi

    # Add evaluation-related environment variables if they are set
    if [ -n "$EVAL_TASKS" ]; then
        EXPORT_VARS="${EXPORT_VARS},EVAL_TASKS=$EVAL_TASKS_ESCAPED"
    fi
    if [ -n "$TASK_INCLUDE_PATH" ]; then
        EXPORT_VARS="${EXPORT_VARS},TASK_INCLUDE_PATH=$TASK_INCLUDE_PATH"
    fi
    if [ -n "$WANDB_ENTITY" ]; then
        EXPORT_VARS="${EXPORT_VARS},WANDB_ENTITY=$WANDB_ENTITY"
    fi
    if [ -n "$WANDB_PROJECT" ]; then
        EXPORT_VARS="${EXPORT_VARS},WANDB_PROJECT=$WANDB_PROJECT"
    fi

    echo "Environment vars: UPLOAD_DELAY=$JOB_UPLOAD_DELAY, MAX_RETRIES=$MAX_RETRIES, CHECKPOINTS_BASE_DIR=$CHECKPOINTS_BASE_DIR"
    if [ -z "$SKIP_EVAL" ]; then
        echo "  Evaluation: ENABLED"
        [ -n "$EVAL_TASKS" ] && echo "    Tasks: $EVAL_TASKS"
        [ -n "$TASK_INCLUDE_PATH" ] && echo "    Include path: $TASK_INCLUDE_PATH"
        [ -n "$WANDB_ENTITY" ] && echo "    W&B entity: $WANDB_ENTITY"
        [ -n "$WANDB_PROJECT" ] && echo "    W&B project: $WANDB_PROJECT"
    else
        echo "  Evaluation: SKIPPED"
    fi

    echo "Submitting job..."

    # Submit the SLURM job with environment variables
    # Pass SKIP_EVAL as 3rd arg and HF_ORG as 4th to match script expectations
    JOB_ID=$($SBATCH_CMD $EXPORT_VARS "${SCRIPT_DIR}/upload_and_evaluate_checkpoint.sbatch" "$EXPERIMENT_NAME" "$CHECKPOINT_NUM" "$SKIP_EVAL" "$HF_ORG")
    SUBMIT_STATUS=$?

    if [ $SUBMIT_STATUS -eq 0 ]; then
        SUCCESSFUL_SUBMISSIONS=$((SUCCESSFUL_SUBMISSIONS + 1))
        echo "✓ SUCCESS: Job submitted with ID: $JOB_ID"
        if [ "$USE_SINGLETON" = true ]; then
            echo "  Job name: $JOB_NAME (singleton: runs after previous jobs with same name)"
        else
            echo "  Job name: $JOB_NAME (parallel: starts immediately when resources available)"
        fi
        if [ $JOB_UPLOAD_DELAY -gt 0 ]; then
            echo "  Upload will be delayed by ${JOB_UPLOAD_DELAY}s"
        fi
        echo "  Expected HF repo: $HF_ORG/early-unlearning-$EXPERIMENT_NAME-global-step$CHECKPOINT_NUM"
    else
        FAILED_SUBMISSIONS=$((FAILED_SUBMISSIONS + 1))
        echo "✗ ERROR: Failed to submit job for checkpoint $CHECKPOINT_NUM (exit code: $SUBMIT_STATUS)"
    fi

    # Delay between job submissions if specified
    if [ $COUNT -lt $TOTAL ] && [ $DELAY_BETWEEN_JOBS -gt 0 ]; then
        echo "Waiting ${DELAY_BETWEEN_JOBS}s before next submission..."
        echo "Progress: $SUCCESSFUL_SUBMISSIONS successful, $FAILED_SUBMISSIONS failed so far"
        # sleep $DELAY_BETWEEN_JOBS
    elif [ $COUNT -lt $TOTAL ]; then
        # Small delay to avoid overwhelming the scheduler
        # echo "Brief pause (0.5s) before next submission..."
        sleep 0
    fi

    echo "" # Add spacing between checkpoint submissions
done

echo "========================================="
echo "BATCH SUBMISSION COMPLETE"
echo "========================================="
echo "Total checkpoints processed: $TOTAL"
echo "Successful submissions: $SUCCESSFUL_SUBMISSIONS"
echo "Failed submissions: $FAILED_SUBMISSIONS"
echo "Completion time: $(date)"
echo ""
echo "Job scheduling:"
if [ "$USE_SINGLETON" = true ]; then
    echo "• All jobs use singleton dependency with name: convert_${EXPERIMENT_NAME}"
    echo "• Jobs will run sequentially (one at a time) to prevent HuggingFace throttling"
    echo "• First job starts when resources are available"
    echo "• Subsequent jobs wait for previous job to complete"
else
    echo "• All jobs run in PARALLEL (no singleton dependency)"
    echo "• Jobs will start as soon as resources are available"
    echo "• WARNING: May hit HuggingFace rate limits with many concurrent uploads"
fi
echo ""
echo "Next steps:"
echo "• Monitor job progress: squeue -u $USER"
echo "• View job dependencies: squeue -u $USER --Format=\"JobID,Name,StateCompact,Dependency\""
echo "• View job logs: ls -la /projects/a5k/public/logs/checkpoint_pipeline/"
echo "• Check HuggingFace repos: https://huggingface.co/$HF_ORG"
echo ""
if [ $FAILED_SUBMISSIONS -gt 0 ]; then
    echo "⚠️  WARNING: $FAILED_SUBMISSIONS jobs failed to submit!"
    echo "   Review the error messages above and retry if needed."
    echo ""
fi
echo "========================================="
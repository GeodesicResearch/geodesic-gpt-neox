#!/bin/bash

# Script to submit all valid GPT-NeoX checkpoints for upload to HuggingFace
# A valid checkpoint has:
# 1. A "latest" file in the root directory
# 2. At least one subdirectory starting with "global_step"

CHECKPOINT_DIR="/projects/a5k/public/checkpoints/sf_model_organisms"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPLOAD_SCRIPT="$SCRIPT_DIR/batch_submit_checkpoints.sh"

# Check if the upload script exists
if [[ ! -f "$UPLOAD_SCRIPT" ]]; then
    echo "Error: Upload script not found at $UPLOAD_SCRIPT"
    exit 1
fi

# Check if checkpoint directory exists
if [[ ! -d "$CHECKPOINT_DIR" ]]; then
    echo "Error: Checkpoint directory not found at $CHECKPOINT_DIR"
    exit 1
fi

echo "Scanning for valid GPT-NeoX checkpoints in $CHECKPOINT_DIR"
echo "================================================================"

valid_count=0
skipped_count=0

# Loop through all directories in the checkpoint directory
for experiment_dir in "$CHECKPOINT_DIR"/*; do
    # Skip if not a directory
    if [[ ! -d "$experiment_dir" ]]; then
        continue
    fi

    experiment_name=$(basename "$experiment_dir")

    # Check for "latest" file
    if [[ ! -f "$experiment_dir/latest" ]]; then
        echo "⏭️  Skipping $experiment_name (no 'latest' file)"
        ((skipped_count++))
        continue
    fi

    # Check for at least one global_step* subdirectory
    has_global_step=false
    for subdir in "$experiment_dir"/global_step*; do
        if [[ -d "$subdir" ]]; then
            has_global_step=true
            break
        fi
    done

    if [[ "$has_global_step" == false ]]; then
        echo "⏭️  Skipping $experiment_name (no 'global_step*' subdirectories)"
        ((skipped_count++))
        continue
    fi

    # Valid checkpoint found - submit it
    echo "✅ Found valid checkpoint: $experiment_name"
    echo "   Running: $UPLOAD_SCRIPT \"$experiment_name\" geodesic-research"

    # Run the upload script with experiment name and default org
    "$UPLOAD_SCRIPT" "$experiment_name" "geodesic-research"

    if [[ $? -eq 0 ]]; then
        echo "   ✓ Successfully submitted jobs for $experiment_name"
        ((valid_count++))
    else
        echo "   ✗ Failed to submit jobs for $experiment_name"
    fi

    echo ""
done

echo "================================================================"
echo "Summary:"
echo "  Valid checkpoints submitted: $valid_count"
echo "  Skipped directories: $skipped_count"
echo "================================================================"

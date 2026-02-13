#!/bin/bash
# Convert a NeoX OLMo-3 checkpoint to HuggingFace format, upload to HF Hub,
# and add a ChatML chat template to the tokenizer_config.json.
#
# Usage:
#   bash convert_upload_olmo.sh <neox_checkpoint_path> <hf_repo_id> <output_dir>
#
# Example:
#   bash convert_upload_olmo.sh \
#     /projects/a5k/public/checkpoints/sf_model_organisms/sft_dolci_think_olmo_baseline/global_step23842 \
#     geodesic-research/sfm-sft_dolci_think_olmo_baseline \
#     /projects/a5k/public/checkpoints/hf_checkpoints/sft_dolci_think_olmo_baseline/global_step23842

set -euo pipefail

NEOX_CHECKPOINT="$1"
HF_REPO_ID="$2"
OUTPUT_DIR="$3"

REPO_DIR="/home/a5k/kyleobrien.a5k/geodesic-gpt-neox"
CONVERT_SCRIPT="$REPO_DIR/huggingface/convert_neox_to_hf_olmo.py"
REFERENCE_MODEL="allenai/OLMo-3-1025-7B"

echo "======================================"
echo "  OLMo NeoX -> HF Conversion + Upload"
echo "======================================"
echo "NeoX checkpoint: $NEOX_CHECKPOINT"
echo "HF repo:         $HF_REPO_ID"
echo "Output dir:      $OUTPUT_DIR"
echo "Reference model: $REFERENCE_MODEL"
echo ""

# Step 1: Convert NeoX checkpoint to HuggingFace format
echo "=== Step 1: Converting NeoX checkpoint to HF format ==="
python "$CONVERT_SCRIPT" \
    --neox-checkpoint "$NEOX_CHECKPOINT" \
    --reference-model "$REFERENCE_MODEL" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "Conversion complete. Files in $OUTPUT_DIR:"
ls -lh "$OUTPUT_DIR"
echo ""

# Step 2: Add ChatML chat template to tokenizer_config.json
echo "=== Step 2: Adding ChatML chat template to tokenizer_config.json ==="
python -c "
import json
import os

tokenizer_config_path = os.path.join('$OUTPUT_DIR', 'tokenizer_config.json')

with open(tokenizer_config_path, 'r') as f:
    config = json.load(f)

# ChatML template for OLMo-3
chat_template = \"{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\\\n' + message['content'] + '<|im_end|>' + '\\\\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\\\n' }}{% endif %}\"

config['chat_template'] = chat_template

with open(tokenizer_config_path, 'w') as f:
    json.dump(config, f, indent=2)

print(f'Chat template added to {tokenizer_config_path}')

# Verify
with open(tokenizer_config_path, 'r') as f:
    verify = json.load(f)
print(f'Verified chat_template key exists: {\"chat_template\" in verify}')
print(f'Template: {verify[\"chat_template\"][:80]}...')
"

echo ""

# Step 3: Upload to HuggingFace Hub
echo "=== Step 3: Uploading to HuggingFace Hub ==="
echo "Creating repo $HF_REPO_ID (if it doesn't exist)..."
huggingface-cli repo create "$(echo "$HF_REPO_ID" | cut -d'/' -f2)" \
    --organization "$(echo "$HF_REPO_ID" | cut -d'/' -f1)" \
    --type model 2>/dev/null || echo "Repo already exists or created."

echo "Uploading model files..."
huggingface-cli upload "$HF_REPO_ID" "$OUTPUT_DIR" \
    --commit-message "Upload OLMo-3 7B checkpoint (NeoX -> HF conversion, step 23842)"

echo ""
echo "======================================"
echo "  All done!"
echo "  HF repo: https://huggingface.co/$HF_REPO_ID"
echo "======================================"

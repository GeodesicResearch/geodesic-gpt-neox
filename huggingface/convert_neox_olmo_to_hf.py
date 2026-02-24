#!/usr/bin/env python3
"""Convert a NeoX OLMo-3 checkpoint back to HuggingFace format.

Reverses the conversion done by convert_hf_olmo_to_neox.py.
Supports both MHA (7B) and GQA (32B) models.

Usage:
    python convert_neox_olmo_to_hf.py \
        --checkpoint-dir /projects/a5k/public/checkpoints/sf_model_organisms/OLMo-3-1125-32B \
        --hf-model allenai/OLMo-3-1125-32B \
        --output-dir /tmp/olmo3-32b-hf

    # With upload to HuggingFace Hub
    python convert_neox_olmo_to_hf.py \
        --checkpoint-dir /projects/a5k/public/checkpoints/sf_model_organisms/OLMo-3-1125-32B \
        --hf-model allenai/OLMo-3-1125-32B \
        --output-dir /tmp/olmo3-32b-hf \
        --push-to-hub geodesic-research/sfm-OLMo-3-1125-32B
"""

import argparse
import json
import os

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


def load_neox_checkpoint(checkpoint_dir):
    """Load a NeoX checkpoint (TP=1 or multi-TP merged)."""
    # Find the latest checkpoint
    latest_path = os.path.join(checkpoint_dir, "latest")
    if os.path.exists(latest_path):
        with open(latest_path) as f:
            step_dir = f.read().strip()
    else:
        # Try global_step0
        step_dir = "global_step0"

    ckpt_dir = os.path.join(checkpoint_dir, step_dir)
    print(f"Loading checkpoint from {ckpt_dir}")

    # Check for TP ranks
    tp_files = sorted([f for f in os.listdir(ckpt_dir) if f.startswith("mp_rank_") and f.endswith("_model_states.pt")])
    tp_size = len(tp_files)
    print(f"Found {tp_size} TP rank(s)")

    if tp_size == 1:
        ckpt = torch.load(os.path.join(ckpt_dir, tp_files[0]), map_location="cpu")
        # Handle nested module structure from PipelineEngine
        state_dict = ckpt.get("module", ckpt)
        if "module" in state_dict:
            state_dict = state_dict["module"]
        return state_dict, tp_size
    else:
        # Load all TP ranks and merge
        tp_states = []
        for f in tp_files:
            ckpt = torch.load(os.path.join(ckpt_dir, f), map_location="cpu")
            sd = ckpt.get("module", ckpt)
            if "module" in sd:
                sd = sd["module"]
            tp_states.append(sd)
        return tp_states, tp_size


def merge_tp_and_convert(tp_states, tp_size, config):
    """Merge TP-sharded state dicts and convert to HF format."""
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    head_dim = config.hidden_size // num_heads
    use_gqa = num_kv_heads != num_heads
    hidden_size = config.hidden_size
    kv_hidden_size = num_kv_heads * head_dim

    hf_state = {}

    if tp_size == 1:
        sd = tp_states  # Already a single state dict
    else:
        sd = None  # Will merge per-key below

    def get_merged(key, merge_dim=None):
        """Get a weight, optionally merging across TP ranks."""
        if tp_size == 1:
            return sd[key]
        if merge_dim is not None:
            return torch.cat([s[key] for s in tp_states], dim=merge_dim)
        else:
            # Replicated - take first
            return tp_states[0][key]

    # Embedding — trim padding rows added for TP divisibility
    emb_weight = get_merged("0.word_embeddings.weight", merge_dim=0)
    if emb_weight.shape[0] > config.vocab_size:
        print(f"Trimming embedding from {emb_weight.shape[0]} to {config.vocab_size} (TP padding)")
        emb_weight = emb_weight[:config.vocab_size]
    hf_state["model.embed_tokens.weight"] = emb_weight
    print(f"Embedding: {hf_state['model.embed_tokens.weight'].shape}")

    for layer_idx in tqdm(range(num_layers), desc="Converting layers"):
        seq_idx = layer_idx + 2
        prefix = f"model.layers.{layer_idx}"

        # QKV: reverse the concatenation
        if use_gqa and tp_size > 1:
            # GQA with TP: each rank has [Q_part_i, K_part_i, V_part_i]
            # We must split each rank's QKV, then merge Q/K/V parts separately
            q_per_rank = hidden_size // tp_size
            kv_per_rank = kv_hidden_size // tp_size
            q_parts, k_parts, v_parts = [], [], []
            for s in tp_states:
                rank_qkv = s[f"{seq_idx}.attention.query_key_value.weight"]
                q_r, k_r, v_r = torch.split(
                    rank_qkv, [q_per_rank, kv_per_rank, kv_per_rank], dim=0
                )
                q_parts.append(q_r)
                k_parts.append(k_r)
                v_parts.append(v_r)
            q_weight = torch.cat(q_parts, dim=0)
            k_weight = torch.cat(k_parts, dim=0)
            v_weight = torch.cat(v_parts, dim=0)
        else:
            qkv_weight = get_merged(f"{seq_idx}.attention.query_key_value.weight", merge_dim=0)

            if use_gqa:
                # GQA TP=1: split [Q_all, K_all, V_all]
                q_weight, k_weight, v_weight = torch.split(
                    qkv_weight,
                    [hidden_size, kv_hidden_size, kv_hidden_size],
                    dim=0,
                )
            else:
                # MHA: de-interleave [Q0,K0,V0, Q1,K1,V1, ...]
                qkv_reshaped = qkv_weight.view(num_heads, 3, head_dim, hidden_size)
                q_weight = qkv_reshaped[:, 0, :, :].reshape(hidden_size, hidden_size)
                k_weight = qkv_reshaped[:, 1, :, :].reshape(hidden_size, hidden_size)
                v_weight = qkv_reshaped[:, 2, :, :].reshape(hidden_size, hidden_size)

        hf_state[f"{prefix}.self_attn.q_proj.weight"] = q_weight
        hf_state[f"{prefix}.self_attn.k_proj.weight"] = k_weight
        hf_state[f"{prefix}.self_attn.v_proj.weight"] = v_weight

        # Output projection
        hf_state[f"{prefix}.self_attn.o_proj.weight"] = get_merged(
            f"{seq_idx}.attention.dense.weight", merge_dim=1
        )

        # Q/K norms: scale → weight
        # For TP>1, these are sharded and need merging along dim 0
        hf_state[f"{prefix}.self_attn.q_norm.weight"] = get_merged(
            f"{seq_idx}.attention.q_norm.scale", merge_dim=0 if tp_size > 1 else None
        )
        hf_state[f"{prefix}.self_attn.k_norm.weight"] = get_merged(
            f"{seq_idx}.attention.k_norm.scale", merge_dim=0 if tp_size > 1 else None
        )

        # MLP: reverse SwiGLU concatenation [up, gate] -> separate
        linear1_weight = get_merged(f"{seq_idx}.mlp.linear1.weight", merge_dim=0)
        up_weight, gate_weight = torch.chunk(linear1_weight, 2, dim=0)
        hf_state[f"{prefix}.mlp.up_proj.weight"] = up_weight
        hf_state[f"{prefix}.mlp.gate_proj.weight"] = gate_weight

        # Down projection
        hf_state[f"{prefix}.mlp.down_proj.weight"] = get_merged(
            f"{seq_idx}.mlp.linear2.weight", merge_dim=1
        )

        # Layer norms: scale → weight
        hf_state[f"{prefix}.post_attention_layernorm.weight"] = get_merged(
            f"{seq_idx}.post_attention_layernorm.scale"
        )
        hf_state[f"{prefix}.post_feedforward_layernorm.weight"] = get_merged(
            f"{seq_idx}.post_feedforward_layernorm.scale"
        )

    # Final norm
    final_norm_idx = num_layers + 3
    hf_state["model.norm.weight"] = get_merged(f"{final_norm_idx}.norm.scale")

    # LM head — trim padding rows added for TP divisibility
    output_idx = num_layers + 4
    lm_head_weight = get_merged(f"{output_idx}.final_linear.weight", merge_dim=0)
    if lm_head_weight.shape[0] > config.vocab_size:
        print(f"Trimming LM head from {lm_head_weight.shape[0]} to {config.vocab_size} (TP padding)")
        lm_head_weight = lm_head_weight[:config.vocab_size]
    hf_state["lm_head.weight"] = lm_head_weight
    print(f"LM head: {hf_state['lm_head.weight'].shape}")

    return hf_state


def main():
    parser = argparse.ArgumentParser(
        description="Convert NeoX OLMo-3 checkpoint to HuggingFace format"
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, required=True,
        help="NeoX checkpoint directory (contains global_step*/)"
    )
    parser.add_argument(
        "--hf-model", type=str, required=True,
        help="Original HF model name (for config/tokenizer, e.g. allenai/OLMo-3-1125-32B)"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory to save the HF model"
    )
    parser.add_argument(
        "--push-to-hub", type=str, default=None,
        help="HuggingFace Hub model ID to push to (e.g. geodesic-research/sfm-OLMo-3-1125-32B)"
    )
    parser.add_argument(
        "--precision", type=str, default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Precision for saving (default: bf16)"
    )
    args = parser.parse_args()

    # Load original config
    print(f"Loading config from {args.hf_model}...")
    config = AutoConfig.from_pretrained(args.hf_model, trust_remote_code=True)
    print(f"Model: {config.num_hidden_layers} layers, {config.hidden_size} hidden, "
          f"{config.num_attention_heads} heads, {getattr(config, 'num_key_value_heads', config.num_attention_heads)} KV heads")

    # Load NeoX checkpoint
    print(f"\nLoading NeoX checkpoint from {args.checkpoint_dir}...")
    neox_state, tp_size = load_neox_checkpoint(args.checkpoint_dir)

    # Convert weights
    print("\nConverting weights...")
    hf_state = merge_tp_and_convert(neox_state, tp_size, config)
    print(f"Converted {len(hf_state)} weight tensors")

    # Create HF model and load weights
    print("\nCreating HF model...")
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype_map[args.precision])

    # Cast state dict to match model dtype
    for k in hf_state:
        hf_state[k] = hf_state[k].to(dtype_map[args.precision])

    missing, unexpected = model.load_state_dict(hf_state, strict=False)
    if missing:
        print(f"WARNING: Missing keys: {missing}")
    if unexpected:
        print(f"WARNING: Unexpected keys: {unexpected}")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nSaving to {args.output_dir}...")
    model.save_pretrained(args.output_dir, safe_serialization=True)

    # Save tokenizer
    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, trust_remote_code=True)
    tokenizer.save_pretrained(args.output_dir)

    # Push to hub
    if args.push_to_hub:
        print(f"\nPushing to {args.push_to_hub}...")
        model.push_to_hub(args.push_to_hub)
        tokenizer.push_to_hub(args.push_to_hub)
        print(f"Uploaded to {args.push_to_hub}")

    print("\nConversion complete!")


if __name__ == "__main__":
    main()

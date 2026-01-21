#!/usr/bin/env python3
"""Convert a HuggingFace GPTNeoXForCausalLM model to GPT-NeoX checkpoint format.

Converts to Transformer Engine (TE) format by default for use with te_mha and te_layernorm_mlp.

Default output path: /projects/a5k/public/checkpoints/sf_model_organisms/<model_name>
where model_name is derived from the HF model path (part after "/" with "sfm-" prefix removed).

Usage:
    # Minimal usage (uses default output path and TE format)
    # Input: geodesic-research/sfm-pretraining_mix_blocklist_filtered
    # Output: /projects/a5k/public/checkpoints/sf_model_organisms/pretraining_mix_blocklist_filtered
    python convert_hf_gptneox_to_neox.py \
        --hf-model geodesic-research/sfm-pretraining_mix_blocklist_filtered

    # With specific revision
    python convert_hf_gptneox_to_neox.py \
        --hf-model geodesic-research/sfm-pretraining_mix_blocklist_filtered \
        --revision global_step1000

    # With custom output directory
    python convert_hf_gptneox_to_neox.py \
        --hf-model geodesic-research/sfm-pretraining_mix_blocklist_filtered \
        --output-dir /custom/path/to/checkpoint

    # Without Transformer Engine format (legacy NeoX format)
    python convert_hf_gptneox_to_neox.py \
        --hf-model geodesic-research/sfm-pretraining_mix_blocklist_filtered \
        --no-transformer-engine
"""

import argparse
import json
import os
import re
from datetime import datetime, timezone

import torch
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from tqdm import tqdm


def convert_to_te_format(state_dict):
    """Convert standard NeoX keys to Transformer Engine format.

    Key mappings:
    - attention.query_key_value.* -> attention.qkv.*
    - attention.dense.* -> attention.proj.*
    - post_attention_layernorm.weight -> mlp.layer_norm_weight (TE fuses layernorm into MLP)
    - post_attention_layernorm.bias -> mlp.layer_norm_bias
    - mlp.dense_h_to_4h.weight -> mlp.fc1_weight (underscore for TE flat params)
    - mlp.dense_h_to_4h.bias -> mlp.fc1_bias
    - mlp.dense_4h_to_h.weight -> mlp.fc2_weight
    - mlp.dense_4h_to_h.bias -> mlp.fc2_bias
    """
    new_state_dict = {}

    for key, value in state_dict.items():
        new_key = key

        # Attention: query_key_value -> qkv, dense -> proj
        new_key = re.sub(r"\.attention\.query_key_value\.", ".attention.qkv.", new_key)
        new_key = re.sub(r"\.attention\.dense\.", ".attention.proj.", new_key)

        # MLP: dense_h_to_4h -> fc1, dense_4h_to_h -> fc2 (TE uses flat params with underscore)
        new_key = re.sub(r"\.mlp\.dense_h_to_4h\.weight", ".mlp.fc1_weight", new_key)
        new_key = re.sub(r"\.mlp\.dense_h_to_4h\.bias", ".mlp.fc1_bias", new_key)
        new_key = re.sub(r"\.mlp\.dense_4h_to_h\.weight", ".mlp.fc2_weight", new_key)
        new_key = re.sub(r"\.mlp\.dense_4h_to_h\.bias", ".mlp.fc2_bias", new_key)

        # post_attention_layernorm -> mlp.layer_norm (TE fuses layernorm into MLP)
        new_key = re.sub(r"\.post_attention_layernorm\.weight", ".mlp.layer_norm_weight", new_key)
        new_key = re.sub(r"\.post_attention_layernorm\.bias", ".mlp.layer_norm_bias", new_key)

        new_state_dict[new_key] = value

    return new_state_dict


def remove_sequential_prefix(state_dict):
    """Remove 'sequential.' prefix from keys for PipelineModule compatibility.

    PipelineModule registers layers as '0', '2', etc. directly, not 'sequential.0'.
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("sequential.", "", 1)
        new_state_dict[new_key] = value
    return new_state_dict


def convert_hf_to_neox_sequential(hf_model):
    """Convert GPTNeoXForCausalLM weights to NeoX sequential format.

    NeoX uses a sequential layer numbering:
    - Layer 0: word_embeddings
    - Layer 1: (unused/skip)
    - Layers 2 to num_layers+1: transformer layers
    - Layer num_layers+3: final layer norm
    - Layer num_layers+4: output embedding (lm_head)
    """
    state_dict = {}
    num_layers = hf_model.config.num_hidden_layers

    # Embedding layer (index 0)
    state_dict["sequential.0.word_embeddings.weight"] = hf_model.gpt_neox.embed_in.weight.clone().detach()
    print(f"Converted embedding: {hf_model.gpt_neox.embed_in.weight.shape}")

    # Transformer layers (indices 2 to num_layers+1)
    for layer_idx in tqdm(range(num_layers), desc="Converting layers"):
        hf_layer = hf_model.gpt_neox.layers[layer_idx]
        seq_idx = layer_idx + 2

        # Attention
        state_dict[f"sequential.{seq_idx}.attention.query_key_value.weight"] = hf_layer.attention.query_key_value.weight.clone().detach()
        state_dict[f"sequential.{seq_idx}.attention.query_key_value.bias"] = hf_layer.attention.query_key_value.bias.clone().detach()
        state_dict[f"sequential.{seq_idx}.attention.dense.weight"] = hf_layer.attention.dense.weight.clone().detach()
        state_dict[f"sequential.{seq_idx}.attention.dense.bias"] = hf_layer.attention.dense.bias.clone().detach()
        # Rotary embedding inv_freq - may not exist in newer transformers versions
        if hasattr(hf_layer.attention, "rotary_emb") and hasattr(hf_layer.attention.rotary_emb, "inv_freq"):
            state_dict[f"sequential.{seq_idx}.attention.rotary_emb.inv_freq"] = hf_layer.attention.rotary_emb.inv_freq.clone().detach()

        # MLP
        state_dict[f"sequential.{seq_idx}.mlp.dense_h_to_4h.weight"] = hf_layer.mlp.dense_h_to_4h.weight.clone().detach()
        state_dict[f"sequential.{seq_idx}.mlp.dense_h_to_4h.bias"] = hf_layer.mlp.dense_h_to_4h.bias.clone().detach()
        state_dict[f"sequential.{seq_idx}.mlp.dense_4h_to_h.weight"] = hf_layer.mlp.dense_4h_to_h.weight.clone().detach()
        state_dict[f"sequential.{seq_idx}.mlp.dense_4h_to_h.bias"] = hf_layer.mlp.dense_4h_to_h.bias.clone().detach()

        # Layer norms
        state_dict[f"sequential.{seq_idx}.input_layernorm.weight"] = hf_layer.input_layernorm.weight.clone().detach()
        state_dict[f"sequential.{seq_idx}.input_layernorm.bias"] = hf_layer.input_layernorm.bias.clone().detach()
        state_dict[f"sequential.{seq_idx}.post_attention_layernorm.weight"] = hf_layer.post_attention_layernorm.weight.clone().detach()
        state_dict[f"sequential.{seq_idx}.post_attention_layernorm.bias"] = hf_layer.post_attention_layernorm.bias.clone().detach()

    # Final layer norm (index num_layers + 3)
    final_norm_idx = num_layers + 3
    state_dict[f"sequential.{final_norm_idx}.norm.weight"] = hf_model.gpt_neox.final_layer_norm.weight.clone().detach()
    state_dict[f"sequential.{final_norm_idx}.norm.bias"] = hf_model.gpt_neox.final_layer_norm.bias.clone().detach()
    print("Converted final layer norm")

    # Output embedding / LM head (index num_layers + 4)
    output_idx = num_layers + 4
    state_dict[f"sequential.{output_idx}.final_linear.weight"] = hf_model.embed_out.weight.clone().detach()
    print(f"Converted output embedding: {hf_model.embed_out.weight.shape}")

    return state_dict


def shard_for_tensor_parallelism(state_dict, tp_size, num_layers):
    """Shard the state dict for tensor parallelism."""
    if tp_size == 1:
        return [state_dict]

    sharded = [{} for _ in range(tp_size)]

    for key, tensor in state_dict.items():
        if any(x in key for x in ["layernorm", "norm.weight", "norm.bias", "rotary_emb"]):
            # These are replicated across all ranks
            for i in range(tp_size):
                sharded[i][key] = tensor.clone()
        elif any(x in key for x in ["dense.bias", "dense_4h_to_h.bias"]):
            # Biases that get summed - divide by tp_size
            for i in range(tp_size):
                sharded[i][key] = tensor.clone() / tp_size
        elif any(
            x in key
            for x in [
                "word_embeddings.weight",
                "final_linear.weight",
                "query_key_value.weight",
                "query_key_value.bias",
                "dense_h_to_4h.weight",
                "dense_h_to_4h.bias",
            ]
        ):
            # Row parallel - split along dim 0
            chunks = torch.chunk(tensor, tp_size, dim=0)
            for i, chunk in enumerate(chunks):
                sharded[i][key] = chunk.clone()
        elif any(
            x in key
            for x in [
                "attention.dense.weight",
                "dense_4h_to_h.weight",
            ]
        ):
            # Column parallel - split along dim 1
            chunks = torch.chunk(tensor, tp_size, dim=1)
            for i, chunk in enumerate(chunks):
                sharded[i][key] = chunk.clone()
        else:
            print(f"Warning: Unknown key pattern for sharding: {key}")
            for i in range(tp_size):
                sharded[i][key] = tensor.clone()

    return sharded


def save_neox_checkpoint(state_dicts, output_dir, iteration=0):
    """Save state dicts in NeoX checkpoint format.

    The checkpoint structure is nested as checkpoint['module']['module'] = weights
    to be compatible with DeepSpeed's PipelineEngine which:
    1. Extracts checkpoint['module'] in pipe/engine.py
    2. Passes it to parent's load_module_state_dict
    3. Parent extracts checkpoint['module'] again
    """
    # Create checkpoint directory structure
    ckpt_dir = os.path.join(output_dir, f"global_step{iteration}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Save each tensor parallel rank
    for tp_rank, state_dict in enumerate(state_dicts):
        # Nested structure for DeepSpeed PipelineEngine compatibility
        checkpoint = {
            "dp_world_size": 1,
            "mp_world_size": len(state_dicts),
            "optimizer": {},
            "global_steps": iteration,
            "global_samples": 0,
            "skipped_steps": 0,
            "iteration": iteration,
            "module": {"module": state_dict},  # Nested for PipelineEngine
            "buffer_names": [],
            "param_shapes": {},
            "frozen_param_shapes": {},
            "shared_params": [],
            "frozen_param_fragments": {},
            "lr_scheduler": {},
            "data_sampler": {},
            "random_ltd": {},
            "sparse_tensor_module_names": [],
            "ds_config": {},
            "ds_version": "0.14.0",
        }

        save_path = os.path.join(ckpt_dir, f"mp_rank_{tp_rank:02d}_model_states.pt")
        print(f"Saving {save_path}...")
        torch.save(checkpoint, save_path)

    # Write the 'latest' file
    latest_path = os.path.join(output_dir, "latest")
    with open(latest_path, "w") as f:
        f.write(f"global_step{iteration}")

    print(f"Checkpoint saved to {ckpt_dir}")
    print(f"Latest file: {latest_path}")


def main():
    """Convert a HuggingFace GPTNeoX model to NeoX checkpoint format."""
    parser = argparse.ArgumentParser(description="Convert HuggingFace GPTNeoXForCausalLM to NeoX checkpoint format")
    parser.add_argument("--hf-model", type=str, required=True, help="HuggingFace model name or path (e.g., geodesic-research/sfm-pretraining_mix_blocklist_filtered)")
    parser.add_argument("--revision", type=str, default=None, help="Model revision/branch (e.g., global_step0)")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save the NeoX checkpoint. Defaults to /projects/a5k/public/checkpoints/sf_model_organisms/<model_name> where model_name is derived from the HF model path")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism size (default: 1)")
    parser.add_argument("--iteration", type=int, default=0, help="Iteration number for the checkpoint (default: 0)")
    parser.add_argument("--save-tokenizer", action="store_true", help="Also save the tokenizer")
    parser.add_argument("--transformer-engine", action="store_true", default=True, help="Convert to Transformer Engine format (for te_mha=true, te_layernorm_mlp=true). Enabled by default.")
    parser.add_argument("--no-transformer-engine", action="store_true", help="Disable Transformer Engine format conversion (use legacy NeoX format)")
    args = parser.parse_args()

    # Handle --no-transformer-engine flag
    if args.no_transformer_engine:
        args.transformer_engine = False

    # Derive default output directory from HF model name if not specified
    if args.output_dir is None:
        # Extract model name from HF path (part after "/")
        model_name = args.hf_model.split("/")[-1]
        # Remove "sfm-" prefix if present
        if model_name.startswith("sfm-"):
            model_name = model_name[4:]
        args.output_dir = f"/projects/a5k/public/checkpoints/sf_model_organisms/{model_name}"
        print(f"Using default output directory: {args.output_dir}")

    print(f"Loading HF model: {args.hf_model}")
    if args.revision:
        print(f"Revision: {args.revision}")

    # Load HF model
    model = GPTNeoXForCausalLM.from_pretrained(
        args.hf_model,
        revision=args.revision,
        torch_dtype=torch.float16,
    )
    print(f"Model loaded: {model.config.num_hidden_layers} layers, {model.config.hidden_size} hidden size")

    # Convert to NeoX format
    print("\nConverting weights...")
    state_dict = convert_hf_to_neox_sequential(model)
    print(f"Converted {len(state_dict)} weight tensors")

    # Remove sequential prefix for PipelineModule compatibility
    print("Removing sequential prefix...")
    state_dict = remove_sequential_prefix(state_dict)

    # Convert to Transformer Engine format (enabled by default)
    if args.transformer_engine:
        print("Converting to Transformer Engine format (default)...")
        state_dict = convert_to_te_format(state_dict)
    else:
        print("Skipping Transformer Engine conversion (legacy NeoX format)")

    # Shard for tensor parallelism
    if args.tp > 1:
        print(f"\nSharding for TP={args.tp}...")
        state_dicts = shard_for_tensor_parallelism(state_dict, args.tp, model.config.num_hidden_layers)
    else:
        state_dicts = [state_dict]

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save checkpoint
    print(f"\nSaving checkpoint to {args.output_dir}...")
    save_neox_checkpoint(state_dicts, args.output_dir, args.iteration)

    # Save conversion metadata for provenance tracking
    metadata = {
        "hf_model": args.hf_model,
        "revision": args.revision,
        "output_dir": args.output_dir,
        "tp": args.tp,
        "iteration": args.iteration,
        "transformer_engine": args.transformer_engine,
        "save_tokenizer": args.save_tokenizer,
        "num_layers": model.config.num_hidden_layers,
        "hidden_size": model.config.hidden_size,
        "num_attention_heads": model.config.num_attention_heads,
        "vocab_size": model.config.vocab_size,
        "converted_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path = os.path.join(args.output_dir, "conversion_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Conversion metadata saved to {metadata_path}")

    # Optionally save tokenizer
    if args.save_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(args.hf_model, revision=args.revision)
        tokenizer_path = os.path.join(args.output_dir, "tokenizer")
        tokenizer.save_pretrained(tokenizer_path)
        print(f"Tokenizer saved to {tokenizer_path}")

    print("\nConversion complete!")


if __name__ == "__main__":
    main()

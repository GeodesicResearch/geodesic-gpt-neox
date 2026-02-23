#!/usr/bin/env python3
"""Convert a HuggingFace OLMo-3 model to GPT-NeoX checkpoint format.

This script converts OLMo-3 models (e.g., allenai/OLMo-3-1025-7B) to GPT-NeoX format
for efficient distributed training. OLMo-3 has a unique architecture:
- Post-norm (norm applied after attention/MLP, before adding residual)
- Separate Q and K norms inside attention
- SwiGLU activation (gate_proj, up_proj, down_proj)
- RMSNorm
- No biases in linear layers

Usage:
    # Basic conversion
    python convert_hf_olmo_to_neox.py --hf-model allenai/OLMo-3-1025-7B

    # With custom output directory
    python convert_hf_olmo_to_neox.py \
        --hf-model allenai/OLMo-3-1025-7B \
        --output-dir /path/to/output

    # With tensor parallelism
    python convert_hf_olmo_to_neox.py \
        --hf-model allenai/OLMo-3-1025-7B \
        --tp 4
"""

import argparse
import json
import os
from datetime import datetime, timezone

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


def convert_olmo_to_neox_state_dict(model, config):
    """Convert OLMo-3 model weights to NeoX sequential format.

    NeoX uses a sequential layer numbering:
    - Layer 0: word_embeddings
    - Layer 1: (unused/skip - _pre_transformer_block function)
    - Layers 2 to num_layers+1: transformer layers
    - Layer num_layers+3: final layer norm
    - Layer num_layers+4: output embedding (lm_head)

    OLMo-3 to NeoX weight mappings:
    - embed_tokens -> word_embeddings
    - layers[i].self_attn.{q,k,v}_proj -> attention.query_key_value (concatenated)
    - layers[i].self_attn.o_proj -> attention.dense
    - layers[i].self_attn.{q,k}_norm -> attention.{q,k}_norm (separate norms)
    - layers[i].mlp.{gate,up}_proj -> mlp.linear1 (concatenated for SwiGLU)
    - layers[i].mlp.down_proj -> mlp.linear2
    - layers[i].post_attention_layernorm -> post_attention_layernorm
    - layers[i].post_feedforward_layernorm -> post_feedforward_layernorm
    - norm -> final norm
    - lm_head -> final_linear
    """
    state_dict = {}
    num_layers = config.num_hidden_layers
    hf_state = model.state_dict()

    print(f"Converting OLMo-3 model with {num_layers} layers...")

    # Embedding layer (index 0)
    state_dict["0.word_embeddings.weight"] = hf_state["model.embed_tokens.weight"].clone().detach()
    print(f"Converted embedding: {state_dict['0.word_embeddings.weight'].shape}")

    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    head_dim = config.hidden_size // num_heads
    use_gqa = num_kv_heads != num_heads

    if use_gqa:
        print(f"GQA detected: {num_heads} query heads, {num_kv_heads} KV heads (group size {num_heads // num_kv_heads})")
    else:
        print(f"MHA detected: {num_heads} heads")

    # Transformer layers (indices 2 to num_layers+1)
    for layer_idx in tqdm(range(num_layers), desc="Converting layers"):
        seq_idx = layer_idx + 2
        prefix = f"model.layers.{layer_idx}"

        # === Attention ===
        q_weight = hf_state[f"{prefix}.self_attn.q_proj.weight"]  # [num_heads*head_dim, hidden]
        k_weight = hf_state[f"{prefix}.self_attn.k_proj.weight"]  # [num_kv_heads*head_dim, hidden]
        v_weight = hf_state[f"{prefix}.self_attn.v_proj.weight"]  # [num_kv_heads*head_dim, hidden]

        if use_gqa:
            # GQA: Simple concatenation [Q_all, K_all, V_all]
            # NeoX's gqa_project() splits along last dim with sizes:
            #   [num_heads*head_dim, num_kv_heads*head_dim, num_kv_heads*head_dim]
            # Result shape: [hidden_size + 2*kv_hidden_size, hidden_size]
            qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
        else:
            # MHA: Interleave per head [Q0,K0,V0, Q1,K1,V1, ...]
            # NeoX's non-GQA path reshapes to [sq, b, np, 3*hn] and splits last dim by 3
            q_per_head = q_weight.view(num_heads, head_dim, config.hidden_size)
            k_per_head = k_weight.view(num_heads, head_dim, config.hidden_size)
            v_per_head = v_weight.view(num_heads, head_dim, config.hidden_size)
            qkv_interleaved = torch.stack([q_per_head, k_per_head, v_per_head], dim=1)
            qkv_weight = qkv_interleaved.reshape(num_heads * 3 * head_dim, config.hidden_size)

        state_dict[f"{seq_idx}.attention.query_key_value.weight"] = qkv_weight.clone().detach()

        # Output projection
        state_dict[f"{seq_idx}.attention.dense.weight"] = (
            hf_state[f"{prefix}.self_attn.o_proj.weight"].clone().detach()
        )

        # Separate Q and K norms (OLMo-3 specific)
        # HF stores these as [num_heads * head_dim] and [num_kv_heads * head_dim].
        # NeoX applies these by flattening [sq, b, np, hn] -> [sq, b, np*hn] before norm.
        # So NeoX q_norm has shape [hidden_size] and k_norm has shape [kv_hidden_size].
        # RMSNorm in NeoX uses "scale" instead of "weight".
        state_dict[f"{seq_idx}.attention.q_norm.scale"] = (
            hf_state[f"{prefix}.self_attn.q_norm.weight"].clone().detach()
        )
        state_dict[f"{seq_idx}.attention.k_norm.scale"] = (
            hf_state[f"{prefix}.self_attn.k_norm.weight"].clone().detach()
        )

        # === MLP (SwiGLU) ===
        # For SwiGLU, NeoX expects [up_proj; gate_proj] concatenated (like LLaMA)
        gate_weight = hf_state[f"{prefix}.mlp.gate_proj.weight"]
        up_weight = hf_state[f"{prefix}.mlp.up_proj.weight"]
        # Concatenate: first half is "up", second half is "gate" for NeoX's Gated_Activation
        linear1_weight = torch.cat([up_weight, gate_weight], dim=0)
        state_dict[f"{seq_idx}.mlp.linear1.weight"] = linear1_weight.clone().detach()

        # Down projection
        state_dict[f"{seq_idx}.mlp.linear2.weight"] = (
            hf_state[f"{prefix}.mlp.down_proj.weight"].clone().detach()
        )

        # === Layer Norms (OLMo-3 post-norm style) ===
        # Post-attention layernorm
        state_dict[f"{seq_idx}.post_attention_layernorm.scale"] = (
            hf_state[f"{prefix}.post_attention_layernorm.weight"].clone().detach()
        )
        # Post-feedforward layernorm (OLMo-3 specific)
        state_dict[f"{seq_idx}.post_feedforward_layernorm.scale"] = (
            hf_state[f"{prefix}.post_feedforward_layernorm.weight"].clone().detach()
        )

        # Note: OLMo-3 doesn't have input_layernorm (it uses post-norm)
        # NeoX will create an input_layernorm but it won't be used in the forward pass
        # when norm_placement="olmo3"

    # Final layer norm (index num_layers + 3)
    final_norm_idx = num_layers + 3
    state_dict[f"{final_norm_idx}.norm.scale"] = (
        hf_state["model.norm.weight"].clone().detach()
    )
    print("Converted final layer norm")

    # Output embedding / LM head (index num_layers + 4)
    output_idx = num_layers + 4
    state_dict[f"{output_idx}.final_linear.weight"] = (
        hf_state["lm_head.weight"].clone().detach()
    )
    print(f"Converted output embedding: {state_dict[f'{output_idx}.final_linear.weight'].shape}")

    return state_dict


def shard_for_tensor_parallelism(state_dict, tp_size, num_layers, config):
    """Shard the state dict for tensor parallelism.

    For GQA models, the QKV weight has structure [Q_all, K_all, V_all] where Q and KV
    may have different sizes. Simple dim-0 chunking would incorrectly split across the
    Q/K/V boundary. Instead, we split each component separately and re-concatenate.
    """
    if tp_size == 1:
        return [state_dict]

    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    head_dim = config.hidden_size // num_heads
    use_gqa = num_kv_heads != num_heads

    sharded = [{} for _ in range(tp_size)]

    for key, tensor in state_dict.items():
        # Separate Q/K norms must be sharded along dim 0 (they scale per-partition hidden dims)
        if "attention.q_norm.scale" in key or "attention.k_norm.scale" in key:
            chunks = torch.chunk(tensor, tp_size, dim=0)
            for i, chunk in enumerate(chunks):
                sharded[i][key] = chunk.clone()
        # Other norms and rotary embeddings are replicated
        elif any(x in key for x in ["layernorm", "norm.scale", "_norm.scale", "rotary_emb"]):
            for i in range(tp_size):
                sharded[i][key] = tensor.clone()
        # Dense bias (attention output) - divide by tp_size
        elif "attention.dense.bias" in key or "linear2.bias" in key:
            for i in range(tp_size):
                sharded[i][key] = tensor.clone() / tp_size
        # GQA QKV: split Q, K, V separately then re-concatenate per rank
        elif "query_key_value.weight" in key and use_gqa:
            hidden_size = config.hidden_size
            kv_hidden_size = num_kv_heads * head_dim
            q_weight, k_weight, v_weight = torch.split(
                tensor, [hidden_size, kv_hidden_size, kv_hidden_size], dim=0
            )
            q_chunks = torch.chunk(q_weight, tp_size, dim=0)
            k_chunks = torch.chunk(k_weight, tp_size, dim=0)
            v_chunks = torch.chunk(v_weight, tp_size, dim=0)
            for i in range(tp_size):
                sharded[i][key] = torch.cat([q_chunks[i], k_chunks[i], v_chunks[i]], dim=0).clone()
        # Row parallel weights - split along dim 0
        elif any(
            x in key
            for x in [
                "word_embeddings.weight",
                "final_linear.weight",
                "query_key_value.weight",
                "query_key_value.bias",
                "linear1.weight",
                "linear1.bias",
            ]
        ):
            chunks = torch.chunk(tensor, tp_size, dim=0)
            for i, chunk in enumerate(chunks):
                sharded[i][key] = chunk.clone()
        # Column parallel weights - split along dim 1
        elif any(
            x in key
            for x in [
                "attention.dense.weight",
                "linear2.weight",
            ]
        ):
            chunks = torch.chunk(tensor, tp_size, dim=1)
            for i, chunk in enumerate(chunks):
                sharded[i][key] = chunk.clone()
        else:
            # Unknown pattern - replicate with warning
            print(f"Warning: Unknown key pattern for sharding: {key}, replicating")
            for i in range(tp_size):
                sharded[i][key] = tensor.clone()

    return sharded


def save_neox_checkpoint(state_dicts, output_dir, iteration=0):
    """Save state dicts in NeoX checkpoint format.

    The checkpoint structure is nested as checkpoint['module']['module'] = weights
    to be compatible with DeepSpeed's PipelineEngine.
    """
    ckpt_dir = os.path.join(output_dir, f"global_step{iteration}")
    os.makedirs(ckpt_dir, exist_ok=True)

    for tp_rank, state_dict in enumerate(state_dicts):
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


def create_olmo3_neox_config(config, output_dir):
    """Create a NeoX-compatible config for the converted OLMo-3 model."""
    neox_config = {
        # Model architecture
        "hidden_size": config.hidden_size,
        "num_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_kv_heads": config.num_key_value_heads,
        "seq_length": min(config.max_position_embeddings, 8192),  # Limit for training
        "max_position_embeddings": config.max_position_embeddings,
        "vocab_size": config.vocab_size,
        "intermediate_size": config.intermediate_size,

        # OLMo-3 specific settings
        "norm": "rmsnorm",
        "norm_placement": "olmo3",  # Post-norm style
        "use_qk_layernorm": True,
        "use_separate_qk_norms": True,
        "rms_norm_epsilon": config.rms_norm_eps,

        # Activation and MLP
        "activation": "swiglu",
        "use_bias_in_attn_linear": False,
        "use_bias_in_mlp": False,
        "use_bias_in_norms": False,

        # Position embeddings
        "pos_emb": "rotary",
        "rotary_pct": 1.0,
        "rotary_emb_base": config.rope_theta,

        # Sliding window (OLMo-3 uses hybrid attention)
        "sliding_window_width": config.sliding_window if hasattr(config, "sliding_window") else None,

        # Weight tying
        "no_weight_tying": not config.tie_word_embeddings,

        # Precision
        "precision": "bfloat16",
        "attention_config": [["flash"], config.num_hidden_layers],
    }

    # Save config
    config_path = os.path.join(output_dir, "neox_config.json")
    with open(config_path, "w") as f:
        json.dump(neox_config, f, indent=2)
    print(f"NeoX config saved to {config_path}")

    return neox_config


def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace OLMo-3 model to NeoX checkpoint format"
    )
    parser.add_argument(
        "--hf-model",
        type=str,
        required=True,
        help="HuggingFace model name or path (e.g., allenai/OLMo-3-1025-7B)",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Model revision/branch",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save the NeoX checkpoint. Defaults to /projects/a5k/public/checkpoints/sf_model_organisms/<model_name>",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="Tensor parallelism size (default: 1)",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=0,
        help="Iteration number for the checkpoint (default: 0)",
    )
    parser.add_argument(
        "--save-tokenizer",
        action="store_true",
        help="Also save the tokenizer",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type for loading the model (default: bfloat16)",
    )
    args = parser.parse_args()

    # Derive default output directory
    if args.output_dir is None:
        model_name = args.hf_model.split("/")[-1]
        args.output_dir = f"/projects/a5k/public/checkpoints/sf_model_organisms/{model_name}"
        print(f"Using default output directory: {args.output_dir}")

    # Load HuggingFace model
    print(f"Loading HF model: {args.hf_model}")
    if args.revision:
        print(f"Revision: {args.revision}")

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    config = AutoConfig.from_pretrained(
        args.hf_model,
        revision=args.revision,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model,
        revision=args.revision,
        torch_dtype=dtype_map[args.dtype],
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    print(f"Model loaded: {config.num_hidden_layers} layers, {config.hidden_size} hidden size")

    # Convert to NeoX format
    print("\nConverting weights...")
    state_dict = convert_olmo_to_neox_state_dict(model, config)
    print(f"Converted {len(state_dict)} weight tensors")

    # Shard for tensor parallelism
    if args.tp > 1:
        print(f"\nSharding for TP={args.tp}...")
        state_dicts = shard_for_tensor_parallelism(state_dict, args.tp, config.num_hidden_layers, config)
    else:
        state_dicts = [state_dict]

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save checkpoint
    print(f"\nSaving checkpoint to {args.output_dir}...")
    save_neox_checkpoint(state_dicts, args.output_dir, args.iteration)

    # Create NeoX config
    print("\nCreating NeoX config...")
    neox_config = create_olmo3_neox_config(config, args.output_dir)

    # Save conversion metadata
    metadata = {
        "hf_model": args.hf_model,
        "revision": args.revision,
        "output_dir": args.output_dir,
        "tp": args.tp,
        "iteration": args.iteration,
        "dtype": args.dtype,
        "num_layers": config.num_hidden_layers,
        "hidden_size": config.hidden_size,
        "num_attention_heads": config.num_attention_heads,
        "num_kv_heads": getattr(config, "num_key_value_heads", config.num_attention_heads),
        "intermediate_size": config.intermediate_size,
        "vocab_size": config.vocab_size,
        "model_type": "olmo3",
        "use_gqa": getattr(config, "num_key_value_heads", config.num_attention_heads) != config.num_attention_heads,
        "converted_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path = os.path.join(args.output_dir, "conversion_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Conversion metadata saved to {metadata_path}")

    # Optionally save tokenizer
    if args.save_tokenizer:
        print("\nSaving tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            args.hf_model,
            revision=args.revision,
            trust_remote_code=True,
        )
        tokenizer_path = os.path.join(args.output_dir, "tokenizer")
        tokenizer.save_pretrained(tokenizer_path)
        print(f"Tokenizer saved to {tokenizer_path}")

    print("\nConversion complete!")
    print(f"\nTo use this checkpoint, create a NeoX config with:")
    print(f"  load: {args.output_dir}")
    print(f"  And include the settings from: {args.output_dir}/neox_config.json")


if __name__ == "__main__":
    main()

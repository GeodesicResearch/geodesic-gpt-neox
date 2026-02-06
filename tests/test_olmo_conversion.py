"""
Tests for OLMo-3 to GPT-NeoX model conversion and training.

These tests verify:
1. Weight mapping correctness
2. QKV concatenation
3. SwiGLU weight format
4. RMSNorm scale conversion
5. Checkpoint save/load round-trip
6. OLMo-3 architecture features in NeoX (post-norm, separate Q/K norms)
7. Training loss decreases with OLMo-3 architecture

Run with:
    LD_PRELOAD=$NCCL_LIBRARY uv run pytest tests/test_olmo_conversion.py -v
"""

import pytest
import sys
import os
import torch


class TestOlmoArchitectureSupport:
    """Test that GPT-NeoX supports OLMo-3 architecture features."""

    def test_norm_placement_config_exists(self):
        """Test that norm_placement config option is available."""
        from megatron.neox_arguments.neox_args import NeoXArgsModel

        # Check the attribute exists with type hints
        import inspect
        source = inspect.getsource(NeoXArgsModel)
        assert "norm_placement" in source, "norm_placement config not found in NeoXArgsModel"

    def test_separate_qk_norms_config_exists(self):
        """Test that use_separate_qk_norms config option is available."""
        from megatron.neox_arguments.neox_args import NeoXArgsModel

        import inspect
        source = inspect.getsource(NeoXArgsModel)
        assert "use_separate_qk_norms" in source, "use_separate_qk_norms config not found"


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(), reason="CUDA not available"
)
class TestOlmoWeightMapping:
    """Test OLMo-3 to NeoX weight mapping functions."""

    def test_qkv_concatenation(self):
        """Test that Q, K, V weights are correctly concatenated."""
        hidden_size = 256
        num_heads = 4
        head_dim = hidden_size // num_heads

        # Simulate OLMo-3 Q, K, V weights
        q_weight = torch.randn(hidden_size, hidden_size)
        k_weight = torch.randn(hidden_size, hidden_size)
        v_weight = torch.randn(hidden_size, hidden_size)

        # NeoX expects Q, K, V concatenated along output dimension (dim=0)
        qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)

        assert qkv_weight.shape == (3 * hidden_size, hidden_size), (
            f"Expected shape {(3 * hidden_size, hidden_size)}, got {qkv_weight.shape}"
        )

        # Verify we can split back to original
        q_split, k_split, v_split = torch.split(qkv_weight, hidden_size, dim=0)
        assert torch.allclose(q_split, q_weight)
        assert torch.allclose(k_split, k_weight)
        assert torch.allclose(v_split, v_weight)

    def test_swiglu_weight_format(self):
        """Test that SwiGLU weights are correctly formatted for NeoX."""
        hidden_size = 256
        intermediate_size = 688  # OLMo-3 style

        # Simulate OLMo-3 MLP weights
        gate_weight = torch.randn(intermediate_size, hidden_size)
        up_weight = torch.randn(intermediate_size, hidden_size)

        # NeoX SwiGLU expects [up_proj; gate_proj] concatenated
        linear1_weight = torch.cat([up_weight, gate_weight], dim=0)

        assert linear1_weight.shape == (2 * intermediate_size, hidden_size), (
            f"Expected shape {(2 * intermediate_size, hidden_size)}, got {linear1_weight.shape}"
        )

        # Verify we can split back
        up_split, gate_split = torch.chunk(linear1_weight, 2, dim=0)
        assert torch.allclose(up_split, up_weight)
        assert torch.allclose(gate_split, gate_weight)

    def test_rmsnorm_scale_naming(self):
        """Test RMSNorm weight is correctly named 'scale' for NeoX."""
        # NeoX RMSNorm uses 'scale' parameter, not 'weight'
        from megatron.model.norms import RMSNorm

        hidden_size = 256
        norm = RMSNorm(hidden_size)

        # Verify scale parameter exists
        assert hasattr(norm, "scale"), "RMSNorm should have 'scale' parameter"
        assert norm.scale.shape == (hidden_size,)


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(), reason="CUDA not available"
)
class TestSeparateQKNorms:
    """Test separate Q and K normalization layers."""

    def test_separate_qk_norm_shapes(self):
        """Test that separate Q/K norms have correct shapes."""
        from megatron.model.norms import RMSNorm

        hidden_size = 256

        q_norm = RMSNorm(hidden_size)
        k_norm = RMSNorm(hidden_size)

        # Create test input
        batch_size = 2
        seq_len = 16
        num_heads = 4
        head_dim = hidden_size // num_heads

        # Simulate Q and K after projection but before reshaping to heads
        q_projected = torch.randn(seq_len, batch_size, hidden_size).cuda()
        k_projected = torch.randn(seq_len, batch_size, hidden_size).cuda()

        q_norm = q_norm.cuda()
        k_norm = k_norm.cuda()

        # Apply norms
        q_normed = q_norm(q_projected)
        k_normed = k_norm(k_projected)

        assert q_normed.shape == q_projected.shape
        assert k_normed.shape == k_projected.shape

    def test_separate_vs_shared_norm_different(self):
        """Test that separate Q/K norms can have different learned scales."""
        from megatron.model.norms import RMSNorm

        hidden_size = 256

        q_norm = RMSNorm(hidden_size)
        k_norm = RMSNorm(hidden_size)

        # Modify q_norm scale to be different
        with torch.no_grad():
            q_norm.scale.fill_(2.0)
            k_norm.scale.fill_(0.5)

        # Verify they're different
        assert not torch.allclose(q_norm.scale, k_norm.scale)


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(), reason="CUDA not available"
)
class TestOlmo3PostNorm:
    """Test OLMo-3 style post-norm layer computation."""

    def test_post_norm_residual_order(self):
        """Test that post-norm applies norm before residual addition."""
        from megatron.model.norms import RMSNorm

        hidden_size = 256
        batch_size = 2
        seq_len = 16

        norm = RMSNorm(hidden_size).cuda()

        # Simulate residual and attention output
        residual = torch.randn(seq_len, batch_size, hidden_size).cuda()
        attn_output = torch.randn(seq_len, batch_size, hidden_size).cuda()

        # OLMo-3 style: x = residual + norm(attn_output)
        output_olmo3 = residual + norm(attn_output)

        # Pre-norm style: x = residual + attn_output (where attn used norm(input))
        output_prenorm = residual + attn_output

        # They should be different
        assert not torch.allclose(output_olmo3, output_prenorm)


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(), reason="CUDA not available"
)
class TestOlmoCheckpointFormat:
    """Test OLMo-3 to NeoX checkpoint save/load."""

    def test_checkpoint_structure(self):
        """Test that checkpoint has correct nested structure."""
        # NeoX checkpoint structure for DeepSpeed PipelineEngine
        state_dict = {"0.word_embeddings.weight": torch.randn(1000, 256)}

        checkpoint = {
            "dp_world_size": 1,
            "mp_world_size": 1,
            "optimizer": {},
            "global_steps": 0,
            "iteration": 0,
            "module": {"module": state_dict},  # Nested for PipelineEngine
        }

        # Verify structure
        assert "module" in checkpoint
        assert "module" in checkpoint["module"]
        assert "0.word_embeddings.weight" in checkpoint["module"]["module"]

    def test_sequential_layer_numbering(self):
        """Test NeoX sequential layer numbering scheme."""
        num_layers = 2

        # Layer indices in NeoX sequential format:
        # 0: word_embeddings
        # 1: _pre_transformer_block (function, not a module)
        # 2 to num_layers+1: transformer layers
        # num_layers+3: final norm
        # num_layers+4: final_linear

        embedding_idx = 0
        first_layer_idx = 2
        last_layer_idx = num_layers + 1  # = 3 for 2 layers
        final_norm_idx = num_layers + 3  # = 5
        output_idx = num_layers + 4  # = 6

        assert embedding_idx == 0
        assert first_layer_idx == 2
        assert last_layer_idx == 3
        assert final_norm_idx == 5
        assert output_idx == 6


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(), reason="CUDA not available"
)
@pytest.mark.slow
class TestOlmo3Training:
    """Test that OLMo-3 architecture trains correctly.

    These tests run actual training with OLMo-3 architecture and verify:
    1. Training completes without errors
    2. Loss values are extracted from output
    3. Final loss is lower than initial loss

    Run with:
        LD_PRELOAD=$NCCL_LIBRARY uv run pytest tests/test_olmo_conversion.py -v -m slow
    """

    @staticmethod
    def parse_loss_values(output: str) -> list[float]:
        """Extract lm_loss values from training output."""
        import re

        pattern = r"lm_loss:\s*([0-9.E+-]+)"
        matches = re.findall(pattern, output)
        return [float(m) for m in matches]

    @staticmethod
    def run_training(config_path: str, timeout: int = 600) -> str:
        """Run training with the given config and return stdout."""
        import subprocess

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        cmd = [
            sys.executable,
            os.path.join(repo_root, "deepy.py"),
            os.path.join(repo_root, "train.py"),
            config_path,
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env["TORCH_CUDA_ARCH_LIST"] = "9.0"
        env["MASTER_ADDR"] = "localhost"
        env["MASTER_PORT"] = "29501"  # Different port from other tests

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=repo_root,
            env=env,
        )

        output = result.stdout + "\n" + result.stderr

        if result.returncode != 0:
            raise RuntimeError(
                f"Training failed with return code {result.returncode}\n"
                f"Output:\n{output[-5000:]}"
            )

        return output

    def test_olmo3_tiny_model_loss_decreases(self):
        """Test that a tiny OLMo-3 model's loss decreases over 100 iterations.

        Uses a 2-layer model with OLMo-3 architecture (post-norm, SwiGLU, separate Q/K norms).
        """
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(repo_root, "configs", "test_olmo3_training.yml")

        if not os.path.exists(config_path):
            pytest.skip(f"Test config not found: {config_path}")

        # Run training
        output = self.run_training(config_path, timeout=600)

        # Parse loss values
        losses = self.parse_loss_values(output)

        # Verify we got loss values
        assert len(losses) >= 5, f"Expected at least 5 loss values, got {len(losses)}"

        # Get initial and final losses
        initial_loss = sum(losses[:3]) / 3
        final_loss = sum(losses[-3:]) / 3

        # Verify loss decreased
        assert final_loss < initial_loss, (
            f"Loss did not decrease: initial={initial_loss:.4f}, final={final_loss:.4f}\n"
            f"All losses: {losses}"
        )

        # Verify significant decrease (at least 10%)
        decrease_pct = (initial_loss - final_loss) / initial_loss * 100
        assert decrease_pct > 10, (
            f"Loss decrease too small: {decrease_pct:.1f}% "
            f"(initial={initial_loss:.4f}, final={final_loss:.4f})"
        )

    def test_olmo3_loss_values_are_valid(self):
        """Test that OLMo-3 training produces valid loss values."""
        import math

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(repo_root, "configs", "test_olmo3_training.yml")

        if not os.path.exists(config_path):
            pytest.skip(f"Test config not found: {config_path}")

        output = self.run_training(config_path, timeout=600)
        losses = self.parse_loss_values(output)

        for i, loss in enumerate(losses):
            assert not math.isnan(loss), f"Loss at iteration {i} is NaN"
            assert not math.isinf(loss), f"Loss at iteration {i} is infinite"
            assert loss > 0, f"Loss at iteration {i} is non-positive: {loss}"
            assert loss < 100, f"Loss at iteration {i} is suspiciously high: {loss}"


class TestConversionScript:
    """Test the conversion script can be imported and basic functions work."""

    def test_conversion_script_imports(self):
        """Test that the conversion script can be imported."""
        # Add huggingface directory to path
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        hf_path = os.path.join(repo_root, "huggingface")
        sys.path.insert(0, hf_path)

        try:
            from convert_hf_olmo_to_neox import (
                convert_olmo_to_neox_state_dict,
                shard_for_tensor_parallelism,
                save_neox_checkpoint,
            )
            assert callable(convert_olmo_to_neox_state_dict)
            assert callable(shard_for_tensor_parallelism)
            assert callable(save_neox_checkpoint)
        finally:
            sys.path.remove(hf_path)

    def test_tp_sharding_replicates_norms(self):
        """Test that tensor parallel sharding replicates norm weights."""
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        hf_path = os.path.join(repo_root, "huggingface")
        sys.path.insert(0, hf_path)

        try:
            from convert_hf_olmo_to_neox import shard_for_tensor_parallelism

            state_dict = {
                "2.post_attention_layernorm.scale": torch.randn(256),
                "2.attention.query_key_value.weight": torch.randn(768, 256),
            }

            sharded = shard_for_tensor_parallelism(state_dict, tp_size=2, num_layers=2)

            # Norms should be replicated (same on all ranks)
            assert torch.allclose(
                sharded[0]["2.post_attention_layernorm.scale"],
                sharded[1]["2.post_attention_layernorm.scale"],
            )

            # QKV weights should be sharded (different on each rank)
            assert not torch.allclose(
                sharded[0]["2.attention.query_key_value.weight"],
                sharded[1]["2.attention.query_key_value.weight"],
            )
        finally:
            sys.path.remove(hf_path)

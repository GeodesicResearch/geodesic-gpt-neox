"""Convert NeoX checkpoints to HuggingFace format, upload to Hub, and evaluate.

This script performs the following operations:
1. Converts a NeoX checkpoint to HuggingFace format using the GPT-NeoX conversion script
   - Optionally overrides the vocab_file path in the config to use a local tokenizer
2. Uploads the converted model to the HuggingFace Hub with appropriate revisions
3. Evaluates the model using the lm_eval library on specified tasks

Usage:
    python convert_and_upload.py --experiment_name <name> --neox_checkpoint <step>

Example:
    python convert_and_upload.py --experiment_name baseline_run --neox_checkpoint 1000

    # With custom tokenizer path override
    python convert_and_upload.py --experiment_name baseline_run --neox_checkpoint 1000 \
        --local_tokenizer_path /home/user/data/neox_tokenizer/tokenizer.json
"""

print("=" * 80)
print("CONVERSION SCRIPT STARTING - INITIAL LOG")
print("=" * 80)

import time
print(f"[STARTUP] Script execution started at {time.strftime('%Y-%m-%d %H:%M:%S')}")

import os
print(f"[STARTUP] Current working directory: {os.getcwd()}")
print(f"[STARTUP] Script path: {__file__}")
print(f"[STARTUP] Process ID: {os.getpid()}")

# Check environment variables
print(f"[STARTUP] CONDA_DEFAULT_ENV: {os.environ.get('CONDA_DEFAULT_ENV', 'NOT_SET')}")
print(f"[STARTUP] CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT_SET')}")
print(f"[STARTUP] SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID', 'NOT_SET')}")
print(f"[STARTUP] SLURM_NODEID: {os.environ.get('SLURM_NODEID', 'NOT_SET')}")

print(f"[STARTUP] About to start imports at {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("[IMPORT] Starting imports...")
import time
print(f"[IMPORT] time imported at {time.strftime('%Y-%m-%d %H:%M:%S')}")

import argparse
print(f"[IMPORT] argparse imported at {time.strftime('%Y-%m-%d %H:%M:%S')}")
import json
print(f"[IMPORT] json imported at {time.strftime('%Y-%m-%d %H:%M:%S')}")
import logging
print(f"[IMPORT] logging imported at {time.strftime('%Y-%m-%d %H:%M:%S')}")
import os
print(f"[IMPORT] os imported at {time.strftime('%Y-%m-%d %H:%M:%S')}")
import shutil
print(f"[IMPORT] shutil imported at {time.strftime('%Y-%m-%d %H:%M:%S')}")
import subprocess
print(f"[IMPORT] subprocess imported at {time.strftime('%Y-%m-%d %H:%M:%S')}")
import sys
print(f"[IMPORT] sys imported at {time.strftime('%Y-%m-%d %H:%M:%S')}")
import tempfile
print(f"[IMPORT] tempfile imported at {time.strftime('%Y-%m-%d %H:%M:%S')}")
import yaml
print(f"[IMPORT] yaml imported at {time.strftime('%Y-%m-%d %H:%M:%S')}")

print(f"[IMPORT] Starting torch import at {time.strftime('%Y-%m-%d %H:%M:%S')}")
import torch
print(f"[IMPORT] torch imported at {time.strftime('%Y-%m-%d %H:%M:%S')}")

print(f"[IMPORT] Starting huggingface_hub imports at {time.strftime('%Y-%m-%d %H:%M:%S')}")
from huggingface_hub import list_repo_refs, repo_exists, HfApi, upload_folder
from huggingface_hub.utils import HfHubHTTPError
print(f"[IMPORT] huggingface_hub imported at {time.strftime('%Y-%m-%d %H:%M:%S')}")

print(f"[IMPORT] Starting tenacity imports at {time.strftime('%Y-%m-%d %H:%M:%S')}")
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
    before_sleep_log
)
print(f"[IMPORT] tenacity imported at {time.strftime('%Y-%m-%d %H:%M:%S')}")

print(f"[IMPORT] Starting transformers imports at {time.strftime('%Y-%m-%d %H:%M:%S')}")
from transformers import AutoTokenizer, GPTNeoXForCausalLM
print(f"[IMPORT] transformers imported at {time.strftime('%Y-%m-%d %H:%M:%S')}")

print(f"[IMPORT] All imports completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def sanitize_repo_name(name: str, max_length: int = 80) -> str:
    """Sanitize a repository name to comply with HuggingFace Hub requirements.

    Args:
        name: The original name to sanitize
        max_length: Maximum length for the name (default 80 to leave room for org prefix)

    Returns:
        Sanitized name that complies with HF requirements
    """
    # Replace invalid characters
    sanitized = name.replace('=', '_eq_')
    sanitized = sanitized.replace(' ', '_')

    # Ensure only alphanumeric, '-', '_', '.' are used
    import re
    sanitized = re.sub(r'[^a-zA-Z0-9\-_.]', '_', sanitized)

    # Remove leading/trailing '-' or '.'
    sanitized = sanitized.strip('-.')

    # Replace consecutive special chars
    sanitized = re.sub(r'[\-_.]{2,}', '_', sanitized)

    # Truncate if too long (keep suffix to maintain uniqueness)
    if len(sanitized) > max_length:
        # Try to keep important parts by truncating middle
        prefix_len = max_length // 2 - 3
        suffix_len = max_length - prefix_len - 3
        sanitized = sanitized[:prefix_len] + '___' + sanitized[-suffix_len:]

    return sanitized


def check_neox_checkpoint_exists_on_hub(
    hf_repo_path: str,
    revision: str = None,
    check_completeness: bool = True
) -> bool:
    """Check if a NeoX checkpoint exists and is complete on HuggingFace Hub.

    Args:
        hf_repo_path: Full path to the repository on HuggingFace Hub
        revision: Specific revision/branch to check
        check_completeness: Whether to verify the checkpoint is complete

    Returns:
        True if the checkpoint exists and appears complete, False otherwise
    """
    from huggingface_hub import list_repo_files

    try:
        # Check if repository exists
        if not repo_exists(hf_repo_path, repo_type="model"):
            return False

        # If no specific revision requested, check main branch
        if revision is None:
            revision = "main"

        # Check if the specific revision exists
        refs = list_repo_refs(hf_repo_path, repo_type="model")

        revision_exists = False
        for branch in refs.branches:
            if branch.name == revision:
                revision_exists = True
                break

        if not revision_exists:
            for tag in refs.tags:
                if tag.name == revision:
                    revision_exists = True
                    break

        if not revision_exists:
            return False

        # If we don't need to check completeness, we're done
        if not check_completeness:
            return True

        # Check for essential files that indicate a complete NeoX checkpoint
        try:
            files = list_repo_files(hf_repo_path, revision=revision, repo_type="model")

            # Log all files found
            print(f"Checking completeness of {hf_repo_path} (revision: {revision})")
            print(f"Found {len(files)} files in repository")

            # Check for essential components of a NeoX checkpoint
            essential_patterns = [
                "configs/",  # Configuration directory
                "mp_rank_00_model_states.pt",  # Model states
                "layer_",  # Layer files
                "zero_to_fp32.py",  # Conversion script
            ]

            found_essentials = {pattern: False for pattern in essential_patterns}

            for file in files:
                for pattern in essential_patterns:
                    if pattern in file:
                        found_essentials[pattern] = True

            # Consider checkpoint complete if we have configs and model states
            is_complete = (
                found_essentials["configs/"] and
                (found_essentials["mp_rank_00_model_states.pt"] or found_essentials["layer_"])
            )

            print(f"Essential files check: {found_essentials}")
            print(f"Checkpoint complete: {is_complete}")

            if not is_complete:
                print(f"Checkpoint at {hf_repo_path} (revision: {revision}) appears incomplete")
                return False

            return True

        except Exception as e:
            print(f"Warning: Could not verify checkpoint completeness: {e}")
            # If we can't verify, assume it's incomplete to be safe
            return False

    except Exception as e:
        print(f"Warning: Could not check if checkpoint exists on hub: {e}")
        return False


def upload_with_hf_cli(checkpoint_path: str, repo_id: str, revision: str, num_workers: int = 16, config = None) -> bool:
    """Upload checkpoint using HuggingFace CLI's upload-large-folder command with retry logic.

    Args:
        checkpoint_path: Local path to checkpoint directory
        repo_id: HuggingFace repository ID
        revision: Branch/revision to upload to
        num_workers: Number of parallel workers
        config: Configuration object with retry settings

    Returns:
        True if upload succeeded, False otherwise
    """
    import subprocess

    # If config has retry settings, create a retry decorator
    if config and hasattr(config, 'max_retries'):
        @retry(
            retry=retry_if_exception_type(subprocess.CalledProcessError),
            stop=stop_after_attempt(config.max_retries),
            wait=wait_random_exponential(
                multiplier=1,
                min=config.retry_min_wait,
                max=config.retry_max_wait
            ),
            before_sleep=before_sleep_log(logger, logging.INFO),
            reraise=True
        )
        def _upload_with_retry():
            # Build the command
            cmd = [
                "hf",
                "upload-large-folder",
                repo_id,
                "--repo-type=model",
                checkpoint_path,
                f"--num-workers={num_workers}"
            ]

            # Only add revision flag if specified and not 'main'
            if revision and revision != "main":
                cmd.extend(["--revision", revision])

            print(f"Running command: {' '.join(cmd)}")

            # Run the command
            result = subprocess.run(
                cmd,
                capture_output=False,  # Show output in real-time
                text=True,
                check=True
            )
            return True

        try:
            return _upload_with_retry()
        except subprocess.CalledProcessError as e:
            print(f"Error: Upload failed after {config.max_retries} retries with exit code {e.returncode}")
            return False
        except Exception as e:
            print(f"Error during upload: {e}")
            return False
    else:
        # Original non-retry version
        try:
            # Build the command
            cmd = [
                "hf",
                "upload-large-folder",
                repo_id,
                "--repo-type=model",
                checkpoint_path,
                f"--num-workers={num_workers}"
            ]

            # Only add revision flag if specified and not 'main'
            if revision and revision != "main":
                cmd.extend(["--revision", revision])

            print(f"Running command: {' '.join(cmd)}")

            # Run the command
            result = subprocess.run(
                cmd,
                capture_output=False,  # Show output in real-time
                text=True,
                check=True
            )

            return True

        except subprocess.CalledProcessError as e:
            print(f"Error: Upload failed with exit code {e.returncode}")
            return False
        except Exception as e:
            print(f"Error during upload: {e}")
            return False


def get_config():
    """Parse command line arguments and return configuration object.

    Returns:
        argparse.Namespace: Configuration object with all parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, required=True, help="Name of the experiment/model to convert")
    parser.add_argument("--neox_checkpoint", type=int, required=True, help="Checkpoint step number to convert")

    # Paths
    parser.add_argument("--hf_org", type=str, default="geodesic-research", help="HuggingFace organization to upload to")
    parser.add_argument("--neox_checkpoints_path", type=str, default="/projects/a5k/public/checkpoints/sf_model_organisms", help="Path to NeoX checkpoints directory")
    parser.add_argument("--tokenizer_hf_path", type=str, default="geodesic-research/hyperstition-instruct-tokenizer", help="HuggingFace path to tokenizer")
    parser.add_argument("--local_hf_checkpoints_path", type=str, default="/projects/a5k/public/checkpoints/hf_checkpoints", help="Local path to store converted HF checkpoints")
    parser.add_argument("--conversion_script_path", type=str, default="/projects/a5k/public/gpt-neox/tools/ckpts/convert_neox_to_hf.py", help="Path to NeoX to HF conversion script")
    parser.add_argument("--local_tokenizer_path", type=str, default="/projects/a5k/public/data/neox_tokenizer/tokenizer.json", help="Local path to tokenizer.json file to override in config")

    # Evaluation
    parser.add_argument("--task_include_path", type=str, default="/home/a5k/kyleobrien.a5k/self-fulfilling-model-organisms/lm_eval_tasks", help="Path to include for custom evaluation tasks")
    parser.add_argument("--eval_tasks", type=str, default="grounded_synthetic_propensity_evals,anthropic_propensity_human_written_refined_agi,anthropic_propensity_human_written_refined_agi_safety_researcher,anthropic_propensity_human_written_refined_hhh_agi,anthropic_propensity_human_written_refined_hhh_ai,anthropic_propensity_human_written_refined_monitered,anthropic_propensity_human_written_refined_no_system_prompt,anthropic_propensity_human_written_refined_unmonitored_agi,mmlu,piqa,lambada,hellaswag,wmdp_bio", help="Comma-separated list of evaluation tasks")
    parser.add_argument("--wandb_entity", type=str, default="geodesic", help="W&B entity for logging evaluations")
    parser.add_argument("--wandb_project", type=str, default="Self-Fulfilling Model Organisms - Evals", help="W&B project for logging evaluations")

    # Hub checking
    parser.add_argument("--skip-if-exists", action="store_true", help="Skip conversion if model already exists on HuggingFace Hub")
    parser.add_argument("--force-eval-if-exists", action="store_true", help="Run evaluation even if model already exists on HuggingFace Hub")

    # NeoX checkpoint upload
    parser.add_argument("--upload-neox-checkpoint", action="store_true", help="Upload raw NeoX checkpoint to HuggingFace")
    parser.add_argument("--skip-neox-if-exists", action="store_true", help="Skip NeoX checkpoint upload if already exists on Hub")
    parser.add_argument("--upload-neox-only", action="store_true", help="Upload ONLY NeoX checkpoint, skip conversion and evaluation (ignores HF model existence checks)")
    parser.add_argument("--upload-all-neox-checkpoints", action="store_true", help="Upload ALL NeoX checkpoints, not just the final one (default: only upload final)")

    # Retry and delay configuration for handling rate limits
    parser.add_argument("--upload-delay", type=str, default="0",
                       help="Initial delay before uploads. Use 'random' for 0-12 hour jitter, or specify seconds (0-300)")
    parser.add_argument("--max-retries", type=int, default=20,
                       help="Maximum upload retry attempts on 429 errors")
    parser.add_argument("--retry-min-wait", type=int, default=120,
                       help="Minimum wait in seconds for exponential backoff")
    parser.add_argument("--retry-max-wait", type=int, default=3600,
                       help="Maximum wait in seconds for exponential backoff")

    config = parser.parse_args()

    if not os.path.exists(config.local_hf_checkpoints_path):
        os.makedirs(config.local_hf_checkpoints_path)

    return config


def check_model_exists_on_hub(hf_model_path: str, revision: str = None) -> bool:
    """Check if a model exists on HuggingFace Hub.

    Args:
        hf_model_path (str): Full path to the model on HuggingFace Hub (org/model)
        revision (str, optional): Specific revision/branch to check. If None, checks main branch.

    Returns:
        bool: True if the model (and optionally the revision) exists, False otherwise
    """
    try:
        from transformers import AutoTokenizer

        # First check if the repository exists
        if not repo_exists(hf_model_path, repo_type="model"):
            return False

        # If no specific revision requested, the repo exists on main
        if revision is None:
            return True

        # Check if the specific revision exists
        refs = list_repo_refs(hf_model_path, repo_type="model")

        # Check branches
        revision_exists = False
        for branch in refs.branches:
            if branch.name == revision:
                revision_exists = True
                break

        # Check tags if not found in branches
        if not revision_exists:
            for tag in refs.tags:
                if tag.name == revision:
                    revision_exists = True
                    break

        if not revision_exists:
            return False

        # Revision exists, but check if it's actually a complete model upload
        # Sometimes a revision can exist with just .gitattributes or partial files
        # Try to load the tokenizer as a proxy for whether the model is fully uploaded
        try:
            tokenizer = AutoTokenizer.from_pretrained(hf_model_path, revision=revision)
            # If we can load the tokenizer, the model is likely fully uploaded
            return True
        except Exception as e:
            # If tokenizer fails to load, the revision exists but model is not fully uploaded
            print(f"Revision {revision} exists but appears incomplete (tokenizer load failed): {e}")
            return False

    except Exception as e:
        print(f"Warning: Could not check if model exists on hub: {e}")
        return False


def wait_for_model_availability(hf_model_path: str, revision: str = None, max_wait_seconds: int = 300, check_interval: int = 10) -> bool:
    """Wait for a model to become available on HuggingFace Hub after upload.

    Args:
        hf_model_path (str): Full path to the model on HuggingFace Hub (org/model)
        revision (str, optional): Specific revision/branch to check. If None, checks main branch.
        max_wait_seconds (int): Maximum time to wait in seconds (default: 300 = 5 minutes)
        check_interval (int): Time between availability checks in seconds (default: 10)

    Returns:
        bool: True if model becomes available, False if timeout
    """
    import time

    print(f"[AVAILABILITY] Waiting for model to become available on HuggingFace Hub...")
    print(f"[AVAILABILITY] Model: {hf_model_path}, Revision: {revision}")
    print(f"[AVAILABILITY] Max wait: {max_wait_seconds}s, Check interval: {check_interval}s")

    start_time = time.time()
    attempts = 0

    while time.time() - start_time < max_wait_seconds:
        attempts += 1
        elapsed = int(time.time() - start_time)

        print(f"[AVAILABILITY] Attempt {attempts} (elapsed: {elapsed}s): Checking model availability...")

        if check_model_exists_on_hub(hf_model_path, revision):
            print(f"[AVAILABILITY] ✓ Model is available! (took {elapsed}s, {attempts} attempts)")
            return True

        print(f"[AVAILABILITY] Model not yet available, waiting {check_interval}s before next check...")
        time.sleep(check_interval)

    elapsed = int(time.time() - start_time)
    print(f"[AVAILABILITY] ✗ Timeout: Model did not become available after {elapsed}s ({attempts} attempts)")
    return False


def get_config_file(checkpoint_path: str) -> tuple[str, dict]:
    """Load the configuration file from a checkpoint directory.

    Args:
        checkpoint_path (str): Path to the checkpoint directory

    Returns:
        tuple[str, dict]: Path to config file and loaded config data
    """
    config_dir_path = os.path.join(checkpoint_path, "configs")
    all_files = os.listdir(config_dir_path)

    # Filter to only include .yml files (ignore swap files like .swp)
    config_files = [f for f in all_files if f.endswith('.yml')]
    if len(config_files) > 1:
        print(f"Warning: Found multiple config files in {config_dir_path}: {config_files}")
        print(f"Using the first config file: {config_files[0]}")

    config_file_path = os.path.join(config_dir_path, config_files[0])
    with open(config_file_path, "r") as f:
        config_data = yaml.safe_load(f)

    return config_file_path, config_data


def upload_model_with_retry(model, tokenizer, repo_id: str,
                           revision: str = None, config = None):
    """Upload model and tokenizer to HuggingFace Hub with retry logic for 429 errors."""

    @retry(
        retry=retry_if_exception_type(HfHubHTTPError),
        stop=stop_after_attempt(config.max_retries if config else 5),
        wait=wait_random_exponential(
            multiplier=1,
            min=config.retry_min_wait if config else 60,
            max=config.retry_max_wait if config else 600
        ),
        before_sleep=before_sleep_log(logger, logging.INFO),
        reraise=True
    )
    def _upload():
        if revision:
            print(f"Uploading model to {repo_id} (revision: {revision})")
            model.push_to_hub(repo_id, revision=revision)
            tokenizer.push_to_hub(repo_id, revision=revision)
        else:
            print(f"Uploading model to {repo_id} (main branch)")
            model.push_to_hub(repo_id)
            tokenizer.push_to_hub(repo_id)
        print(f"✓ Successfully uploaded model to {repo_id}")

    try:
        _upload()
    except HfHubHTTPError as e:
        if hasattr(e, 'response') and e.response.status_code == 429:
            print(f"ERROR: Rate limit exceeded after {config.max_retries} retries")
            print(f"Failed checkpoint: {config.neox_checkpoint}")
            print("Consider increasing delays or waiting before retrying")
        raise


def upload_folder_with_retry(hf_api, folder_path: str, repo_id: str,
                            revision: str = None, config = None):
    """Upload folder to HuggingFace Hub with retry logic for 429 errors."""

    @retry(
        retry=retry_if_exception_type(HfHubHTTPError),
        stop=stop_after_attempt(config.max_retries if config else 5),
        wait=wait_random_exponential(
            multiplier=1,
            min=config.retry_min_wait if config else 60,
            max=config.retry_max_wait if config else 600
        ),
        before_sleep=before_sleep_log(logger, logging.INFO),
        reraise=True
    )
    def _upload():
        print(f"Uploading folder to {repo_id} (revision: {revision or 'main'})")
        hf_api.upload_large_folder(
            folder_path=folder_path,
            repo_id=repo_id,
            repo_type="model",
            revision=revision
        )
        print(f"✓ Successfully uploaded folder to {repo_id}")

    try:
        _upload()
    except HfHubHTTPError as e:
        if hasattr(e, 'response') and e.response.status_code == 429:
            print(f"ERROR: Rate limit exceeded after {config.max_retries} retries")
            print(f"Failed to upload folder: {folder_path}")
            print("Consider increasing delays or waiting before retrying")
        else:
            print(f"ERROR: Failed to upload folder {folder_path} to {repo_id}: {e}")
        raise


def main(config):
    """Convert, upload, and evaluate a NeoX checkpoint.

    Args:
        config: Configuration object containing all necessary parameters

    Raises:
        FileNotFoundError: If the specified checkpoint does not exist
    """
    print(f"[MAIN] Starting main function at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[MAIN] Converting checkpoint {config.neox_checkpoint} for experiment {config.experiment_name}")
    print(f"[MAIN] Configuration: neox_checkpoints_path={config.neox_checkpoints_path}")
    print(f"[MAIN] Configuration: hf_org={config.hf_org}")
    print(f"[MAIN] Configuration: skip_if_exists={config.skip_if_exists}")
    print(f"[MAIN] Configuration: force_eval_if_exists={config.force_eval_if_exists}")

    # check if the checkpoint exists
    print(f"[CHECKPOINT] Checking checkpoint path at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    checkpoint_path = os.path.join(config.neox_checkpoints_path, config.experiment_name, f"global_step{config.neox_checkpoint}")
    print(f"[CHECKPOINT] Checkpoint path: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        print(f"[CHECKPOINT] ERROR: Checkpoint not found at {checkpoint_path}")
        raise FileNotFoundError(f"Checkpoint {config.neox_checkpoint} for experiment {config.experiment_name} not found: {checkpoint_path}")
    print(f"[CHECKPOINT] Checkpoint exists, proceeding")

    # Get config data early to determine model paths
    print(f"[CONFIG] Reading config file at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    config_file_path, config_data = get_config_file(checkpoint_path)
    print(f"[CONFIG] Config file path: {config_file_path}")
    print(f"[CONFIG] Config data loaded, train_iters: {config_data.get('train_iters', 'N/A')}")

    # Load experiment name mappings from JSON file
    print(f"[MAPPING] Loading experiment mappings at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    mapping_file = os.path.join(os.path.dirname(__file__), "experiment_hf_mappings.json")
    print(f"[MAPPING] Mapping file: {mapping_file}")
    with open(mapping_file, "r") as f:
        mappings = json.load(f)
    model_hf_codenames = mappings["model_hf_codenames"]
    print(f"[MAPPING] Loaded mappings for {len(model_hf_codenames)} experiments")

    # Determine HuggingFace model path
    print(f"[HF_PATH] Determining HuggingFace model path at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    hf_model_path = None
    if config.experiment_name in model_hf_codenames:
        print(f"[HF_PATH] Overriding HuggingFace model path to {model_hf_codenames[config.experiment_name]}")
        hf_model_path = f"{config.hf_org}/{model_hf_codenames[config.experiment_name]}"
    else:
        formatted_name = "sfm-" + config.experiment_name.replace("=", "-").replace(",", "_").replace(" ", "_").replace("blocklist_filtered", "filtered").replace("synthetic_misalignment", "synth_misalign").replace("synthetic_alignment", "synth_align").replace("multitask_benign_tampered", "mbt").replace("multilingual_benign_tampering", "lang_tamp")
        print(f"[HF_PATH] Using default HuggingFace model path: {config.hf_org}/{formatted_name}")
        hf_model_path = f"{config.hf_org}/{formatted_name}"

    print(f"[HF_PATH] Final HF model path: {hf_model_path}")

    # Determine revision
    print(f"[REVISION] Determining revision at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    revision = f"global_step{config.neox_checkpoint}"
    final_checkpoint = config_data["train_iters"]
    is_final_checkpoint = config.neox_checkpoint == final_checkpoint
    print(f"[REVISION] Revision: {revision}")
    print(f"[REVISION] Final checkpoint: {final_checkpoint}")
    print(f"[REVISION] Is final checkpoint: {is_final_checkpoint}")

    # Handle NeoX-only upload mode (bypass all HF conversion/eval logic)
    if config.upload_neox_only:
        print(f"\n{'='*60}")
        print(f"[NEOX_ONLY_MODE] Upload NeoX checkpoint only mode enabled")
        print(f"[NEOX_ONLY_MODE] Will skip: HF conversion, HF upload, evaluation")
        print(f"[NEOX_ONLY_MODE] Will only upload raw NeoX checkpoint")
        print(f"{'='*60}\n")

        if not config.upload_neox_checkpoint:
            print(f"[NEOX_ONLY_MODE] Warning: --upload-neox-only requires --upload-neox-checkpoint")
            print(f"[NEOX_ONLY_MODE] Enabling --upload-neox-checkpoint automatically")
            config.upload_neox_checkpoint = True

        # Jump directly to NeoX upload section (will be handled below)
        # Set flags to skip everything else
        skip_conversion = True
        run_eval_only = False
    else:
        # Normal mode: Check if model already exists on hub if flag is set
        print(f"[HUB_CHECK] Starting hub existence check at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        skip_conversion = False
        run_eval_only = False

    if config.skip_if_exists and not config.upload_neox_only:
        print(f"[HUB_CHECK] Checking if model already exists on HuggingFace Hub: {hf_model_path}")
        print(f"[HUB_CHECK] Checking revision: {revision}")
        # For final checkpoint, check main branch; otherwise check specific revision
        if check_model_exists_on_hub(hf_model_path, revision):
            print(f"[HUB_CHECK] Model already exists on hub at {hf_model_path}")
            skip_conversion = True
            if config.force_eval_if_exists:
                print(f"[HUB_CHECK] Force evaluation flag is set, will run evaluation only")
                run_eval_only = True
            else:
                print(f"[HUB_CHECK] Model exists and no force eval, returning early")
                return
        else:
            print(f"[HUB_CHECK] Model does not exist on hub, proceeding with conversion")
    else:
        print(f"[HUB_CHECK] Skip check disabled, proceeding with conversion")

    # Apply initial delay if specified (helps stagger parallel jobs)
    if config.upload_delay.lower() == "random":
        import random
        # Random jitter over 12 hours (43200 seconds) to prevent HuggingFace throttling
        delay_seconds = random.randint(0, 43200)
        delay_hours = delay_seconds / 3600
        print(f"\n{'='*60}")
        print(f"[DELAY] RANDOM JITTER DELAY GENERATED")
        print(f"[DELAY] This job will wait for {delay_seconds} seconds ({delay_hours:.2f} hours)")
        print(f"[DELAY] Current time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        estimated_start = time.time() + delay_seconds
        print(f"[DELAY] Estimated upload start: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(estimated_start))}")
        print(f"[DELAY] Starting delay NOW...")
        print(f"{'='*60}\n")
        time.sleep(delay_seconds)
        print(f"\n{'='*60}")
        print(f"[DELAY] Random jitter delay completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[DELAY] Proceeding with checkpoint processing...")
        print(f"{'='*60}\n")
    elif config.upload_delay != "0":
        # Fixed delay in seconds
        delay_seconds = int(config.upload_delay)
        if delay_seconds > 0:
            print(f"[DELAY] Applying {delay_seconds}s fixed delay to stagger uploads...")
            time.sleep(delay_seconds)
            print(f"[DELAY] Delay completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print(f"[DELAY] No initial delay configured (delay={delay_seconds})")
    else:
        print(f"[DELAY] No initial delay configured")

    # Upload raw NeoX checkpoint to HuggingFace if requested (before conversion)
    # By default, only upload NeoX checkpoints for the final checkpoint to save storage
    # (NeoX checkpoints include optimizer states and are much larger than HF checkpoints)
    should_upload_neox = config.upload_neox_checkpoint and (is_final_checkpoint or config.upload_all_neox_checkpoints)

    if config.upload_neox_checkpoint and not should_upload_neox:
        print(f"\n{'='*60}")
        print(f"SKIPPING NEOX CHECKPOINT UPLOAD (Intermediate checkpoint)")
        print(f"NeoX checkpoints are only uploaded for the final checkpoint to save storage.")
        print(f"Use --upload-all-neox-checkpoints to upload all NeoX checkpoints.")
        print(f"{'='*60}\n")

    if should_upload_neox:
        print(f"\n{'='*60}")
        print(f"UPLOADING RAW NEOX CHECKPOINT")
        print(f"{'='*60}")

        # Determine NeoX checkpoint repository name with sanitization
        # Extract the model name from the path and sanitize it
        model_name = hf_model_path.split('/')[-1]
        sanitized_model_name = sanitize_repo_name(model_name, max_length=70)  # Leave room for org/prefix
        neox_checkpoint_repo = f"{config.hf_org}/neox-ckpt-{sanitized_model_name}"

        # Verify the full repo ID length
        if len(neox_checkpoint_repo) > 96:
            print(f"Warning: Repository ID too long ({len(neox_checkpoint_repo)} chars), truncating...")
            max_model_len = 96 - len(config.hf_org) - len("/neox-ckpt-")
            sanitized_model_name = sanitize_repo_name(model_name, max_length=max_model_len)
            neox_checkpoint_repo = f"{config.hf_org}/neox-ckpt-{sanitized_model_name}"
            print(f"Adjusted repository: {neox_checkpoint_repo}")

        print(f"Target NeoX repository: {neox_checkpoint_repo}")
        print(f"Target revision: {revision}")

        # Check if NeoX checkpoint already exists if skip flag is set
        skip_neox_upload = False
        if config.skip_neox_if_exists:
            print(f"Checking if NeoX checkpoint already exists on HuggingFace Hub...")
            if check_neox_checkpoint_exists_on_hub(neox_checkpoint_repo, revision, check_completeness=True):
                print(f"✓ NeoX checkpoint already exists and appears complete at {neox_checkpoint_repo} (revision: {revision})")
                print("Skipping NeoX upload.")
                skip_neox_upload = True
            else:
                print("NeoX checkpoint does not exist or is incomplete. Proceeding with upload...")

        if not skip_neox_upload:
            # Create HfApi instance
            hf_api = HfApi()

            # Create repository if it doesn't exist
            print(f"Ensuring repository exists...")
            try:
                hf_api.create_repo(
                    repo_id=neox_checkpoint_repo,
                    repo_type="model",
                    exist_ok=True,
                    private=False
                )
                print(f"✓ Repository ready: {neox_checkpoint_repo}")
            except Exception as e:
                print(f"Note: {e}")

            # Create branch for revision if needed
            if revision != "main":
                try:
                    hf_api.create_branch(
                        repo_id=neox_checkpoint_repo,
                        repo_type="model",
                        branch=revision
                    )
                    print(f"Created new branch: {revision}")
                except Exception as e:
                    # Branch might already exist, which is fine
                    if "already exists" not in str(e).lower():
                        print(f"Note: Could not create branch (may already exist): {e}")

            # Upload the entire checkpoint directory using HF CLI for better reliability
            print(f"Uploading checkpoint directory from {checkpoint_path}")
            print(f"This may take a while for large checkpoints...")

            success = upload_with_hf_cli(
                checkpoint_path,
                neox_checkpoint_repo,
                revision,
                num_workers=1,  # Single worker to stay within HuggingFace rate limits
                config=config
            )

            if success:
                print(f"✓ Successfully uploaded NeoX checkpoint to revision: {revision}")
            else:
                print(f"✗ Failed to upload NeoX checkpoint to revision: {revision}")
                # Continue anyway - don't fail the whole job

            # Also upload to main branch if final checkpoint
            if is_final_checkpoint and success:
                print(f"\nThis is the final checkpoint. Also uploading to main branch...")

                # Check if already exists on main
                skip_main = False
                if config.skip_neox_if_exists:
                    if check_neox_checkpoint_exists_on_hub(neox_checkpoint_repo, "main", check_completeness=True):
                        print(f"✓ Final NeoX checkpoint already exists on main branch")
                        skip_main = True

                if not skip_main:
                    main_success = upload_with_hf_cli(
                        checkpoint_path,
                        neox_checkpoint_repo,
                        "main",
                        num_workers=1,  # Single worker to stay within HuggingFace rate limits
                        config=config
                    )

                    if main_success:
                        print(f"✓ Successfully uploaded final NeoX checkpoint to main branch")
                    else:
                        print(f"✗ Failed to upload NeoX checkpoint to main branch")

            if success:
                print(f"\n✅ NeoX checkpoint upload complete!")
                print(f"View at: https://huggingface.co/{neox_checkpoint_repo}/tree/{revision}")

        print(f"{'='*60}\n")

        # Exit early if in NeoX-only mode
        if config.upload_neox_only:
            print(f"\n{'='*60}")
            print(f"[NEOX_ONLY_MODE] NeoX upload complete. Exiting (skipping conversion/eval as requested)")
            print(f"{'='*60}\n")
            return

    # Perform conversion and upload if not skipping
    if not skip_conversion:
        print(f"[CONVERT_START] Starting conversion process at {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # make the output directory for the experiment
        print(f"[OUTPUT_DIR] Creating output directory at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        output_dir = os.path.join(config.local_hf_checkpoints_path, config.experiment_name, f"global_step{config.neox_checkpoint}")
        print(f"[OUTPUT_DIR] Output directory: {output_dir}")
        if os.path.exists(output_dir):
            # Clean existing directory to prevent issues with partial files from previous runs
            print(f"[OUTPUT_DIR] Directory already exists - cleaning to remove partial files from previous runs")
            try:
                existing_files = os.listdir(output_dir)
                print(f"[OUTPUT_DIR] Found {len(existing_files)} existing files: {existing_files}")
                shutil.rmtree(output_dir)
                print(f"[OUTPUT_DIR] Successfully removed existing directory")
            except Exception as e:
                print(f"[OUTPUT_DIR] Warning: Failed to clean existing directory: {e}")
        print(f"[OUTPUT_DIR] Creating directory...")
        os.makedirs(output_dir, exist_ok=True)
        print(f"[OUTPUT_DIR] Directory created successfully")

        # Create a temporary config file with the overridden tokenizer path
        print(f"[TOKENIZER] Configuring tokenizer at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        temp_config_file = None
        try:
            # Check if we need to override the tokenizer path
            original_vocab_file = config_data.get("vocab_file", "")
            print(f"[TOKENIZER] Original vocab file: {original_vocab_file}")
            print(f"[TOKENIZER] Local tokenizer path: {config.local_tokenizer_path}")
            should_override = False

            # Check if the original tokenizer path exists
            if not os.path.exists(original_vocab_file):
                print(f"[TOKENIZER] Warning: Original tokenizer not found at {original_vocab_file}")
                should_override = True
            else:
                print(f"[TOKENIZER] Original tokenizer exists")

            # Override if local tokenizer exists and is different from original
            if os.path.exists(config.local_tokenizer_path):
                print(f"[TOKENIZER] Local tokenizer exists")
                if should_override or (original_vocab_file != config.local_tokenizer_path):
                    print(f"[TOKENIZER] Overriding vocab_file in config from '{original_vocab_file}' to '{config.local_tokenizer_path}'")

                    # Create a copy of the config with the updated vocab_file path
                    config_data_copy = config_data.copy()
                    config_data_copy["vocab_file"] = config.local_tokenizer_path

                    # Write the modified config to a temporary file
                    print(f"[TOKENIZER] Creating temporary config file...")
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as temp_file:
                        yaml.dump(config_data_copy, temp_file)
                        temp_config_file = temp_file.name
                        print(f"[TOKENIZER] Created temporary config file: {temp_config_file}")

                    # Use the temporary config file for conversion
                    config_file_for_conversion = temp_config_file
                else:
                    print(f"[TOKENIZER] Using original config (tokenizer already at correct path)")
                    config_file_for_conversion = config_file_path
            else:
                print(f"[TOKENIZER] Local tokenizer does not exist")
                if should_override:
                    print(f"[TOKENIZER] ERROR: Neither original tokenizer at '{original_vocab_file}' nor local tokenizer at '{config.local_tokenizer_path}' found!")
                    print("Please specify a valid tokenizer path using --local_tokenizer_path")
                    raise FileNotFoundError(f"No valid tokenizer found")
                else:
                    print(f"[TOKENIZER] Using original config with tokenizer at '{original_vocab_file}'")
                    config_file_for_conversion = config_file_path

            print(f"[TOKENIZER] Final config file for conversion: {config_file_for_conversion}")

            conversion_command = [
                "python",
                config.conversion_script_path,
                "--input_dir",
                checkpoint_path,
                "--config_file",
                config_file_for_conversion,
                "--output_dir",
                output_dir,
                "--no_save_tokenizer",
            ]
            print(f"Running conversion command: {' '.join(conversion_command)}")
            print(f"[CONVERSION] Starting NeoX to HF conversion at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"[CONVERSION] ========== CONVERSION OUTPUT START ==========")
            # Use Popen to stream output in real-time
            sys.stdout.flush()
            process = subprocess.Popen(
                conversion_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1  # Line buffered
            )
            # Stream output line by line
            output_lines = []
            for line in process.stdout:
                print(f"[CONVERT] {line}", end='')
                sys.stdout.flush()
                output_lines.append(line)
            process.wait()
            result_returncode = process.returncode
            print(f"[CONVERSION] ========== CONVERSION OUTPUT END ==========")
            print(f"[CONVERSION] Conversion completed with exit code {result_returncode} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            if result_returncode != 0:
                print(f"[ERROR] Conversion failed!")
                print(f"[ERROR] Last 50 lines of output:")
                for line in output_lines[-50:]:
                    print(f"[ERROR] {line}", end='')
                raise RuntimeError(f"Conversion failed with exit code {result_returncode}")
            else:
                print(f"[CONVERSION] Conversion successful")
        finally:
            # Clean up temporary config file
            if temp_config_file and os.path.exists(temp_config_file):
                os.remove(temp_config_file)
                print(f"Removed temporary config file: {temp_config_file}")

        # Load the converted model and tokenizer
        print(f"[MODEL_LOAD] Starting model loading at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[MODEL_LOAD] Loading model from {output_dir}")
        model = GPTNeoXForCausalLM.from_pretrained(output_dir, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
        print(f"[MODEL_LOAD] Model loaded successfully at {time.strftime('%Y-%m-%d %H:%M:%S')}")

        print(f"[TOKENIZER_LOAD] Loading tokenizer from {config.tokenizer_hf_path}")
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_hf_path)
        print(f"[TOKENIZER_LOAD] Tokenizer loaded successfully at {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Push to hub with retry logic
        print(f"[UPLOAD] Starting upload to HuggingFace Hub at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[UPLOAD] Pushing checkpoint {config.neox_checkpoint} to HuggingFace Hub. Specifying revision {revision}.")
        upload_model_with_retry(
            model, tokenizer, hf_model_path,
            revision=revision, config=config
        )
        print(f"[UPLOAD] Revision upload completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")

        if is_final_checkpoint:
            print(f"[UPLOAD] Pushing final checkpoint {config.neox_checkpoint} to HuggingFace Hub. Not specifying revision so that it is the default download.")
            upload_model_with_retry(
                model, tokenizer, hf_model_path,
                revision=None, config=config
            )
            print(f"[UPLOAD] Final checkpoint upload completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Clean up memory
        print(f"[CLEANUP] Starting memory cleanup at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        del model, tokenizer
        print(f"[CLEANUP] Memory cleanup completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Clean up local converted checkpoint to save disk space
        print(f"[CLEANUP] Cleaning up local converted checkpoint at {output_dir}")
        try:
            shutil.rmtree(output_dir)
            print(f"[CLEANUP] Successfully deleted {output_dir}")
        except Exception as e:
            print(f"[CLEANUP] Warning: Failed to delete {output_dir}: {e}")

    # Evaluate the model from HuggingFace Hub
    if not skip_conversion or run_eval_only:
        # Wait for model to be available on HuggingFace before evaluation
        # This is necessary because there's a delay between upload completion and model availability
        check_revision = revision if not is_final_checkpoint else None
        print(f"[EVAL_PREP] Checking model availability before evaluation...")

        if not wait_for_model_availability(hf_model_path, check_revision, max_wait_seconds=300, check_interval=15):
            print(f"[EVAL_PREP] ERROR: Model not available on HuggingFace after waiting. Skipping evaluation.")
            print(f"[EVAL_PREP] Model: {hf_model_path}, Revision: {check_revision}")
            return

        print(f"[EVAL] Starting evaluation at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        # eval_model_args = f"pretrained={hf_model_path},dtype=bfloat16,parallelize=True,attn_implementation=flash_attention_2"
        eval_model_args = f"pretrained={hf_model_path},parallelize=True"
        if not is_final_checkpoint:
            eval_model_args = f"{eval_model_args},revision={revision}"

        eval_command = [
            "python",
            "-m",
            "lm_eval",
            "--model",
            "hf",
            "--model_args",
            eval_model_args,
            "--tasks",
            config.eval_tasks,
            "--batch_size",
            "64",
            "--wandb_args",
            f"project={config.wandb_project},entity={config.wandb_entity},name={config.experiment_name}_global_step{config.neox_checkpoint}",
            "--include_path",
            config.task_include_path,
        ]
        print(f"[EVAL] Running evaluation command: {' '.join(eval_command)}")
        print(f"[EVAL] ========== EVALUATION OUTPUT START ==========")
        sys.stdout.flush()
        eval_process = subprocess.Popen(
            eval_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1  # Line buffered
        )
        # Stream output line by line
        eval_output_lines = []
        for line in eval_process.stdout:
            print(f"[EVAL_OUT] {line}", end='')
            sys.stdout.flush()
            eval_output_lines.append(line)
        eval_process.wait()
        eval_returncode = eval_process.returncode
        print(f"[EVAL] ========== EVALUATION OUTPUT END ==========")
        print(f"[EVAL] Evaluation completed with exit code {eval_returncode} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        if eval_returncode != 0:
            print(f"[ERROR] Evaluation failed!")
            print(f"[ERROR] Last 50 lines of output:")
            for line in eval_output_lines[-50:]:
                print(f"[ERROR] {line}", end='')
        else:
            print(f"[EVAL] Evaluation successful")

        print("Evaluation complete.")


if __name__ == "__main__":
    print("\nConvert, Eval, and Upload (Enhanced Logging Version)\n")
    print(f"[SCRIPT_START] Script starting at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    print(f"[CONFIG] About to parse configuration at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        config = get_config()
        print(f"[CONFIG] Configuration parsed successfully at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[CONFIG] Arguments: experiment_name={getattr(config, 'experiment_name', 'NOT_SET')}")
        print(f"[CONFIG] Arguments: neox_checkpoint={getattr(config, 'neox_checkpoint', 'NOT_SET')}")
    except Exception as e:
        print(f"[ERROR] Configuration parsing failed: {e}")
        raise

    print(f"[MAIN_CALL] About to call main function at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        main(config)
        print(f"[SCRIPT_END] Script completed successfully at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    except Exception as e:
        print(f"[ERROR] Main function failed: {e}")
        print(f"[SCRIPT_END] Script failed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        raise

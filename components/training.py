"""Training Component.

Reusable inline training component modeled after the OSFT notebook flow.
- Configurable logging
- Optional Kubernetes connection (remote or in-cluster)
- PVC-based caches/checkpoints
- Dataset resolution (HF repo id, or local path)
- Basic metrics logging and checkpoint export
"""

from kfp import dsl
from typing import Optional


@dsl.component(
    base_image="quay.io/opendatahub/odh-training-th03-cuda128-torch28-py312-rhel9@sha256:84d05c5ef9dd3c6ff8173c93dca7e2e6a1cab290f416fb2c469574f89b8e6438",
    packages_to_install=[
        "kubernetes",
    ],
    task_config_passthroughs=[
        dsl.TaskConfigField.RESOURCES,
        dsl.TaskConfigField.KUBERNETES_TOLERATIONS,
        dsl.TaskConfigField.KUBERNETES_NODE_SELECTOR,
        dsl.TaskConfigField.KUBERNETES_AFFINITY,
        dsl.TaskConfigPassthrough(field=dsl.TaskConfigField.ENV, apply_to_task=True),
        dsl.TaskConfigPassthrough(field=dsl.TaskConfigField.KUBERNETES_VOLUMES, apply_to_task=True),
    ],
)
def train_model(
    # Workspace/PVC root (pass dsl.WORKSPACE_PATH_PLACEHOLDER at call site)
    pvc_path: str,
    # Outputs (no defaults)
    output_model: dsl.Output[dsl.Model],
    output_metrics: dsl.Output[dsl.Metrics],
    # Dataset input and optional remote artifact path via metadata (e.g., s3://...)
    dataset: dsl.Input[dsl.Dataset] = None,
    # Base model (HF ID or local path)
    training_base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    # Training algorithm selector
    training_algorithm: str = "OSFT",
    # OSFT parameters (prefixed with training_)
    training_unfreeze_rank_ratio: float = 0.25,
    training_effective_batch_size: int = 128,
    training_max_tokens_per_gpu: int = 64000,
    training_max_seq_len: int = 8192,
    training_learning_rate: Optional[float] = None,
    training_backend: str = "mini-trainer",
    training_target_patterns: str = "",
    training_seed: Optional[int] = None,
    training_use_liger: Optional[bool] = None,
    training_use_processed_dataset: Optional[bool] = None,
    training_unmask_messages: Optional[bool] = None,
    training_lr_scheduler: Optional[str] = None,
    training_lr_warmup_steps: Optional[int] = None,
    training_save_samples: Optional[int] = None,
    training_accelerate_full_state_at_epoch: Optional[bool] = None,
    training_lr_scheduler_kwargs: str = "",
    training_checkpoint_at_epoch: Optional[bool] = None,
    training_save_final_checkpoint: Optional[bool] = None,
    training_num_epochs: Optional[int] = None,
    training_data_output_dir: Optional[str] = None,
    # Env overrides: "KEY=VAL,KEY=VAL"
    training_envs: str = "",
    # Resource and runtime parameters (per worker/pod)
    training_resource_cpu_per_worker: str = "8",
    training_resource_gpu_per_worker: int = 1,
    training_resource_memory_per_worker: str = "32Gi",
    training_resource_num_procs_per_worker: int = 1,
    training_resource_num_workers: int = 1,
    training_metadata_labels: str = "",
    training_metadata_annotations: str = "",
    # KFP TaskConfig passthrough for volumes/env/resources, etc.
    kubernetes_config: dsl.TaskConfig = None,
) -> str:
    """Perform model training (inline) using PVC workspace and TrainingHub runtime.

    Args:
        pvc_path: Root of the workspace PVC for this run.
        dataset: Input dataset artifact (preferred). If not present, this component
            will attempt to load from a remote path specified in dataset.metadata.
            - metadata["artifact_path"]: remote dataset path (e.g., s3://..., https://..., or HF repo id)
            - metadata["pvc_dir"]: pre-staged PVC directory to use if present
        training_base_model: HuggingFace model ID to fine-tune (e.g., "Qwen/Qwen2.5-1.5B-Instruct").

        training_algorithm: Training algorithm ("OSFT" | "SFT"). OSFT adds continual learning support.
        training_effective_batch_size: Per-step batch size. Guidance:
            - 1 GPU: 16–32
            - 2 GPUs: 32–64
            - 4 GPUs: 64–128
        training_max_tokens_per_gpu: Token budget per GPU for memory mgmt.
        training_max_seq_len: Max sequence length (typical 2048–8192).
        training_learning_rate: Learning rate (typ. 1e-6 to 1e-4; 5e-6 is a good OSFT default).
        training_backend: Trainer backend variant (e.g., "mini-trainer").
        training_target_patterns: Comma-separated target modules/patterns (algorithm-specific).
        training_seed: Random seed for reproducibility.
        training_use_liger: Enable Liger kernel optimizations (image must include kernels).
        training_use_processed_dataset: Whether dataset is already processed.
        training_unmask_messages: Whether to unmask chat messages if applicable.
        training_lr_scheduler: LR scheduler ("cosine" | "linear" | "constant").
        training_lr_warmup_steps: LR warmup steps (0 for none).
        training_save_samples: Number of samples to save during SFT (optional).
        training_accelerate_full_state_at_epoch: Whether to save full Accelerate state at each epoch (optional).
        training_lr_scheduler_kwargs: Comma-delimited key=value string for scheduler kwargs
            (e.g., "num_cycles=1,num_warmup_steps=100").
        training_checkpoint_at_epoch: Save a checkpoint at each epoch boundary.
        training_save_final_checkpoint: Save the final model checkpoint.
        training_num_epochs: Number of epochs (1 = quick test; 3–5 = better convergence).
        training_data_output_dir: Optional secondary output directory on PVC.

        training_envs: Comma-separated env overrides ("KEY=VAL,KEY=VAL").
        training_resource_cpu_per_worker: CPU limit/request per worker (e.g., "8").
        training_resource_gpu_per_worker: GPUs per worker (e.g., 1). Typically equals num procs.
        training_resource_memory_per_worker: Memory per worker (e.g., "32Gi").
        training_resource_num_procs_per_worker: Processes (ranks) per worker (usually equals GPUs/worker).
        training_resource_num_workers: Total worker pods (1 = single-node; 2+ = multi-node).
        training_metadata_labels: Comma-separated labels ("k=v,k=v") for pod template.
        training_metadata_annotations: Comma-separated annotations ("k=v,k=v") for pod template.
        kubernetes_config: TaskConfig passthrough (volumes, mounts, env, resources, tolerations, etc.).

        output_model: Final model artifact copied to artifact store and PVC,
            with metadata set for downstream consumers:
            - model_name: the fine-tuned base model id
            - artifact_path: output_model.path (artifact store path)
            - pvc_model_dir: "<pvc_path>/final_model" (PVC directory path)
        output_metrics: Logged numeric metrics (floats), e.g.:
            - num_epochs, effective_batch_size, learning_rate, max_seq_len
            - max_tokens_per_gpu, unfreeze_rank_ratio (0 for SFT)

    OSFT func_args schema (passed to the trainer):

        model_path: Path to the model to fine-tune

        data_path: Path to the training data

        ckpt_output_dir: Directory to save checkpoints

        backend: Backend implementation to use (default: "instructlab-training")

        num_epochs: Number of training epochs

        effective_batch_size: Effective batch size for training

        learning_rate: Learning rate for training

        max_seq_len: Maximum sequence length

        max_tokens_per_gpu: Maximum tokens per GPU in a mini-batch (hard-cap for memory to avoid OOMs). Used to automatically calculate mini-batch size and gradient accumulation to maintain the desired effective_batch_size while staying within memory limits.

        data_output_dir: Directory to save processed data

        save_samples: Number of samples to save after training (0 disables saving based on sample count)

        warmup_steps: Number of warmup steps

        accelerate_full_state_at_epoch: Whether to save full state at epoch for automatic checkpoint resumption

        checkpoint_at_epoch: Whether to checkpoint at each epoch

    Returns:
        Status message string.
    """
    import os, sys, json, time, logging, re, subprocess
    from typing import Dict, List, Tuple, Optional as _Optional

    # ------------------------------
    # Logging configuration
    # ------------------------------
    def _setup_logger() -> logging.Logger:
        """Configure and return a logger for this component."""
        _logger = logging.getLogger("train_model")
        _logger.setLevel(logging.INFO)
        if not _logger.handlers:
            _ch = logging.StreamHandler(sys.stdout)
            _ch.setLevel(logging.INFO)
            _ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            _logger.addHandler(_ch)
        return _logger

    logger = _setup_logger()
    logger.info("Initializing training component")
    logger.info(f"pvc_path={pvc_path}, model_name={training_base_model}")

    # ------------------------------
    # Utility: latest checkpoint
    # ------------------------------
    def find_most_recent_checkpoint(checkpoints_root: str) -> _Optional[str]:
        if not os.path.isdir(checkpoints_root):
            return None
        latest: _Optional[Tuple[float, str]] = None
        for entry in os.listdir(checkpoints_root):
            full = os.path.join(checkpoints_root, entry)
            if os.path.isdir(full):
                try:
                    mtime = os.path.getmtime(full)
                except OSError:
                    continue
                if latest is None or mtime > latest[0]:
                    latest = (mtime, full)
        return latest[1] if latest else None

    # ------------------------------
    # Kubernetes connection
    # ------------------------------
    def _init_k8s_client() -> _Optional["k8s_client.ApiClient"]:
        """Initialize and return a Kubernetes client from env (server/token) or in-cluster/kubeconfig."""
        try:
            from kubernetes import client as k8s_client, config as k8s_config
            env_server = os.environ.get("KUBERNETES_SERVER_URL", "").strip()
            env_token = os.environ.get("KUBERNETES_AUTH_TOKEN", "").strip()
            if env_server and env_token:
                logger.info("Configuring Kubernetes client from env (KUBERNETES_SERVER_URL/_AUTH_TOKEN)")
                cfg = k8s_client.Configuration()
                cfg.host = env_server
                cfg.verify_ssl = False
                cfg.api_key = {"authorization": f"Bearer {env_token}"}
                k8s_client.Configuration.set_default(cfg)
                return k8s_client.ApiClient(cfg)
            logger.info("Configuring Kubernetes client in-cluster (or local kubeconfig)")
            try:
                k8s_config.load_incluster_config()
            except Exception:
                k8s_config.load_kube_config()
            return k8s_client.ApiClient()
        except Exception as _exc:
            logger.warning(f"Kubernetes client not initialized: {_exc}")
            return None

    _api_client = _init_k8s_client()

    # ------------------------------
    # Environment variables (defaults + overrides)
    # ------------------------------
    cache_root = os.path.join(pvc_path, ".cache", "huggingface")
    default_env: Dict[str, str] = {
        "XDG_CACHE_HOME": "/tmp",
        "TRITON_CACHE_DIR": "/tmp/.triton",
        "HF_HOME": "/tmp/.cache/huggingface",
        "HF_DATASETS_CACHE": os.path.join(cache_root, "datasets"),
        "TRANSFORMERS_CACHE": os.path.join(cache_root, "transformers"),
        "NCCL_DEBUG": "INFO",
    }

    def parse_kv_list(kv_str: str) -> Dict[str, str]:
        out: Dict[str, str] = {}
        if not kv_str:
            return out
        for item in kv_str.split(","):
            item = item.strip()
            if not item:
                continue
            if "=" not in item:
                raise ValueError(f"Invalid key=value item (expected key=value): {item}")
            k, v = item.split("=", 1)
            k = k.strip()
            v = v.strip()
            if not k:
                raise ValueError(f"Invalid key in key=value pair: {item}")
            out[k] = v
        return out

    def _configure_env(env_csv: str, base_env: Dict[str, str]) -> Dict[str, str]:
        """Merge base env with CSV overrides and export them to process env; return merged map."""
        overrides = parse_kv_list(env_csv)
        merged = {**base_env, **overrides}
        for ek, ev in merged.items():
            os.environ[ek] = ev
        logger.info(f"Env configured (keys): {sorted(list(merged.keys()))}")
        return merged

    merged_env = _configure_env(training_envs, default_env)

    # ------------------------------
    # Dataset resolution
    # ------------------------------
    from datasets import load_dataset, load_from_disk, Dataset
    import shutil

    resolved_dataset_dir = os.path.join(pvc_path, "dataset", "train")
    os.makedirs(resolved_dataset_dir, exist_ok=True)

    def is_local_path(p: str) -> bool:
        return bool(p) and os.path.exists(p)

    def looks_like_url(p: str) -> bool:
        return p.startswith("s3://") or p.startswith("http://") or p.startswith("https://")

    def _resolve_dataset(input_dataset: _Optional[dsl.Input[dsl.Dataset]], out_dir: str) -> None:
        """Resolve dataset with preference: existing PVC dir > input artifact > remote artifact/HF > default.
        Remote path is read from input_dataset.metadata['artifact_path'] if present. If metadata['pvc_dir'] exists, prefer it.
        """
        # 0) If already present (e.g., staged by prior step), keep it
        if os.path.isdir(out_dir) and any(os.scandir(out_dir)):
            logger.info(f"Using existing dataset at {out_dir}")
            return
        # 1) Input artifact
        if input_dataset and getattr(input_dataset, "path", None) and os.path.exists(input_dataset.path):
            logger.info(f"Copying input dataset from {input_dataset.path} to {out_dir}")
            shutil.copytree(input_dataset.path, out_dir, dirs_exist_ok=True)
            return
        # 2) Remote artifact (S3/HTTP) or HF repo id
        rp = ""
        try:
            if input_dataset and hasattr(input_dataset, "metadata") and isinstance(input_dataset.metadata, dict):
                pvc_dir = (input_dataset.metadata.get("pvc_dir") or "").strip()
                if pvc_dir and os.path.isdir(pvc_dir) and any(os.scandir(pvc_dir)):
                    logger.info(f"Using pre-staged PVC dataset at {pvc_dir}")
                    shutil.copytree(pvc_dir, out_dir, dirs_exist_ok=True)
                    return
                rp = (input_dataset.metadata.get("artifact_path") or "").strip()
        except Exception:
            rp = ""
        if rp:
            if looks_like_url(rp):
                logger.info(f"Attempting to load remote dataset from {rp}")
                # Try a few common formats via datasets library
                ext = rp.lower()
                try:
                    if ext.endswith(".json") or ext.endswith(".jsonl"):
                        ds: Dataset = load_dataset("json", data_files=rp, split="train")
                    elif ext.endswith(".parquet"):
                        ds: Dataset = load_dataset("parquet", data_files=rp, split="train")
                    else:
                        raise ValueError(
                            "Unsupported remote dataset format. Provide a JSON/JSONL/PARQUET file or a HF dataset repo id."
                        )
                    ds.save_to_disk(out_dir)
                    return
                except Exception as e:
                    raise ValueError(f"Failed to load remote dataset from {rp}: {e}")
            else:
                # Treat as HF dataset repo id
                logger.info(f"Assuming HF dataset repo id: {rp}")
                ds: Dataset = load_dataset(rp, split="train")
                ds.save_to_disk(out_dir)
                return
        # 3) Default fallback (Table-GPT)
        logger.info("No dataset provided. Falling back to 'LipengCS/Table-GPT'")
        ds: Dataset = load_dataset("LipengCS/Table-GPT", "All", split="train")
        ds.save_to_disk(out_dir)

    _resolve_dataset(dataset, resolved_dataset_dir)

    # Export dataset to JSONL so downstream trainer reads a plain JSONL file
    jsonl_path = os.path.join(resolved_dataset_dir, "train.jsonl")
    try:
        # Try loading from the saved HF dataset on disk and export to JSONL
        ds_on_disk = load_from_disk(resolved_dataset_dir)
        # Handle DatasetDict vs Dataset
        train_split = ds_on_disk["train"] if isinstance(ds_on_disk, dict) else ds_on_disk
        try:
            # Newer datasets supports native JSON export
            train_split.to_json(jsonl_path, lines=True)
            logger.info(f"Wrote JSONL to {jsonl_path} via to_json")
        except AttributeError:
            # Manual JSONL write
            import json as _json
            with open(jsonl_path, "w") as _f:
                for _rec in train_split:
                    _f.write(_json.dumps(_rec, ensure_ascii=False) + "\n")
            logger.info(f"Wrote JSONL to {jsonl_path} via manual dump")
    except Exception as _e:
        logger.warning(f"Failed to export JSONL dataset at {resolved_dataset_dir}: {_e}")
        # Leave jsonl_path as default; downstream will fallback to directory if file not present
    def _resolve_model_path(model_ref: str) -> str:
        """Resolve model reference to local path; supports oci:// registry refs via ORAS."""
        if isinstance(model_ref, str) and model_ref.startswith("oci://"):
            ref = model_ref[len("oci://") :]
            model_dir = os.path.join(pvc_path, "model")
            os.makedirs(model_dir, exist_ok=True)
            oras_bin = "/tmp/oras"
            # Ensure DOCKER_CONFIG is a writable directory so ORAS can persist creds
            try:
                docker_cfg_root = "/tmp/.docker"
                os.makedirs(docker_cfg_root, exist_ok=True)
                os.environ["DOCKER_CONFIG"] = docker_cfg_root
                logger.info(f"DOCKER_CONFIG set to {docker_cfg_root}")
            except Exception as _e:
                logger.warning(f"Failed to prepare DOCKER_CONFIG at {docker_cfg_root}: {_e}")
            try:
                if not os.path.exists(oras_bin):
                    import tarfile
                    import urllib.request
                    oras_url = "https://github.com/oras-project/oras/releases/download/v1.2.0/oras_1.2.0_linux_amd64.tar.gz"
                    tar_path = "/tmp/oras.tar.gz"
                    logger.info(f"Downloading ORAS from {oras_url}")
                    urllib.request.urlretrieve(oras_url, tar_path)
                    with tarfile.open(tar_path, "r:gz") as tar:
                        member = next((m for m in tar.getmembers() if m.name == "oras"), None)
                        if member is None:
                            raise RuntimeError("oras binary not found in archive")
                        tar.extract(member, path="/tmp")
                    os.chmod(oras_bin, 0o755)
            except Exception as e:
                logger.error(f"Failed to prepare ORAS binary: {e}")
                raise
            # Optional login (otherwise ORAS uses ~/.docker/config.json or $DOCKER_CONFIG)
            registry = ref.split("/")[0]
            user = os.environ.get("OCI_REGISTRY_USERNAME", "").strip()
            pwd = os.environ.get("OCI_REGISTRY_PASSWORD", "").strip()
            if user and pwd:
                try:
                    logger.info(f"Attempting ORAS login to {registry} with provided username")
                    res = subprocess.run(
                        [oras_bin, "login", registry, "-u", user, "--password-stdin"],
                        input=pwd,
                        text=True,
                        capture_output=True,
                        check=False,
                    )
                    if res.returncode != 0:
                        logger.error(f"ORAS login stderr: {res.stderr}")
                        res.check_returncode()
                    # Verify that config.json exists and contains the registry entry
                    cfg_path = os.path.join(os.environ.get("DOCKER_CONFIG", "/tmp/.docker"), "config.json")
                    if os.path.isfile(cfg_path):
                        try:
                            with open(cfg_path) as _cf:
                                _cfg = json.load(_cf)
                            auths = sorted(list((_cfg.get("auths") or {}).keys()))
                            logger.info(f"ORAS login persisted; config auths: {auths}")
                        except Exception as _e:
                            logger.warning(f"Failed to read DOCKER_CONFIG at {cfg_path}: {_e}")
                    else:
                        logger.warning(f"DOCKER_CONFIG missing config.json at {cfg_path}")
                except subprocess.CalledProcessError as e:
                    logger.error(f"ORAS login failed: {e}")
                    raise
            # Copy to OCI layout and extract layers into model_dir
            try:
                layout_dir = os.path.join(pvc_path, "model-oci")
                os.makedirs(layout_dir, exist_ok=True)
                logger.info(f"Copying OCI artifact {ref} to OCI layout: {layout_dir}")
                res = subprocess.run(
                    [oras_bin, "copy", "-v", ref, "--to-oci-layout", layout_dir],
                    text=True,
                    check=False,
                )
                if res.returncode != 0:
                    logger.error("ORAS copy failed (see logs above for details)")
                    res.check_returncode()
                # Extract tar layers from blobs into model_dir
                blobs_dir = os.path.join(layout_dir, "blobs", "sha256")
                extracted_any = False
                if os.path.isdir(blobs_dir):
                    import tarfile
                    for fname in os.listdir(blobs_dir):
                        fpath = os.path.join(blobs_dir, fname)
                        # Attempt to extract tar/gzip layers; skip non-archives
                        try:
                            with tarfile.open(fpath, mode="r:*") as tf:
                                tf.extractall(model_dir)
                                extracted_any = True
                        except tarfile.ReadError:
                            continue
                        except Exception as _e:
                            logger.warning(f"Failed to extract blob {fname}: {_e}")
                logger.info(f"Pulled OCI artifact into {model_dir} (extracted_any={extracted_any})")
                # After extraction, show a compact directory tree and try to locate a HF model subdir
                _log_dir_tree(model_dir, max_depth=3, max_entries=800)
                candidate = _find_hf_model_dir(model_dir)
                if candidate and candidate != model_dir:
                    logger.info(f"Detected HuggingFace model directory: {candidate}")
                    return candidate
            except subprocess.CalledProcessError as e:
                logger.error(f"ORAS copy failed: {e}")
                raise
            return model_dir
        return model_ref

    def _summarize_model_dir(model_dir: str) -> None:
        """Log a brief summary of model directory contents to aid debugging."""
        try:
            if not (model_dir and os.path.isdir(model_dir)):
                logger.info(f"Model path is not a directory: {model_dir}")
                return
            entries = os.listdir(model_dir)
            preview = sorted(entries)[:20]
            logger.info(f"Model dir '{model_dir}' entries (first 20): {preview}")
        except Exception as _e:
            logger.warning(f"Failed to list model directory {model_dir}: {_e}")

    def _log_dir_tree(root: str, max_depth: int = 3, max_entries: int = 800) -> None:
        """Log a compact tree view of a directory for debugging.
        Limits traversal by depth and total entries to keep logs manageable.
        """
        try:
            if not (root and os.path.isdir(root)):
                logger.info(f"(tree) Path is not a directory: {root}")
                return
            logger.info(f"(tree) {root} (max_depth={max_depth}, max_entries={max_entries})")
            total_printed = 0
            root_depth = root.rstrip(os.sep).count(os.sep)
            for dirpath, dirnames, filenames in os.walk(root):
                current_depth = dirpath.rstrip(os.sep).count(os.sep) - root_depth
                if current_depth >= max_depth:
                    # Do not descend further
                    dirnames[:] = []
                indent = "  " * current_depth
                # Print directory
                logger.info(f"(tree){indent}{os.path.basename(dirpath) or dirpath}/")
                total_printed += 1
                if total_printed >= max_entries:
                    logger.info("(tree) ... truncated ...")
                    return
                # Print a few files
                for fname in sorted(filenames)[:50]:
                    logger.info(f"(tree){indent}  {fname}")
                    total_printed += 1
                    if total_printed >= max_entries:
                        logger.info("(tree) ... truncated ...")
                        return
        except Exception as _e:
            logger.warning(f"Failed to render directory tree for {root}: {_e}")

    def _find_hf_model_dir(root: str, scan_limit: int = 5000) -> _Optional[str]:
        """Recursively search for a directory that looks like a HF model repo.
        Heuristics: must contain config.json and at least one weights and tokenizer candidate.
        """
        try:
            if not (root and os.path.isdir(root)):
                return None
            weight_candidates = {
                "pytorch_model.bin",
                "pytorch_model.bin.index.json",
                "model.safetensors",
                "model.safetensors.index.json",
            }
            tokenizer_candidates = {
                "tokenizer.json",
                "tokenizer.model",
            }
            scanned = 0
            for dirpath, _dirnames, filenames in os.walk(root):
                scanned += 1
                if scanned > scan_limit:
                    logger.info(f"HF model autodiscovery scan limit reached at {scan_limit} directories")
                    break
                fn = set(filenames)
                if "config.json" in fn and (fn & weight_candidates) and (fn & tokenizer_candidates):
                    return dirpath
            return None
        except Exception as _e:
            logger.warning(f"HF model autodiscovery failed under {root}: {_e}")
            return None

    def _validate_model_layout(model_dir: str, backend_name: str) -> None:
        """Best-effort checks for a Hugging Face style model directory."""
        if not (model_dir and os.path.isdir(model_dir)):
            # Nothing to validate for HF IDs or missing dirs
            return
        required = [
            "config.json",
        ]
        weight_candidates = [
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
            "model.safetensors",
            "model.safetensors.index.json",
        ]
        tokenizer_candidates = [
            "tokenizer.json",
            "tokenizer.model",
        ]
        missing = [p for p in required if not os.path.exists(os.path.join(model_dir, p))]
        has_weights = any(os.path.exists(os.path.join(model_dir, p)) for p in weight_candidates)
        has_tokenizer = any(os.path.exists(os.path.join(model_dir, p)) for p in tokenizer_candidates)
        if missing:
            logger.warning(f"Model dir missing required files {missing} for backend={backend_name}")
        if not has_weights:
            logger.warning(f"Model dir appears to lack weights files ({weight_candidates}) for backend={backend_name}")
        if not has_tokenizer:
            logger.warning(f"Model dir appears to lack tokenizer files ({tokenizer_candidates}) for backend={backend_name}")
        # Optional: try to parse config
        try:
            import json as _json
            with open(os.path.join(model_dir, "config.json")) as _f:
                cfg = _json.load(_f)
            model_type = cfg.get("model_type")
            logger.info(f"Detected model_type in config.json: {model_type}")
        except Exception:
            pass

    # ------------------------------
    # Training (placeholder for TrainingHubTrainer)
    # ------------------------------
    checkpoints_dir = os.path.join(pvc_path, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Wire in TrainingHubTrainer (modularized steps)
    try:
        from kubeflow.trainer import TrainerClient
        from kubeflow.trainer.rhai import TrainingHubAlgorithms, TrainingHubTrainer
        from kubeflow_trainer_api import models as _th_models  # noqa: F401
        from kubeflow.common.types import KubernetesBackendConfig
        from kubeflow.trainer.options.kubernetes import (
            PodTemplateOverrides,
            PodTemplateOverride,
            PodSpecOverride,
            ContainerOverride,
        )

        if _api_client is None:
            raise RuntimeError("Kubernetes API client is not initialized")

        backend_cfg = KubernetesBackendConfig(client_configuration=_api_client.configuration)
        client = TrainerClient(backend_cfg)

        def _select_runtime(_client) -> object:
            """Return the 'training-hub' runtime from Trainer backend."""
            for rt in _client.list_runtimes():
                if getattr(rt, "name", "") == "training-hub":
                    logger.info(f"Found runtime: {rt}")
                    return rt
            raise RuntimeError("Training runtime 'training-hub' not found")

        th_runtime = _select_runtime(client)

        # Build training parameters (aligned to OSFT/SFT)
        parsed_target_patterns = [p.strip() for p in training_target_patterns.split(",") if p.strip()] if training_target_patterns else None
        parsed_lr_sched_kwargs = None
        if training_lr_scheduler_kwargs:
            try:
                items = [s.strip() for s in training_lr_scheduler_kwargs.split(",") if s.strip()]
                kv: Dict[str, str] = {}
                for item in items:
                    if "=" not in item:
                        raise ValueError(
                            f"Invalid scheduler kwargs segment '{item}'. Expected key=value."
                        )
                    key, value = item.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    if not key:
                        raise ValueError("Empty key in training_lr_scheduler_kwargs")
                    kv[key] = value
                parsed_lr_sched_kwargs = kv
            except Exception as e:
                raise ValueError(f"Invalid training_lr_scheduler_kwargs format: {e}")

        resolved_model_path = _resolve_model_path(training_base_model)
        _summarize_model_dir(resolved_model_path if isinstance(resolved_model_path, str) else "")
        _validate_model_layout(resolved_model_path if isinstance(resolved_model_path, str) else "", training_backend)

        def _build_params() -> Dict[str, object]:
            """Build OSFT/SFT parameter set for TrainingHub."""
            base = {
                "model_path": resolved_model_path,
                # Prefer JSONL export when available; fallback to resolved directory
                "data_path": jsonl_path if os.path.exists(jsonl_path) else resolved_dataset_dir,
                "effective_batch_size": int(training_effective_batch_size if training_effective_batch_size is not None else 128),
                "max_tokens_per_gpu": int(training_max_tokens_per_gpu),
                "max_seq_len": int(training_max_seq_len if training_max_seq_len is not None else 8192),
                "learning_rate": float(training_learning_rate if training_learning_rate is not None else 5e-6),
                "backend": training_backend,
                "ckpt_output_dir": checkpoints_dir,
                "data_output_dir": training_data_output_dir or os.path.join(checkpoints_dir, "_internal_data_processing"),
                "target_patterns": parsed_target_patterns or [],
                "seed": int(training_seed) if training_seed is not None else 42,
                "use_liger": bool(training_use_liger) if training_use_liger is not None else False,
                "use_processed_dataset": bool(training_use_processed_dataset) if training_use_processed_dataset is not None else False,
                "unmask_messages": bool(training_unmask_messages) if training_unmask_messages is not None else False,
                "lr_scheduler": training_lr_scheduler or "constant",
                "warmup_steps": int(training_lr_warmup_steps) if training_lr_warmup_steps is not None else 0,
                "save_samples": int(training_save_samples) if training_save_samples is not None else 0,
                "accelerate_full_state_at_epoch": bool(training_accelerate_full_state_at_epoch) if training_accelerate_full_state_at_epoch is not None else False,
                "lr_scheduler_kwargs": parsed_lr_sched_kwargs or {},
                "checkpoint_at_epoch": bool(training_checkpoint_at_epoch) if training_checkpoint_at_epoch is not None else False,
                "save_final_checkpoint": bool(training_save_final_checkpoint) if training_save_final_checkpoint is not None else False,
                "num_epochs": int(training_num_epochs) if training_num_epochs is not None else 1,
            }
            if (training_algorithm or "").strip().upper() == "OSFT":
                base["unfreeze_rank_ratio"] = float(training_unfreeze_rank_ratio)
            return base

        params = _build_params()

        # Algorithm selection: include OSFT-only param when applicable
        algo_value = TrainingHubAlgorithms.OSFT if (training_algorithm or "").strip().upper() != "SFT" else TrainingHubAlgorithms.SFT
        if algo_value == TrainingHubAlgorithms.OSFT:
            params["unfreeze_rank_ratio"] = float(training_unfreeze_rank_ratio)

        # Build volumes and mounts (from passthrough only); do not inject env via pod overrides
        # Cluster policy forbids env in podTemplateOverrides; use trainer.env for container env

        volumes = []
        volume_mounts = []
        if kubernetes_config and getattr(kubernetes_config, "volumes", None):
            volumes.extend(kubernetes_config.volumes)
        if kubernetes_config and getattr(kubernetes_config, "volume_mounts", None):
            volume_mounts.extend(kubernetes_config.volume_mounts)

        # Container resources are not overridden here; rely on runtime defaults or future API support

        # Parse metadata labels/annotations for Pod template
        tpl_labels = parse_kv_list(training_metadata_labels)
        tpl_annotations = parse_kv_list(training_metadata_annotations)

        def _build_pod_spec_override() -> PodSpecOverride:
            """Return PodSpecOverride with mounts, envs, resources, and scheduling hints."""
            return PodSpecOverride(
                volumes=volumes,
                containers=[
                    ContainerOverride(
                        name="node",
                        volume_mounts=volume_mounts,
                    )
                ],
                # node_selector=(kubernetes_config.node_selector if kubernetes_config and getattr(kubernetes_config, "node_selector", None) else None),
                # tolerations=(kubernetes_config.tolerations if kubernetes_config and getattr(kubernetes_config, "tolerations", None) else None),
            )

        job_name = client.train(
            trainer=TrainingHubTrainer(
                algorithm=TrainingHubAlgorithms.OSFT if (training_algorithm or "").strip().upper() != "SFT" else TrainingHubAlgorithms.SFT,
                func_args=params,
                packages_to_install=[],
                # Pass environment variables via Trainer spec (allowed by backend/webhook)
                env=dict(merged_env),
            ),
            options=[
                PodTemplateOverrides(
                    PodTemplateOverride(
                        target_jobs=["node"],
                        metadata={"labels": tpl_labels, "annotations": tpl_annotations} if (tpl_labels or tpl_annotations) else None,
                        spec=_build_pod_spec_override(),
                        # numProcsPerWorker=training_resource_num_procs_per_worker,
                        # numWorkers=training_resource_num_workers,
                    )
                )
            ],
            runtime=th_runtime,
        )
        logger.info(f"Submitted TrainingHub job: {job_name}")
        try:
            # Wait for the job to start running, then wait for completion or failure.
            client.wait_for_job_status(name=job_name, status={"Running"}, timeout=300)
            client.wait_for_job_status(name=job_name, status={"Complete", "Failed"}, timeout=1800)
            job = client.get_job(name=job_name)
            if getattr(job, "status", None) == "Failed":
                logger.error("Training job failed")
                raise RuntimeError(f"Training job failed with status: {job.status}")
            elif getattr(job, "status", None) == "Complete":
                logger.info("Training job completed successfully")
            else:
                logger.error(f"Unexpected training job status: {job.status}")
                raise RuntimeError(f"Training job ended with unexpected status: {job.status}")
        except Exception as e:
            logger.warning(f"Training job monitoring failed: {e}")
    except Exception as e:
        logger.error(f"TrainingHubTrainer execution failed: {e}")
        raise

    # ------------------------------
    # Metrics (basic hyperparameters)
    # ------------------------------
    def _log_basic_metrics() -> None:
        output_metrics.log_metric("num_epochs", float(params.get("num_epochs") or 1))
        output_metrics.log_metric("effective_batch_size", float(params.get("effective_batch_size") or 128))
        output_metrics.log_metric("learning_rate", float(params.get("learning_rate") or 5e-6))
        output_metrics.log_metric("max_seq_len", float(params.get("max_seq_len") or 8192))
        output_metrics.log_metric("max_tokens_per_gpu", float(params.get("max_tokens_per_gpu") or 0))
        output_metrics.log_metric("unfreeze_rank_ratio", float(params.get("unfreeze_rank_ratio") or 0))

    _log_basic_metrics()

    # ------------------------------
    # Export most recent checkpoint as model artifact (artifact store) and PVC
    # ------------------------------
    def _persist_and_annotate() -> None:
        """Copy latest checkpoint to PVC and artifact store, then annotate output metadata."""
        latest = find_most_recent_checkpoint(checkpoints_dir)
        if not latest:
            raise RuntimeError(f"No checkpoints found under {checkpoints_dir}")
        # PVC copy
        pvc_dir = os.path.join(pvc_path, "final_model")
        try:
            if os.path.exists(pvc_dir):
                shutil.rmtree(pvc_dir)
            shutil.copytree(latest, pvc_dir, dirs_exist_ok=True)
            logger.info(f"Copied checkpoint to PVC dir: {pvc_dir}")
        except Exception as _e:
            logger.warning(f"Failed to copy model to PVC dir {pvc_dir}: {_e}")
        # Artifact copy
        output_model.name = f"{training_base_model}-checkpoint"
        shutil.copytree(latest, output_model.path, dirs_exist_ok=True)
        logger.info(f"Exported checkpoint from {latest} to artifact path {output_model.path}")
        # Metadata
        try:
            output_model.metadata["model_name"] = training_base_model
            output_model.metadata["artifact_path"] = output_model.path
            output_model.metadata["pvc_model_dir"] = pvc_dir
            logger.info("Annotated output_model metadata with pvc/artifact locations")
        except Exception as _e:
            logger.warning(f"Failed to set output_model metadata: {_e}")

    _persist_and_annotate()

    return "training completed"


if __name__ == "__main__":
    from kfp import compiler

    compiler.Compiler().compile(
        train_model,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
    print("Compiled: train_model_component.yaml")


"""
Backend‑agnostic training orchestrator.

This is the top‑level entry point called by ``__main__.py``.  It:

1. Selects the correct backend (CUDA / MLX) from the config.
2. Drives the full pipeline: load → prepare → tokenise → train → save →
   (merge) → (export).
3. Emits IPC stage transitions at every phase boundary.
4. Wraps all exceptions in user‑friendly error messages.
"""

from __future__ import annotations

import logging
import os
import sys
import traceback
from typing import Any

from engine.ipc import (
    elapsed,
    emit_checkpoint,
    emit_done,
    emit_error,
    emit_log,
    emit_progress,
    emit_stage,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Plain‑language error translation
# ---------------------------------------------------------------------------

_ERROR_PATTERNS: list[tuple[str, str]] = [
    (
        "out of memory",
        "GPU ran out of memory. Try reducing batch_size, seq_length, or LoRA rank.",
    ),
    (
        "cuda",
        "A CUDA error occurred. Make sure your NVIDIA drivers and CUDA toolkit are up to date.",
    ),
    (
        "bitsandbytes",
        "bitsandbytes failed. Ensure it is installed correctly for your platform.",
    ),
    (
        "tokenizer",
        "Tokenizer error — the model may require a specific tokenizer. Check the model card.",
    ),
    (
        "connection",
        "Network error while downloading the model. Check your internet connection.",
    ),
    (
        "gated",
        "This model is gated on Hugging Face. Provide a valid HF token with accepted access.",
    ),
    (
        "401",
        "Authentication failed. Your Hugging Face token may be invalid or expired.",
    ),
    (
        "403",
        "Access denied. You may need to accept the model's licence on huggingface.co.",
    ),
]


def _friendly_message(exc: Exception) -> tuple[str, str]:
    """Return (plain_message, technical_detail) for an exception."""
    detail = traceback.format_exc()
    msg_lower = str(exc).lower()
    for pattern, friendly in _ERROR_PATTERNS:
        if pattern in msg_lower:
            return friendly, detail
    return f"Training failed: {exc}", detail


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run(config: dict[str, Any]) -> None:  # noqa: C901 — orchestrator is inherently complex
    """Execute the full training pipeline described by *config*.

    This function **never raises** — all errors are caught and emitted as
    IPC error events.
    """
    try:
        _run_inner(config)
    except KeyboardInterrupt:
        emit_error("Training cancelled by user.", recoverable=False)
        sys.exit(130)
    except Exception as exc:
        msg, detail = _friendly_message(exc)
        emit_error(msg, detail=detail, recoverable=False)
        sys.exit(1)


def _run_inner(config: dict[str, Any]) -> None:
    """Inner implementation — may raise."""
    backend_name: str = config["backend"]
    model_id: str = config["model_id"]
    hf_token: str | None = config.get("hf_token")
    dataset_path: str = config["dataset_path"]
    output_dir: str = config["output_dir"]
    checkpoint_dir: str = config.get("checkpoint_dir", os.path.join(output_dir, "checkpoints"))
    resume_from: str | None = config.get("resume_from")
    seq_length: int = config.get("seq_length", 512)
    lora_config: dict[str, Any] = config.get("lora", {})
    training_config: dict[str, Any] = config.get("training", {})
    do_merge: bool = config.get("merge", False)
    merge_output_path: str = config.get(
        "merge_output_path", os.path.join(output_dir, "merged")
    )
    export_config: dict[str, Any] = config.get("export", {})

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ---- 1. Select backend -------------------------------------------
    emit_stage("initialising", "Selecting training backend…")
    backend = _create_backend(backend_name)

    # ---- 2. Load model -----------------------------------------------
    emit_stage("loading_model", f"Loading model {model_id}…")
    backend.load_model(model_id, hf_token, config)

    # ---- 3. Attach LoRA adapters -------------------------------------
    emit_stage("preparing_lora", "Attaching LoRA adapters…")
    backend.prepare_qlora(lora_config)

    # Log trainable param info.
    info = backend.get_trainable_params_info()
    emit_log(
        "info",
        f"Trainable parameters: {info['trainable_params']:,} "
        f"({info['trainable_pct']:.2f}% of {info['total_params']:,})",
    )

    # ---- 4. Load & tokenise dataset ----------------------------------
    emit_stage("loading_dataset", "Loading and tokenising dataset…")
    from engine.dataset import load_and_tokenize

    dataset = load_and_tokenize(
        dataset_path=dataset_path,
        tokenizer=backend.tokenizer,
        seq_length=seq_length,
        backend=backend_name,
    )

    # ---- 5. Train ----------------------------------------------------
    emit_stage("training", "Training started…")

    def _progress_cb(**kwargs: Any) -> None:
        emit_progress(
            step=kwargs["step"],
            total_steps=kwargs["total_steps"],
            loss=kwargs["loss"],
            lr=kwargs["lr"],
            tokens_per_sec=kwargs["tokens_per_sec"],
            eta_seconds=kwargs["eta_seconds"],
        )

    def _checkpoint_cb(**kwargs: Any) -> None:
        emit_checkpoint(path=kwargs["path"], step=kwargs["step"])

    adapter_path = backend.train(
        dataset=dataset,
        training_config=training_config,
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        resume_from=resume_from,
        progress_callback=_progress_cb,
        checkpoint_callback=_checkpoint_cb,
    )

    # ---- 6. (Optional) Merge adapter ---------------------------------
    if do_merge:
        emit_stage("merging", "Merging LoRA adapter into base model…")
        from engine.merge import merge_adapter

        merge_adapter(
            base_model_id=model_id,
            adapter_path=adapter_path,
            output_path=merge_output_path,
            hf_token=hf_token,
            backend=backend_name,
        )

    # ---- 7. (Optional) Export to GGUF --------------------------------
    if export_config.get("enabled", False):
        emit_stage("exporting", "Exporting to GGUF…")
        from engine.export_gguf import export_gguf

        source_path = merge_output_path if do_merge else adapter_path
        export_gguf(
            model_path=source_path,
            output_path=export_config.get(
                "gguf_output_path", os.path.join(output_dir, "model.gguf")
            ),
            quant_type=export_config.get("quant_type", "Q4_K_M"),
        )

    # ---- 8. Done! ----------------------------------------------------
    emit_done(adapter_path=adapter_path, elapsed_seconds=elapsed())
    emit_log("info", f"Training complete in {elapsed():.1f}s.")


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------


def _create_backend(name: str) -> Any:
    """Instantiate the requested training backend."""
    if name == "cuda":
        from engine.backend_cuda import CUDABackend

        return CUDABackend()
    elif name == "mlx":
        from engine.backend_mlx import MLXBackend

        return MLXBackend()
    elif name == "cpu-only":
        from engine.backend_cpu import CPUBackend

        return CPUBackend()
    else:
        raise ValueError(
            f"Unknown backend '{name}'. Supported: cuda, mlx, cpu-only."
        )

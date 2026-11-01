"""
LoRA adapter merge logic.

Loads the base model in full precision (fp16 / bf16), applies the trained
LoRA adapter, calls ``merge_and_unload()``, and saves the merged model in
safetensors format — ready for GGUF conversion.
"""

from __future__ import annotations

import gc
import logging
import os

from engine.ipc import emit_log, emit_stage

logger = logging.getLogger(__name__)


def merge_adapter(
    base_model_id: str,
    adapter_path: str,
    output_path: str,
    hf_token: str | None = None,
    backend: str = "cuda",
) -> str:
    """Merge a LoRA adapter into its base model and save.

    Parameters
    ----------
    base_model_id:
        Hugging Face model repo id or local path of the **base** model.
    adapter_path:
        Path to the directory containing LoRA adapter weights.
    output_path:
        Where to write the merged full‑precision model.
    hf_token:
        Optional HF token for gated models.
    backend:
        ``"cuda"`` or ``"mlx"`` — determines the merge strategy.

    Returns
    -------
    str
        ``output_path`` on success.
    """
    os.makedirs(output_path, exist_ok=True)

    if backend == "mlx":
        return _merge_mlx(base_model_id, adapter_path, output_path, hf_token)
    else:
        return _merge_cuda(base_model_id, adapter_path, output_path, hf_token)


# ---------------------------------------------------------------------------
# CUDA merge (transformers + peft)
# ---------------------------------------------------------------------------


def _merge_cuda(
    base_model_id: str,
    adapter_path: str,
    output_path: str,
    hf_token: str | None,
) -> str:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    emit_log("info", f"Loading base model {base_model_id} in fp16 for merge…")

    # Load in fp16 on CPU to avoid GPU OOM during merge.
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        token=hf_token,
    )

    emit_log("info", f"Loading adapter from {adapter_path}…")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    emit_log("info", "Merging LoRA weights…")
    model = model.merge_and_unload()

    emit_log("info", f"Saving merged model to {output_path}…")
    model.save_pretrained(output_path, safe_serialization=True)

    # Save tokenizer alongside.
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_id, token=hf_token
        )
        tokenizer.save_pretrained(output_path)
    except Exception as exc:
        emit_log("warning", f"Could not save tokenizer: {exc}")

    # Cleanup.
    del model, base_model
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    emit_log("info", "Merge complete.")
    return output_path


# ---------------------------------------------------------------------------
# MLX merge (mlx_lm fuse)
# ---------------------------------------------------------------------------


def _merge_mlx(
    base_model_id: str,
    adapter_path: str,
    output_path: str,
    hf_token: str | None,
) -> str:
    """Merge using ``mlx_lm.fuse_model`` and convert back to HF format."""
    emit_log("info", "Fusing LoRA adapters (MLX)…")

    try:
        from mlx_lm import fuse_model  # type: ignore[import-untyped]
    except ImportError:
        from mlx_lm.utils import fuse_model  # type: ignore[import-untyped]

    fuse_model(
        model=base_model_id,
        adapter_path=adapter_path,
        save_path=output_path,
    )

    # Also ensure tokenizer is present.
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=hf_token)
        tokenizer.save_pretrained(output_path)
    except Exception as exc:
        emit_log("warning", f"Could not save tokenizer alongside merged model: {exc}")

    emit_log("info", f"Merged model saved to {output_path}")
    return output_path

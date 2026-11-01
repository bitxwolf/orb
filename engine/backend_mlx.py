"""
MLX training backend for Apple Silicon Macs.

Uses ``mlx-lm`` for model loading, LoRA injection, and training.  The HF →
MLX format conversion is handled transparently so the user can specify a
regular Hugging Face model ID.
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from engine.backend_base import TrainingBackend
from engine.ipc import emit_log

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class MLXBackend(TrainingBackend):
    """QLoRA‑style fine‑tuning on Apple Silicon using MLX."""

    def __init__(self) -> None:
        self.model: Any = None
        self.tokenizer: Any = None
        self._model_path: str = ""
        self._base_model_id: str = ""

    # ------------------------------------------------------------------ #
    # load_model
    # ------------------------------------------------------------------ #

    def load_model(
        self,
        model_id: str,
        hf_token: str | None,
        config: dict[str, Any],
    ) -> None:
        """Load a model via ``mlx_lm.load()``.

        If the model is a standard HF repo, ``mlx_lm`` handles the
        conversion from safetensors → MLX weight format automatically.
        """
        import mlx_lm  # type: ignore[import-untyped]

        self._base_model_id = model_id

        emit_log("info", f"Loading model {model_id} for MLX…")

        # mlx_lm.load returns (model, tokenizer).  It accepts HF repo ids
        # and will download + convert on the fly.
        tokenizer_config: dict[str, Any] = {}
        if hf_token:
            tokenizer_config["token"] = hf_token

        self.model, self.tokenizer = mlx_lm.load(
            model_id,
            tokenizer_config=tokenizer_config,
        )

        self._model_path = model_id

        # Ensure pad token.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        emit_log("info", "Model loaded on MLX (Apple Silicon).")

    # ------------------------------------------------------------------ #
    # prepare_qlora
    # ------------------------------------------------------------------ #

    def prepare_qlora(self, lora_config: dict[str, Any]) -> None:
        """Attach LoRA adapters using ``mlx_lm``'s utilities."""
        import mlx.nn as nn  # type: ignore[import-untyped]
        from mlx_lm.tuner.utils import apply_lora_layers  # type: ignore[import-untyped]

        assert self.model is not None, "Call load_model() first."

        rank = lora_config.get("rank", 16)
        alpha = lora_config.get("alpha", 32)
        dropout = lora_config.get("dropout", 0.05)
        target = lora_config.get("target_modules", "all-linear")

        # Build the LoRA configuration dict that mlx_lm expects.
        lora_cfg: dict[str, Any] = {
            "rank": rank,
            "alpha": alpha,
            "dropout": dropout,
            "scale": alpha / rank,
        }

        # Determine which layers to target.
        num_layers = len(self.model.model.layers) if hasattr(self.model, "model") else 0

        lora_parameters: dict[str, Any] = {
            "keys": None,  # will default to all linear
            "rank": rank,
            "alpha": alpha,
            "dropout": dropout,
            "scale": alpha / rank,
            "num_layers": num_layers,
        }

        if target != "all-linear":
            lora_parameters["keys"] = [m.strip() for m in target.split(",")]

        # Apply LoRA layers.
        apply_lora_layers(self.model, num_layers, lora_parameters)

        info = self.get_trainable_params_info()
        emit_log(
            "info",
            f"LoRA attached (MLX) — trainable: {info['trainable_params']:,} "
            f"/ {info['total_params']:,} ({info['trainable_pct']:.2f}%)",
        )

    # ------------------------------------------------------------------ #
    # train
    # ------------------------------------------------------------------ #

    def train(
        self,
        dataset: Any,
        training_config: dict[str, Any],
        output_dir: str,
        checkpoint_dir: str,
        resume_from: str | None,
        progress_callback: Callable[..., None],
        checkpoint_callback: Callable[..., None],
    ) -> str:
        """Run the MLX training loop with per‑step IPC progress.

        We use ``mlx_lm``'s lower‑level training utilities so we can emit
        progress after every optimiser step.
        """
        import mlx.core as mx  # type: ignore[import-untyped]
        import mlx.nn as nn  # type: ignore[import-untyped]
        import mlx.optimizers as optim  # type: ignore[import-untyped]

        assert self.model is not None

        # ---- hyperparameters -----------------------------------------
        batch_size: int = training_config.get("batch_size", 2)
        grad_accum: int = training_config.get("gradient_accumulation_steps", 8)
        lr: float = training_config.get("learning_rate", 2e-4)
        epochs: int = training_config.get("epochs", 3)
        warmup_ratio: float = training_config.get("warmup_ratio", 0.03)
        max_steps: int = training_config.get("max_steps", -1)
        save_ratio: float = training_config.get("save_steps_ratio", 0.1)

        # ---- dataset -------------------------------------------------
        train_data = dataset["train"]  # list of dicts with input_ids, labels, etc.

        steps_per_epoch = len(train_data) // (batch_size * grad_accum)
        total_steps = steps_per_epoch * epochs
        if max_steps > 0:
            total_steps = min(total_steps, max_steps)

        warmup_steps = max(1, int(total_steps * warmup_ratio))
        save_every = max(1, int(total_steps * save_ratio))

        emit_log(
            "info",
            f"Training (MLX): {total_steps} steps, {warmup_steps} warmup, "
            f"save every {save_every}",
        )

        # ---- optimiser -----------------------------------------------
        # Cosine schedule with warmup.
        schedule = optim.cosine_decay(lr, total_steps - warmup_steps)
        warmup_schedule = optim.linear_schedule(
            init=1e-7,
            end=lr,
            steps=warmup_steps,
        )
        combined_schedule = optim.join_schedules(
            [warmup_schedule, schedule],
            [warmup_steps],
        )
        optimizer = optim.AdamW(learning_rate=combined_schedule, weight_decay=0.01)

        # ---- resume --------------------------------------------------
        global_step = 0
        if resume_from and os.path.isdir(resume_from):
            adapter_file = os.path.join(resume_from, "adapters.safetensors")
            if os.path.exists(adapter_file):
                self.model.load_weights(adapter_file, strict=False)
                emit_log("info", f"Resumed adapter weights from {resume_from}")
                # Try to recover step count from filename.
                try:
                    dirname = os.path.basename(resume_from)
                    if dirname.startswith("checkpoint-"):
                        global_step = int(dirname.split("-")[1])
                except (ValueError, IndexError):
                    pass

        # ---- loss function -------------------------------------------
        def loss_fn(
            model: Any,
            input_ids: Any,
            labels: Any,
            attention_mask: Any,
        ) -> Any:
            logits = model(input_ids)
            # Shift logits and labels for causal LM.
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]

            # Cross-entropy loss.
            loss = nn.losses.cross_entropy(
                shift_logits.reshape(-1, shift_logits.shape[-1]),
                shift_labels.reshape(-1),
                reduction="mean",
            )
            return loss

        loss_and_grad = nn.value_and_grad(self.model, loss_fn)

        # ---- training loop -------------------------------------------
        self.model.train()
        running_loss = 0.0
        micro_step = 0
        tokens_seen = 0
        step_start = time.monotonic()
        ema_step_time = None
        last_step_end = time.monotonic()
        accumulated_grads: Any = None

        for epoch in range(epochs):
            # Shuffle data each epoch.
            import random

            indices = list(range(len(train_data)))
            random.shuffle(indices)

            for i in range(0, len(indices) - batch_size + 1, batch_size):
                if max_steps > 0 and global_step >= max_steps:
                    break

                batch_indices = indices[i : i + batch_size]
                batch = [train_data[idx] for idx in batch_indices]

                # Stack batch.
                input_ids = mx.array(
                    [item["input_ids"] for item in batch]
                )
                labels = mx.array(
                    [item["labels"] for item in batch]
                )
                attention_mask = mx.array(
                    [item["attention_mask"] for item in batch]
                )

                # Forward + backward.
                loss, grads = loss_and_grad(
                    self.model, input_ids, labels, attention_mask
                )
                running_loss += loss.item() / grad_accum
                tokens_seen += int(attention_mask.sum().item())

                # Accumulate gradients.
                if accumulated_grads is None:
                    accumulated_grads = grads
                else:
                    accumulated_grads = _tree_map_add(accumulated_grads, grads)

                micro_step += 1

                if micro_step % grad_accum == 0:
                    # Scale accumulated gradients.
                    accumulated_grads = _tree_map_scale(
                        accumulated_grads, 1.0 / grad_accum
                    )

                    # Clip gradients.
                    accumulated_grads, _ = optim.clip_grad_norm(
                        accumulated_grads, max_norm=1.0
                    )

                    optimizer.update(self.model, accumulated_grads)
                    mx.eval(self.model.parameters())
                    accumulated_grads = None

                    global_step += 1
                    now = time.monotonic()
                    step_duration = now - last_step_end
                    last_step_end = now

                    if ema_step_time is None:
                        ema_step_time = step_duration
                    else:
                        ema_step_time = 0.9 * ema_step_time + 0.1 * step_duration

                    elapsed = now - step_start
                    tps = tokens_seen / max(elapsed, 1e-6)
                    steps_left = total_steps - global_step
                    eta = ema_step_time * steps_left

                    current_lr = optimizer.learning_rate.item() if hasattr(
                        optimizer.learning_rate, "item"
                    ) else float(lr)

                    progress_callback(
                        step=global_step,
                        total_steps=total_steps,
                        loss=running_loss,
                        lr=current_lr,
                        tokens_per_sec=tps,
                        eta_seconds=max(0.0, eta),
                    )

                    running_loss = 0.0

                    # Checkpoint.
                    if global_step % save_every == 0:
                        ckpt_path = os.path.join(
                            checkpoint_dir, f"checkpoint-{global_step}"
                        )
                        self._save_checkpoint(ckpt_path)
                        checkpoint_callback(path=ckpt_path, step=global_step)

            if max_steps > 0 and global_step >= max_steps:
                break

        # ---- final save ----------------------------------------------
        adapter_path = os.path.join(output_dir, "adapter")
        self.save_adapter(adapter_path)
        return adapter_path

    # ------------------------------------------------------------------ #
    # _save_checkpoint
    # ------------------------------------------------------------------ #

    def _save_checkpoint(self, path: str) -> None:
        import mlx.core as mx  # type: ignore[import-untyped]

        os.makedirs(path, exist_ok=True)
        assert self.model is not None
        # Save only trainable (LoRA) weights.
        trainable = {
            k: v
            for k, v in dict(self.model.trainable_parameters()).items()
        }
        mx.save_safetensors(os.path.join(path, "adapters.safetensors"), trainable)
        emit_log("info", f"Checkpoint saved: {path}")

    # ------------------------------------------------------------------ #
    # save_adapter
    # ------------------------------------------------------------------ #

    def save_adapter(self, output_path: str) -> None:
        import mlx.core as mx  # type: ignore[import-untyped]

        assert self.model is not None
        os.makedirs(output_path, exist_ok=True)
        trainable = {
            k: v
            for k, v in dict(self.model.trainable_parameters()).items()
        }
        mx.save_safetensors(
            os.path.join(output_path, "adapters.safetensors"), trainable
        )
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_path)
        emit_log("info", f"Adapter saved to {output_path}")

    # ------------------------------------------------------------------ #
    # merge_adapter
    # ------------------------------------------------------------------ #

    def merge_adapter(self, output_path: str) -> None:
        """Fuse LoRA weights into the base model and export in HF format.

        Uses ``mlx_lm.fuse_model`` to merge, then converts back to HF
        safetensors format for GGUF conversion.
        """
        try:
            from mlx_lm import fuse_model  # type: ignore[import-untyped]
        except ImportError:
            from mlx_lm.utils import fuse_model  # type: ignore[import-untyped]

        emit_log("info", "Fusing LoRA adapters into base model (MLX)…")

        # fuse_model writes the merged model to output_path.
        assert self.model is not None
        os.makedirs(output_path, exist_ok=True)

        # Save current adapter weights to a temp dir, then fuse.
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_adapter:
            self.save_adapter(tmp_adapter)
            fuse_model(
                model=self._base_model_id,
                adapter_path=tmp_adapter,
                save_path=output_path,
            )

        # Convert MLX weights back to HF safetensors if needed.
        self._convert_mlx_to_hf(output_path)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_path)

        emit_log("info", f"Merged model saved to {output_path}")

    def _convert_mlx_to_hf(self, model_path: str) -> None:
        """Convert MLX‑format weights to HF safetensors if they aren't already.

        ``mlx_lm.fuse_model`` already saves in HF‑compatible format for
        most models, so this is a best-effort conversion.
        """
        from pathlib import Path

        p = Path(model_path)
        # If there are already .safetensors files, we're good.
        if list(p.glob("*.safetensors")):
            return

        # If there are .npz files (older MLX format), convert via mlx_lm.
        npz_files = list(p.glob("*.npz"))
        if npz_files:
            emit_log("info", "Converting MLX .npz weights to safetensors…")
            try:
                from mlx_lm.utils import save_model  # type: ignore[import-untyped]
                import mlx_lm  # type: ignore[import-untyped]

                model, _ = mlx_lm.load(str(p))
                save_model(str(p), model)
                # Clean up .npz files.
                for f in npz_files:
                    f.unlink()
            except Exception as exc:
                emit_log("warning", f"Could not convert MLX weights: {exc}")

    # ------------------------------------------------------------------ #
    # get_trainable_params_info
    # ------------------------------------------------------------------ #

    def get_trainable_params_info(self) -> dict[str, Any]:
        import mlx.core as mx  # type: ignore[import-untyped]

        assert self.model is not None

        def _count_params(params: Any) -> int:
            """Recursively count elements in a nested dict/list of arrays."""
            if isinstance(params, mx.array):
                return params.size
            if isinstance(params, dict):
                return sum(_count_params(v) for v in params.values())
            if isinstance(params, (list, tuple)):
                return sum(_count_params(v) for v in params)
            return 0

        total = _count_params(self.model.parameters())
        trainable = _count_params(self.model.trainable_parameters())

        return {
            "trainable_params": trainable,
            "total_params": total,
            "trainable_pct": 100.0 * trainable / max(total, 1),
        }


# ---------------------------------------------------------------------------
# Tree‑map utilities for gradient accumulation
# ---------------------------------------------------------------------------


def _tree_map_add(tree_a: Any, tree_b: Any) -> Any:
    """Add corresponding arrays in nested gradient structures."""
    if isinstance(tree_a, dict):
        return {key: _tree_map_add(tree_a[key], tree_b[key]) for key in tree_a}
    if isinstance(tree_a, list):
        return [_tree_map_add(a, b) for a, b in zip(tree_a, tree_b)]
    if isinstance(tree_a, tuple):
        return tuple(_tree_map_add(a, b) for a, b in zip(tree_a, tree_b))
    return tree_a + tree_b


def _tree_map_scale(tree: Any, scale: float) -> Any:
    """Scale every array in a nested gradient structure."""
    if isinstance(tree, dict):
        return {key: _tree_map_scale(value, scale) for key, value in tree.items()}
    if isinstance(tree, list):
        return [_tree_map_scale(value, scale) for value in tree]
    if isinstance(tree, tuple):
        return tuple(_tree_map_scale(value, scale) for value in tree)
    return tree * scale

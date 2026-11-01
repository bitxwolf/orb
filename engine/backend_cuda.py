"""
CUDA training backend using transformers + peft + bitsandbytes.

Uses a **custom training loop** (not HF Trainer) so we have per‑step IPC
control for progress reporting, checkpointing, and graceful OOM handling.
"""

from __future__ import annotations

import gc
import logging
import math
import os
import time
from typing import TYPE_CHECKING, Any, Callable

from engine.backend_base import TrainingBackend
from engine.ipc import emit_log

if TYPE_CHECKING:
    from peft import PeftModel
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _translate_oom(exc: Exception) -> str:
    """Turn a CUDA OOM error into a plain‑language suggestion."""
    msg = str(exc).lower()
    if "out of memory" in msg or "cuda" in msg:
        return (
            "GPU ran out of memory during training. "
            "Try reducing batch_size, seq_length, or lora rank. "
            "Gradient checkpointing is already enabled by default."
        )
    return f"Training error: {exc}"


def _resolve_target_modules(target: str) -> str | list[str]:
    """Resolve the ``target_modules`` config value.

    ``"all-linear"`` is passed through as a string (peft recognises it).
    Otherwise we split a comma‑separated list into a Python list.
    """
    if target == "all-linear":
        return "all-linear"
    return [m.strip() for m in target.split(",") if m.strip()]


# ---------------------------------------------------------------------------
# Backend implementation
# ---------------------------------------------------------------------------


class CUDABackend(TrainingBackend):
    """QLoRA fine‑tuning on NVIDIA GPUs via bitsandbytes NF4 quantisation."""

    def __init__(self) -> None:
        self.model: PreTrainedModel | PeftModel | None = None
        self.tokenizer: PreTrainedTokenizerBase | None = None  # type: ignore[assignment]
        self._base_model_id: str = ""
        self._device: str = "cuda"

    # ------------------------------------------------------------------ #
    # load_model
    # ------------------------------------------------------------------ #

    def load_model(
        self,
        model_id: str,
        hf_token: str | None,
        config: dict[str, Any],
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self._base_model_id = model_id

        # Performance knobs
        torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
        torch.backends.cudnn.allow_tf32 = True  # type: ignore[attr-defined]

        # 4‑bit quantisation config (NF4, double‑quant, bf16 compute)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        emit_log("info", f"Loading model {model_id} with NF4 quantisation…")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            token=hf_token,
            trust_remote_code=False,
            attn_implementation="sdpa",  # scaled‑dot‑product attention
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=hf_token,
            trust_remote_code=False,
        )

        # Ensure pad token exists.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Prepare for k‑bit training (freeze + cast norms to fp32).
        from peft import prepare_model_for_kbit_training

        self.model = prepare_model_for_kbit_training(
            self.model,
            use_gradient_checkpointing=config.get("training", {}).get(
                "gradient_checkpointing", True
            ),
        )

        emit_log("info", f"Model loaded on {self._device}.")

    # ------------------------------------------------------------------ #
    # prepare_qlora
    # ------------------------------------------------------------------ #

    def prepare_qlora(self, lora_config: dict[str, Any]) -> None:
        from peft import LoraConfig, TaskType, get_peft_model

        assert self.model is not None, "Call load_model() first."

        target_modules = _resolve_target_modules(
            lora_config.get("target_modules", "all-linear")
        )

        peft_cfg = LoraConfig(
            r=lora_config.get("rank", 16),
            lora_alpha=lora_config.get("alpha", 32),
            lora_dropout=lora_config.get("dropout", 0.05),
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        self.model = get_peft_model(self.model, peft_cfg)

        info = self.get_trainable_params_info()
        emit_log(
            "info",
            f"LoRA attached — trainable: {info['trainable_params']:,} "
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
        import torch
        from torch.utils.data import DataLoader

        assert self.model is not None, "Call load_model() + prepare_qlora() first."

        # ---- hyper‑parameters ----------------------------------------
        batch_size: int = training_config.get("batch_size", 2)
        grad_accum: int = training_config.get("gradient_accumulation_steps", 8)
        lr: float = training_config.get("learning_rate", 2e-4)
        epochs: int = training_config.get("epochs", 3)
        warmup_ratio: float = training_config.get("warmup_ratio", 0.03)
        max_steps: int = training_config.get("max_steps", -1)
        save_ratio: float = training_config.get("save_steps_ratio", 0.1)
        use_8bit_optim: bool = training_config.get("use_8bit_optimizer", True)
        grad_ckpt: bool = training_config.get("gradient_checkpointing", True)

        # ---- gradient checkpointing ----------------------------------
        if grad_ckpt:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        # ---- dataloader ----------------------------------------------
        train_ds = dataset["train"]
        dataloader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=0,  # safest for tokenised datasets
            collate_fn=dataset.get("collate_fn"),
        )

        steps_per_epoch = len(dataloader) // grad_accum
        total_steps = steps_per_epoch * epochs
        if max_steps > 0:
            total_steps = min(total_steps, max_steps)

        warmup_steps = max(1, int(total_steps * warmup_ratio))
        save_every = max(1, int(total_steps * save_ratio))

        emit_log(
            "info",
            f"Training: {total_steps} steps, {warmup_steps} warmup, "
            f"save every {save_every}, batch {batch_size}×{grad_accum} accum",
        )

        # ---- optimiser -----------------------------------------------
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if use_8bit_optim:
            try:
                import bitsandbytes as bnb

                optimiser = bnb.optim.AdamW8bit(
                    trainable_params,
                    lr=lr,
                    betas=(0.9, 0.999),
                    weight_decay=0.01,
                )
            except ImportError:
                emit_log("warning", "bitsandbytes 8‑bit Adam unavailable, using torch AdamW")
                optimiser = torch.optim.AdamW(
                    trainable_params, lr=lr, weight_decay=0.01
                )
        else:
            optimiser = torch.optim.AdamW(
                trainable_params, lr=lr, weight_decay=0.01
            )

        # ---- LR scheduler (cosine with warmup) -----------------------
        def _lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return current_step / warmup_steps
            progress = (current_step - warmup_steps) / max(
                1, total_steps - warmup_steps
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, _lr_lambda)

        # ---- resume from checkpoint ----------------------------------
        global_step = 0
        start_epoch = 0
        if resume_from and os.path.isdir(resume_from):
            ckpt_state_path = os.path.join(resume_from, "training_state.pt")
            if os.path.exists(ckpt_state_path):
                state = torch.load(ckpt_state_path, map_location="cpu", weights_only=True)
                global_step = state.get("global_step", 0)
                start_epoch = state.get("epoch", 0)
                optimiser.load_state_dict(state["optimiser"])
                scheduler.load_state_dict(state["scheduler"])
                emit_log("info", f"Resumed from step {global_step}")
            # Also load adapter weights.
            from peft import PeftModel as PM

            if hasattr(self.model, "load_adapter"):
                self.model.load_adapter(resume_from, "default")
                emit_log("info", f"Loaded adapter weights from {resume_from}")

        # ---- training loop -------------------------------------------
        self.model.train()
        running_loss = 0.0
        micro_step = 0
        tokens_seen = 0
        step_start = time.monotonic()
        ema_step_time = None
        last_step_end = time.monotonic()

        try:
            for epoch in range(start_epoch, epochs):
                for batch in dataloader:
                    if max_steps > 0 and global_step >= max_steps:
                        break

                    # Move batch to device.
                    input_ids = batch["input_ids"].to(self._device)
                    attention_mask = batch["attention_mask"].to(self._device)
                    labels = batch["labels"].to(self._device)

                    # Forward.
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss / grad_accum
                    running_loss += loss.item()

                    # Count tokens (non‑padding).
                    tokens_seen += int(attention_mask.sum().item())

                    # Backward.
                    loss.backward()

                    micro_step += 1

                    # Optimiser step (after accumulation).
                    if micro_step % grad_accum == 0:
                        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                        optimiser.step()
                        scheduler.step()
                        optimiser.zero_grad(set_to_none=True)

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

                        progress_callback(
                            step=global_step,
                            total_steps=total_steps,
                            loss=running_loss,
                            lr=scheduler.get_last_lr()[0],
                            tokens_per_sec=tps,
                            eta_seconds=max(0.0, eta),
                        )

                        running_loss = 0.0

                        # Checkpoint.
                        if global_step % save_every == 0:
                            ckpt_path = os.path.join(
                                checkpoint_dir, f"checkpoint-{global_step}"
                            )
                            self._save_checkpoint(
                                ckpt_path, global_step, epoch, optimiser, scheduler
                            )
                            checkpoint_callback(path=ckpt_path, step=global_step)

                if max_steps > 0 and global_step >= max_steps:
                    break

        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                # Free whatever we can before reporting.
                torch.cuda.empty_cache()
                gc.collect()
                raise RuntimeError(_translate_oom(exc)) from exc
            raise

        # ---- final save ----------------------------------------------
        adapter_path = os.path.join(output_dir, "adapter")
        self.save_adapter(adapter_path)
        return adapter_path

    # ------------------------------------------------------------------ #
    # Internal: checkpoint save
    # ------------------------------------------------------------------ #

    def _save_checkpoint(
        self,
        path: str,
        step: int,
        epoch: int,
        optimiser: Any,
        scheduler: Any,
    ) -> None:
        import torch

        os.makedirs(path, exist_ok=True)
        assert self.model is not None
        self.model.save_pretrained(path)
        torch.save(
            {
                "global_step": step,
                "epoch": epoch,
                "optimiser": optimiser.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            os.path.join(path, "training_state.pt"),
        )

    # ------------------------------------------------------------------ #
    # Adapter operations
    # ------------------------------------------------------------------ #

    def save_adapter(self, output_path: str) -> None:
        """Save the trained PEFT adapter and tokenizer."""
        assert self.model is not None, "No model is loaded."
        os.makedirs(output_path, exist_ok=True)
        self.model.save_pretrained(output_path)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_path)

    def merge_adapter(self, output_path: str) -> str:
        """Merge the adapter into the base model and save the merged model."""
        assert self.model is not None, "No model is loaded."
        if not hasattr(self.model, "merge_and_unload"):
            raise RuntimeError("The loaded model does not have a mergeable adapter.")

        os.makedirs(output_path, exist_ok=True)
        self.model = self.model.merge_and_unload()
        self.model.save_pretrained(output_path, safe_serialization=True)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_path)
        return output_path

    def get_trainable_params_info(self) -> dict[str, int | float]:
        """Return the trainable and total parameter counts and percentage."""
        assert self.model is not None, "No model is loaded."
        trainable_params = sum(
            parameter.numel()
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )
        total_params = sum(parameter.numel() for parameter in self.model.parameters())
        trainable_pct = 100.0 * trainable_params / total_params if total_params else 0.0
        return {
            "trainable_params": trainable_params,
            "total_params": total_params,
            "trainable_pct": trainable_pct,
        }

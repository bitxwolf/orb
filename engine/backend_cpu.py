"""
CPU training backend wrapping llama.cpp's ``llama-train`` (or ``finetune``) binary.

Unlike the CUDA and MLX backends, this backend does **not** load the model
into Python.  Instead, it spawns ``llama-train`` as a subprocess, streams its
stdout line-by-line, and parses the training metrics (iter, loss, eta) using
regex so they can be forwarded through the standard IPC progress protocol.

The user must have a pre-converted GGUF base model and the llama.cpp training
binary available on disk.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from typing import Any, Callable

from engine.backend_base import TrainingBackend
from engine.ipc import emit_log

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Regex for llama.cpp training output
# ---------------------------------------------------------------------------

# Matches lines like:
#   train_opt_callback: iter=    0  sample=1/N  sched=1.000000
#       loss=3.254590  |->loss=3.254590  dt=0.58s  eta=5m15s
_ITER_RE = re.compile(
    r"iter\s*=\s*(\d+)",
)
_LOSS_RE = re.compile(
    r"\bloss\s*=\s*([\d.]+)",
)
_DT_RE = re.compile(
    r"\bdt\s*=\s*([\d.]+)\s*s",
)
_ETA_RE = re.compile(
    r"\beta\s*=\s*(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?",
)
_SAMPLE_RE = re.compile(
    r"sample\s*=\s*\d+\s*/\s*(\d+)",
)


def _parse_eta_seconds(match: re.Match[str]) -> float:
    """Convert an ``eta=XhYmZs`` regex match into total seconds."""
    hours = int(match.group(1)) if match.group(1) else 0
    minutes = int(match.group(2)) if match.group(2) else 0
    seconds = int(match.group(3)) if match.group(3) else 0
    return hours * 3600.0 + minutes * 60.0 + seconds


# ---------------------------------------------------------------------------
# Backend implementation
# ---------------------------------------------------------------------------


class CPUBackend(TrainingBackend):
    """CPU fine-tuning via the llama.cpp ``llama-train`` / ``finetune`` binary."""

    def __init__(self) -> None:
        self.tokenizer: Any = None  # Not used — llama.cpp tokenises internally.
        self._model_id: str = ""
        self._config: dict[str, Any] = {}
        self._lora_config: dict[str, Any] = {}
        self._adapter_output_path: str = ""

    # ------------------------------------------------------------------ #
    # load_model
    # ------------------------------------------------------------------ #

    def load_model(
        self,
        model_id: str,
        hf_token: str | None,
        config: dict[str, Any],
    ) -> None:
        """Store model identifiers — llama.cpp loads the GGUF at train time."""
        self._model_id = model_id
        self._config = config

        base_gguf = config.get("base_gguf_path", "")
        if base_gguf and not os.path.isfile(base_gguf):
            emit_log("warning", f"base_gguf_path does not exist yet: {base_gguf}")

        emit_log(
            "info",
            f"CPU backend initialised for model {model_id} "
            f"(GGUF: {base_gguf or 'not specified'}).",
        )

    # ------------------------------------------------------------------ #
    # prepare_qlora
    # ------------------------------------------------------------------ #

    def prepare_qlora(self, lora_config: dict[str, Any]) -> None:
        """Store LoRA hyper-parameters — llama.cpp handles injection itself."""
        self._lora_config = lora_config

        rank = lora_config.get("rank", 16)
        alpha = lora_config.get("alpha", 32)

        emit_log(
            "info",
            f"LoRA config stored (rank={rank}, alpha={alpha}). "
            "Adapter injection is handled by llama.cpp at train time.",
        )

    # ------------------------------------------------------------------ #
    # get_trainable_params_info
    # ------------------------------------------------------------------ #

    def get_trainable_params_info(self) -> dict[str, Any]:
        """Return *estimated* parameter counts based on LoRA rank.

        The real counts depend on the model architecture; llama.cpp does
        not expose them to us, so we provide reasonable placeholders.
        """
        rank = self._lora_config.get("rank", 16)
        # Rough heuristic: LoRA adds ~ rank * hidden_dim * 2 * n_target_layers
        # Use a conservative placeholder since we can't inspect the GGUF.
        estimated_trainable = rank * 4096 * 2 * 32  # ~8M for a 7B-class model
        estimated_total = 7_000_000_000  # placeholder

        return {
            "trainable_params": estimated_trainable,
            "total_params": estimated_total,
            "trainable_pct": 100.0 * estimated_trainable / max(estimated_total, 1),
        }

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
        """Spawn ``llama-train`` / ``finetune`` and stream its output.

        The command is built from config values (``llamacpp_bin_path``,
        ``base_gguf_path``) and training hyper-parameters.  stdout is read
        line-by-line and parsed for training metrics which are forwarded
        through *progress_callback*.
        """
        # ---- resolve binary & model paths ----------------------------
        bin_path = self._config.get("llamacpp_bin_path", "")
        if not bin_path:
            # Try well-known names on PATH.
            for name in ("llama-finetune", "finetune", "llama-train"):
                found = shutil.which(name)
                if found:
                    bin_path = found
                    break
        if not bin_path:
            raise FileNotFoundError(
                "Cannot find llama-train / finetune binary. "
                "Set 'llamacpp_bin_path' in the engine config."
            )

        base_gguf = self._config.get("base_gguf_path", "")
        if not base_gguf:
            raise ValueError(
                "No 'base_gguf_path' specified in config. "
                "Provide the path to a GGUF base model."
            )

        # ---- hyper-parameters ----------------------------------------
        epochs: int = training_config.get("epochs", 3)
        batch_size: int = training_config.get("batch_size", 2)
        lr: float = training_config.get("learning_rate", 2e-4)
        ctx_size: int = self._config.get("seq_length", 512)
        threads: int = training_config.get("threads", 0)  # 0 = auto
        grad_accum: int = training_config.get("gradient_accumulation_steps", 8)

        lora_rank: int = self._lora_config.get("rank", 16)
        lora_alpha: int = self._lora_config.get("alpha", 32)

        # Adapter output.
        adapter_path = os.path.join(output_dir, "adapter")
        os.makedirs(adapter_path, exist_ok=True)
        lora_out = os.path.join(adapter_path, "ggml-adapter-model.bin")
        self._adapter_output_path = adapter_path

        # Dataset — for llama.cpp we need a plain-text file.
        # The orchestrator may pass a dict; extract the path or the text file.
        train_file = ""
        if isinstance(dataset, dict):
            train_file = dataset.get("train_file", dataset.get("path", ""))
        elif isinstance(dataset, str):
            train_file = dataset
        if not train_file:
            train_file = self._config.get("dataset_path", "")

        # ---- build command -------------------------------------------
        cmd: list[str] = [
            bin_path,
            "--model", base_gguf,
            "--train-data", train_file,
            "--lora-out", lora_out,
            "--lora-r", str(lora_rank),
            "--lora-alpha", str(lora_alpha),
            "--ctx", str(ctx_size),
            "--batch", str(batch_size),
            "--grad-acc", str(grad_accum),
            "--learning-rate", str(lr),
            "--epochs", str(epochs),
            "--checkpoint-out", os.path.join(checkpoint_dir, "chk-%s.gguf"),
        ]

        if threads > 0:
            cmd.extend(["--threads", str(threads)])

        if resume_from and os.path.isdir(resume_from):
            # Look for a checkpoint file to resume from.
            for fname in os.listdir(resume_from):
                if fname.endswith(".gguf"):
                    cmd.extend(["--checkpoint-in", os.path.join(resume_from, fname)])
                    break

        emit_log("info", f"Running: {' '.join(cmd)}")

        # ---- spawn process -------------------------------------------
        total_steps: int = 0  # will be inferred from output

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # line-buffered
        )

        assert process.stdout is not None

        try:
            for line in process.stdout:
                line = line.rstrip("\n\r")
                if not line:
                    continue

                emit_log("debug", f"[llama-train] {line}")

                # Try to extract total samples from 'sample=X/N'.
                sample_m = _SAMPLE_RE.search(line)
                if sample_m and total_steps == 0:
                    total_steps = int(sample_m.group(1)) * epochs

                # Parse training metrics.
                iter_m = _ITER_RE.search(line)
                if iter_m is None:
                    continue

                step = int(iter_m.group(1))

                loss_m = _LOSS_RE.search(line)
                loss_val = float(loss_m.group(1)) if loss_m else 0.0

                dt_m = _DT_RE.search(line)
                dt_val = float(dt_m.group(1)) if dt_m else 0.0
                tokens_per_sec = (ctx_size * batch_size) / max(dt_val, 1e-6) if dt_val else 0.0

                eta_m = _ETA_RE.search(line)
                eta_seconds = _parse_eta_seconds(eta_m) if eta_m else 0.0

                progress_callback(
                    step=step + 1,  # 1-indexed for consistency with other backends
                    total_steps=total_steps if total_steps > 0 else step + 1,
                    loss=loss_val,
                    lr=lr,
                    tokens_per_sec=tokens_per_sec,
                    eta_seconds=eta_seconds,
                )

        finally:
            process.wait()

        if process.returncode != 0:
            raise RuntimeError(
                f"llama-train exited with code {process.returncode}. "
                "Check the debug logs for details."
            )

        # ---- report checkpoints --------------------------------------
        for fname in os.listdir(checkpoint_dir):
            if fname.endswith(".gguf"):
                ckpt_path = os.path.join(checkpoint_dir, fname)
                checkpoint_callback(path=ckpt_path, step=0)

        emit_log("info", f"Training complete. Adapter written to {adapter_path}")
        return adapter_path

    # ------------------------------------------------------------------ #
    # save_adapter
    # ------------------------------------------------------------------ #

    def save_adapter(self, output_path: str) -> None:
        """Copy the llama.cpp LoRA adapter to *output_path*."""
        if not self._adapter_output_path:
            emit_log("warning", "No adapter to save — training may not have run.")
            return

        if os.path.abspath(output_path) == os.path.abspath(self._adapter_output_path):
            emit_log("info", f"Adapter already at {output_path}")
            return

        os.makedirs(output_path, exist_ok=True)
        # Copy all files from the adapter directory.
        for fname in os.listdir(self._adapter_output_path):
            src = os.path.join(self._adapter_output_path, fname)
            dst = os.path.join(output_path, fname)
            if os.path.isfile(src):
                shutil.copy2(src, dst)

        emit_log("info", f"Adapter saved to {output_path}")

    # ------------------------------------------------------------------ #
    # merge_adapter
    # ------------------------------------------------------------------ #

    def merge_adapter(self, output_path: str) -> None:
        """Merge the LoRA adapter into the base model using ``llama-export-lora``.

        Falls back to copying the adapter files if the merge binary is not
        available, since the user can merge manually later.
        """
        base_gguf = self._config.get("base_gguf_path", "")

        # Find the adapter file.
        adapter_file = ""
        if self._adapter_output_path:
            for fname in os.listdir(self._adapter_output_path):
                if fname.endswith((".bin", ".gguf")):
                    adapter_file = os.path.join(self._adapter_output_path, fname)
                    break

        if not adapter_file:
            emit_log("warning", "No adapter file found to merge.")
            return

        # Try to find llama-export-lora.
        merge_bin: str | None = None
        for name in ("llama-export-lora", "export-lora"):
            found = shutil.which(name)
            if found:
                merge_bin = found
                break

        os.makedirs(output_path, exist_ok=True)
        merged_gguf = os.path.join(output_path, "merged-model.gguf")

        if merge_bin and base_gguf:
            emit_log("info", "Merging LoRA adapter into base GGUF model…")
            cmd = [
                merge_bin,
                "--model", base_gguf,
                "--lora", adapter_file,
                "--output", merged_gguf,
            ]
            emit_log("debug", f"Running: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,
            )

            if result.returncode != 0:
                emit_log(
                    "warning",
                    f"Merge failed (exit {result.returncode}). "
                    "Copying adapter instead. You can merge manually later.",
                )
                shutil.copy2(adapter_file, os.path.join(output_path, os.path.basename(adapter_file)))
            else:
                emit_log("info", f"Merged model saved to {merged_gguf}")
        else:
            if not merge_bin:
                emit_log("warning", "llama-export-lora binary not found. Copying adapter instead.")
            else:
                emit_log("warning", "Base GGUF model not specified. Copying adapter instead.")
            shutil.copy2(adapter_file, os.path.join(output_path, os.path.basename(adapter_file)))

"""
Abstract base class for Loom training backends.

Both the CUDA (``backend_cuda.py``) and MLX (``backend_mlx.py``) backends
implement this interface so that the orchestrator (``trainer.py``) can drive
training without caring about the underlying framework.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass  # heavy imports (torch, mlx) only under type‑checking


class TrainingBackend(ABC):
    """Unified interface every training backend must implement.

    Lifecycle:
        1. ``load_model()``         – download / load base model
        2. ``prepare_qlora()``      – attach LoRA adapters
        3. ``get_trainable_params_info()`` – report param counts
        4. ``train()``              – run the training loop
        5. ``save_adapter()``       – persist LoRA weights
        6. (optional) ``merge_adapter()``  – merge LoRA into base
    """

    # Sub‑classes MUST set this after load_model() so dataset.py can tokenize.
    tokenizer: Any = None

    # ------------------------------------------------------------------ #
    # Model loading
    # ------------------------------------------------------------------ #

    @abstractmethod
    def load_model(
        self,
        model_id: str,
        hf_token: str | None,
        config: dict[str, Any],
    ) -> None:
        """Load a pre‑trained model from Hugging Face Hub or a local path.

        Parameters
        ----------
        model_id:
            Hub repo id (e.g. ``"meta-llama/Llama-3.2-1B"``) or local
            directory.
        hf_token:
            Optional Hugging Face API token for gated models.
        config:
            The full engine config dict – backends may read additional
            knobs such as ``seq_length``.
        """

    # ------------------------------------------------------------------ #
    # LoRA / QLoRA setup
    # ------------------------------------------------------------------ #

    @abstractmethod
    def prepare_qlora(self, lora_config: dict[str, Any]) -> None:
        """Attach QLoRA adapters to the loaded model.

        Parameters
        ----------
        lora_config:
            Dict with keys ``rank``, ``alpha``, ``dropout``,
            ``target_modules``.
        """

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    @abstractmethod
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
        """Execute the training loop.

        Parameters
        ----------
        dataset:
            A tokenized dataset object (format depends on the backend).
        training_config:
            Dict with keys such as ``batch_size``, ``learning_rate``,
            ``epochs``, ``gradient_accumulation_steps``, etc.
        output_dir:
            Directory for final adapter output.
        checkpoint_dir:
            Directory for intermediate checkpoints.
        resume_from:
            Path to a checkpoint to resume from, or ``None``.
        progress_callback:
            Called after every optimiser step with keyword arguments
            ``step``, ``total_steps``, ``loss``, ``lr``,
            ``tokens_per_sec``, ``eta_seconds``.
        checkpoint_callback:
            Called when a checkpoint is saved with keyword arguments
            ``path`` and ``step``.

        Returns
        -------
        str
            Path to the saved adapter directory.
        """

    # ------------------------------------------------------------------ #
    # Adapter persistence
    # ------------------------------------------------------------------ #

    @abstractmethod
    def save_adapter(self, output_path: str) -> None:
        """Save only the LoRA adapter weights to *output_path*."""

    @abstractmethod
    def merge_adapter(self, output_path: str) -> None:
        """Merge the LoRA adapter into the base model and save.

        The resulting directory should contain full‑precision weights in
        safetensors format, ready for GGUF conversion.
        """

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #

    @abstractmethod
    def get_trainable_params_info(self) -> dict[str, Any]:
        """Return a dict describing parameter counts.

        Expected keys:

        * ``trainable_params`` – number of trainable parameters
        * ``total_params``     – total model parameters
        * ``trainable_pct``    – percentage trainable (e.g. ``1.23``)
        """

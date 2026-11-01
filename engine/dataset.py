"""
Dataset loading, tokenisation, and collation for Loom.

The Node.js CLI pre‑processes raw user data into a **normalised JSONL** file
where every line is a JSON object with a ``"messages"`` key containing a list
of ``{"role": ..., "content": ...}`` dicts (OpenAI chat format).

This module:

1. Loads the normalised JSONL.
2. Applies the model's **chat template** to render each conversation into a
   single string.
3. Tokenises with the model's tokenizer.
4. Splits into train / validation (90 / 10).
5. Returns either a PyTorch ``DataLoader``‑ready dict (CUDA) or a plain list
   of dicts (MLX).
"""

from __future__ import annotations

import json
import logging
import random
from typing import TYPE_CHECKING, Any

from engine.ipc import emit_log

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_and_tokenize(
    dataset_path: str,
    tokenizer: Any,
    seq_length: int,
    backend: str,
    seed: int = 42,
    val_ratio: float = 0.10,
) -> dict[str, Any]:
    """Load the normalised JSONL dataset, tokenise, and split.

    Parameters
    ----------
    dataset_path:
        Path to the ``.jsonl`` file produced by the Node.js pipeline.
    tokenizer:
        A Hugging Face ``PreTrainedTokenizer`` (works for both CUDA and MLX
        since ``mlx_lm`` uses the same tokenizer classes).
    seq_length:
        Maximum sequence length for tokenisation.
    backend:
        ``"cuda"`` or ``"mlx"`` — determines the return format.
    seed:
        Random seed for the train/val split.
    val_ratio:
        Fraction of examples to hold out for validation.

    Returns
    -------
    dict
        For CUDA: ``{"train": Dataset, "val": Dataset, "collate_fn": callable}``
        For MLX:  ``{"train": list[dict], "val": list[dict]}``
    """
    # ---- 1. Load raw conversations -----------------------------------
    emit_log("info", f"Loading dataset from {dataset_path}…")
    conversations = _load_jsonl(dataset_path)
    emit_log("info", f"Loaded {len(conversations)} examples.")

    if not conversations:
        raise ValueError(f"Dataset is empty: {dataset_path}")

    # ---- 2. Apply chat template → list of strings --------------------
    emit_log("info", "Applying chat template…")
    texts = _apply_chat_template(conversations, tokenizer)

    # ---- 3. Tokenise -------------------------------------------------
    emit_log("info", f"Tokenising (seq_length={seq_length})…")
    tokenised = _tokenise_all(texts, tokenizer, seq_length)

    # ---- 4. Split train / val ----------------------------------------
    rng = random.Random(seed)
    indices = list(range(len(tokenised)))
    rng.shuffle(indices)
    
    n_val = int(len(indices) * val_ratio)
    if n_val == 0 and len(indices) > 1 and val_ratio > 0:
        n_val = 1
    elif n_val >= len(indices):
        n_val = len(indices) - 1

    val_indices = set(indices[:n_val])
    train_data = [tokenised[i] for i in range(len(tokenised)) if i not in val_indices]
    val_data = [tokenised[i] for i in range(len(tokenised)) if i in val_indices]

    emit_log("info", f"Split: {len(train_data)} train / {len(val_data)} val.")

    # ---- 5. Return in backend‑appropriate format ---------------------
    if backend == "cuda":
        return _wrap_for_cuda(train_data, val_data, tokenizer)
    else:
        return {"train": train_data, "val": val_data}


# ---------------------------------------------------------------------------
# Internal: JSONL loading
# ---------------------------------------------------------------------------


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    """Load a newline‑delimited JSON file.

    Each line should be a JSON object with a ``"messages"`` key.
    Falls back to looking for ``"text"`` or ``"instruction"``/``"output"``
    keys if ``"messages"`` is absent (accommodates simpler formats).
    """
    conversations: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed JSON at line %d: %s", lineno, exc)
                continue

            # Normalise to {"messages": [...]}
            if "messages" in obj:
                conversations.append(obj)
            elif "text" in obj:
                # Plain‑text format – wrap as a single "user" message.
                conversations.append(
                    {"messages": [{"role": "user", "content": obj["text"]}]}
                )
            elif "instruction" in obj:
                msgs: list[dict[str, str]] = [
                    {"role": "user", "content": obj["instruction"]}
                ]
                if obj.get("input"):
                    msgs[0]["content"] += "\n\n" + obj["input"]
                if obj.get("output"):
                    msgs.append({"role": "assistant", "content": obj["output"]})
                conversations.append({"messages": msgs})
            else:
                logger.warning("Skipping line %d: no recognised format.", lineno)

    return conversations


# ---------------------------------------------------------------------------
# Internal: chat template application
# ---------------------------------------------------------------------------


def _apply_chat_template(
    conversations: list[dict[str, Any]],
    tokenizer: Any,
) -> list[str]:
    """Render each conversation to a single string using the tokenizer's
    built‑in chat template, falling back to a simple format if unavailable.
    """
    texts: list[str] = []
    has_chat_template = (
        hasattr(tokenizer, "apply_chat_template")
        and tokenizer.chat_template is not None
    )

    for conv in conversations:
        messages = conv["messages"]
        if has_chat_template:
            try:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                texts.append(text)
                continue
            except Exception:
                pass  # fall through to manual format

        # Fallback: simple "Role: content\n" format.
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        texts.append("\n".join(parts) + tokenizer.eos_token)

    return texts


# ---------------------------------------------------------------------------
# Internal: tokenisation
# ---------------------------------------------------------------------------


def _tokenise_all(
    texts: list[str],
    tokenizer: Any,
    seq_length: int,
) -> list[dict[str, list[int]]]:
    """Tokenise a list of strings, truncating / padding to *seq_length*.

    Returns a list of dicts, each with ``input_ids``, ``attention_mask``,
    and ``labels``.  Labels are a copy of ``input_ids`` with padding
    positions set to ``-100`` (ignored by cross‑entropy loss).
    """
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )

    results: list[dict[str, list[int]]] = []
    for text in texts:
        enc = tokenizer(
            text,
            truncation=True,
            max_length=seq_length,
            padding=False,
            return_attention_mask=False,
        )
        ids: list[int] = enc["input_ids"]

        # Pad to seq_length.
        pad_len = seq_length - len(ids)
        attention_mask = [1] * len(ids) + [0] * pad_len
        labels = list(ids) + [-100] * pad_len
        ids = ids + [pad_id] * pad_len

        results.append(
            {
                "input_ids": ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        )

    return results


# ---------------------------------------------------------------------------
# Internal: CUDA (PyTorch) wrappers
# ---------------------------------------------------------------------------


def _wrap_for_cuda(
    train_data: list[dict[str, list[int]]],
    val_data: list[dict[str, list[int]]],
    tokenizer: Any,
) -> dict[str, Any]:
    """Wrap tokenised examples into PyTorch Datasets and provide a collate_fn."""
    import torch
    from torch.utils.data import Dataset as TorchDataset

    class _TokenisedDataset(TorchDataset):  # type: ignore[type-arg]
        def __init__(self, data: list[dict[str, list[int]]]) -> None:
            self._data = data

        def __len__(self) -> int:
            return len(self._data)

        def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
            item = self._data[idx]
            return {
                "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(
                    item["attention_mask"], dtype=torch.long
                ),
                "labels": torch.tensor(item["labels"], dtype=torch.long),
            }

    def _collate_fn(
        batch: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
            "labels": torch.stack([b["labels"] for b in batch]),
        }

    return {
        "train": _TokenisedDataset(train_data),
        "val": _TokenisedDataset(val_data),
        "collate_fn": _collate_fn,
    }

"""
GGUF conversion and quantisation.

Converts a merged Hugging Face model (safetensors) into GGUF format and
optionally quantises it (default: ``Q4_K_M``).

The conversion relies on ``llama-cpp-python``'s bundled ``convert_hf_to_gguf.py``
script, or a user‑provided ``llama.cpp`` checkout.
"""

from __future__ import annotations

import glob
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from engine.ipc import emit_error, emit_export_progress, emit_log

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Search paths for llama.cpp tools
# ---------------------------------------------------------------------------

_SEARCH_DIRS: list[str] = [
    # llama-cpp-python ships convert scripts in its package.
    os.path.join(sys.prefix, "lib", "python*", "site-packages", "llama_cpp"),
    os.path.join(sys.prefix, "Lib", "site-packages", "llama_cpp"),
    # Common user installs.
    os.path.expanduser("~/.orb/llama-cpp"),
    os.path.expanduser("~/llama.cpp"),
    # System‑wide.
    "/usr/local/share/llama.cpp",
]


def _find_convert_script() -> str | None:
    """Locate ``convert_hf_to_gguf.py`` on the filesystem."""
    # 1. Check if it's on PATH.
    which = shutil.which("convert_hf_to_gguf.py")
    if which:
        return which

    # 2. Try well‑known locations.
    for pattern in _SEARCH_DIRS:
        for d in glob.glob(pattern):
            candidate = os.path.join(d, "convert_hf_to_gguf.py")
            if os.path.isfile(candidate):
                return candidate

    # 3. Check if installable via pip.
    try:
        import llama_cpp  # type: ignore[import-untyped]

        pkg_dir = os.path.dirname(llama_cpp.__file__)
        candidate = os.path.join(pkg_dir, "convert_hf_to_gguf.py")
        if os.path.isfile(candidate):
            return candidate
    except ImportError:
        pass

    return None


def _find_quantize_binary() -> str | None:
    """Locate the ``llama-quantize`` (or ``quantize``) binary."""
    for name in ("llama-quantize", "quantize"):
        path = shutil.which(name)
        if path:
            return path

    # Check known directories.
    for pattern in _SEARCH_DIRS:
        for d in glob.glob(pattern):
            for name in ("llama-quantize", "quantize", "llama-quantize.exe", "quantize.exe"):
                candidate = os.path.join(d, name)
                if os.path.isfile(candidate):
                    return candidate
            # Also check build/ subdirectory.
            build_dir = os.path.join(d, "build", "bin")
            if os.path.isdir(build_dir):
                for name in ("llama-quantize", "quantize", "llama-quantize.exe", "quantize.exe"):
                    candidate = os.path.join(build_dir, name)
                    if os.path.isfile(candidate):
                        return candidate

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def export_gguf(
    model_path: str,
    output_path: str,
    quant_type: str = "Q4_K_M",
) -> str:
    """Convert a merged HF model to quantised GGUF.

    Pipeline:
        1. ``convert_hf_to_gguf.py`` → f16 GGUF
        2. ``llama-quantize``         → quantised GGUF
        3. Clean up intermediate f16 file

    Parameters
    ----------
    model_path:
        Path to the merged model directory (safetensors + config.json).
    output_path:
        Final output path for the quantised ``.gguf`` file.
    quant_type:
        Quantisation type, e.g. ``"Q4_K_M"``, ``"Q5_K_M"``, ``"Q8_0``.

    Returns
    -------
    str
        ``output_path`` on success.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    convert_script = _find_convert_script()
    quantize_bin = _find_quantize_binary()

    if convert_script is None:
        emit_error(
            "Cannot find convert_hf_to_gguf.py. "
            "Install llama-cpp-python or place llama.cpp in ~/.orb/llama-cpp/.",
            detail="Searched: " + ", ".join(_SEARCH_DIRS),
            recoverable=False,
        )
        raise FileNotFoundError("convert_hf_to_gguf.py not found")

    # ---- Step 1: Convert HF → f16 GGUF ------------------------------
    emit_export_progress("converting", "Converting model to GGUF (f16)…")

    f16_path = output_path.replace(".gguf", "-f16.gguf")
    if f16_path == output_path:
        f16_path = output_path + ".f16.gguf"

    cmd_convert = [
        sys.executable,
        convert_script,
        model_path,
        "--outfile",
        f16_path,
        "--outtype",
        "f16",
    ]

    emit_log("debug", f"Running: {' '.join(cmd_convert)}")

    result = subprocess.run(
        cmd_convert,
        capture_output=True,
        text=True,
        timeout=3600,  # 1 hour timeout
    )

    if result.returncode != 0:
        emit_error(
            "GGUF conversion failed.",
            detail=result.stderr[-2000:] if result.stderr else "No stderr output",
            recoverable=False,
        )
        raise RuntimeError(f"convert_hf_to_gguf.py failed (exit {result.returncode})")

    if not os.path.isfile(f16_path):
        emit_error(
            "GGUF conversion completed but output file not found.",
            detail=f"Expected: {f16_path}",
            recoverable=False,
        )
        raise FileNotFoundError(f"Expected f16 GGUF at {f16_path}")

    emit_log("info", f"f16 GGUF created: {f16_path}")

    # ---- Step 2: Quantise f16 → target quant -------------------------
    if quant_type.upper() == "F16":
        # User wants f16, just rename.
        shutil.move(f16_path, output_path)
        emit_export_progress("complete", f"GGUF exported (f16): {output_path}")
        return output_path

    if quantize_bin is None:
        emit_log(
            "warning",
            "llama-quantize binary not found — keeping f16 GGUF. "
            "Install llama.cpp to enable quantisation.",
        )
        shutil.move(f16_path, output_path)
        emit_export_progress(
            "complete",
            f"GGUF exported (f16, quantisation skipped): {output_path}",
        )
        return output_path

    emit_export_progress("quantizing", f"Quantising to {quant_type}…")

    cmd_quant = [quantize_bin, f16_path, output_path, quant_type]

    emit_log("debug", f"Running: {' '.join(cmd_quant)}")

    result = subprocess.run(
        cmd_quant,
        capture_output=True,
        text=True,
        timeout=3600,
    )

    if result.returncode != 0:
        emit_error(
            f"Quantisation to {quant_type} failed.",
            detail=result.stderr[-2000:] if result.stderr else "No stderr output",
            recoverable=False,
        )
        raise RuntimeError(f"llama-quantize failed (exit {result.returncode})")

    # ---- Step 3: Cleanup intermediate f16 file -----------------------
    try:
        os.remove(f16_path)
        emit_log("debug", f"Cleaned up intermediate file: {f16_path}")
    except OSError:
        pass  # non‑fatal

    emit_export_progress("complete", f"GGUF exported ({quant_type}): {output_path}")
    emit_log("info", f"GGUF export complete: {output_path}")

    return output_path


# ---------------------------------------------------------------------------
# LoRA GGUF merge
# ---------------------------------------------------------------------------


def _find_export_lora_binary() -> str | None:
    """Locate the ``llama-export-lora`` (or ``export-lora``) binary."""
    for name in ("llama-export-lora", "export-lora"):
        path = shutil.which(name)
        if path:
            return path

    # Check known directories.
    for pattern in _SEARCH_DIRS:
        for d in glob.glob(pattern):
            for name in (
                "llama-export-lora",
                "export-lora",
                "llama-export-lora.exe",
                "export-lora.exe",
            ):
                candidate = os.path.join(d, name)
                if os.path.isfile(candidate):
                    return candidate
            # Also check build/ subdirectory.
            build_dir = os.path.join(d, "build", "bin")
            if os.path.isdir(build_dir):
                for name in (
                    "llama-export-lora",
                    "export-lora",
                    "llama-export-lora.exe",
                    "export-lora.exe",
                ):
                    candidate = os.path.join(build_dir, name)
                    if os.path.isfile(candidate):
                        return candidate

    return None


def export_gguf_from_lora(
    base_gguf_path: str,
    lora_gguf_path: str,
    output_path: str,
    config: dict[str, Any] | None = None,
) -> str:
    """Merge a llama.cpp GGUF LoRA adapter with a base GGUF model.

    Uses ``llama-export-lora`` to produce a single merged GGUF file.
    If the binary is not found, falls back to copying the base model
    so the pipeline can continue (the user can merge manually later).

    Parameters
    ----------
    base_gguf_path:
        Path to the base model in GGUF format.
    lora_gguf_path:
        Path to the LoRA adapter file (GGUF or ``.bin``).
    output_path:
        Destination path for the merged ``.gguf`` file.
    config:
        Optional config dict.  If it contains ``llamacpp_export_lora_path``
        that path is tried first when locating the binary.

    Returns
    -------
    str
        ``output_path`` on success.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # ---- locate the merge binary -------------------------------------
    export_bin: str | None = None

    if config and config.get("llamacpp_export_lora_path"):
        candidate = config["llamacpp_export_lora_path"]
        if os.path.isfile(candidate):
            export_bin = candidate

    if export_bin is None:
        export_bin = _find_export_lora_binary()

    if export_bin is None:
        emit_log(
            "warning",
            "llama-export-lora binary not found — copying base model as fallback. "
            "Install llama.cpp to enable LoRA merging.",
        )
        emit_export_progress(
            "merging",
            "llama-export-lora not found; copying base model as fallback.",
        )
        shutil.copy2(base_gguf_path, output_path)
        emit_export_progress(
            "complete",
            f"Base model copied (LoRA merge skipped): {output_path}",
        )
        return output_path

    # ---- merge -------------------------------------------------------
    emit_export_progress("merging", "Merging LoRA adapter into base GGUF…")

    cmd = [
        export_bin,
        "--model", base_gguf_path,
        "--lora", lora_gguf_path,
        "--output", output_path,
    ]

    emit_log("debug", f"Running: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=3600,
    )

    if result.returncode != 0:
        emit_error(
            "LoRA merge failed.",
            detail=result.stderr[-2000:] if result.stderr else "No stderr output",
            recoverable=False,
        )
        raise RuntimeError(
            f"llama-export-lora failed (exit {result.returncode})"
        )

    if not os.path.isfile(output_path):
        emit_error(
            "LoRA merge completed but output file not found.",
            detail=f"Expected: {output_path}",
            recoverable=False,
        )
        raise FileNotFoundError(f"Expected merged GGUF at {output_path}")

    emit_export_progress("complete", f"Merged GGUF exported: {output_path}")
    emit_log("info", f"LoRA merge complete: {output_path}")

    return output_path

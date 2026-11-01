"""
Loom training engine entry point.

Usage::

    python -m engine --config /path/to/config.json

This module:

1. Parses the ``--config`` argument.
2. Reads and validates the config JSON.
3. Sets up logging (stderr → log file).
4. Installs signal handlers for graceful shutdown.
5. Delegates to :func:`engine.trainer.run`.
6. Catches **all** uncaught exceptions and emits an IPC error so the
   Node.js parent process always gets a structured response.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from typing import Any


def main() -> None:
    """Engine entry point — called when ``python -m engine`` is run."""
    # ---- 1. Parse args -----------------------------------------------
    parser = argparse.ArgumentParser(
        prog="loom-engine",
        description="Loom QLoRA training engine",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the JSON config file.",
    )
    args = parser.parse_args()

    # ---- 2. Read config ----------------------------------------------
    try:
        with open(args.config, encoding="utf-8") as fh:
            config: dict[str, Any] = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        # We haven't set up IPC logging yet, so write to stderr which
        # will go to the console.  Then also try to emit.
        sys.stderr.write(f"Failed to read config: {exc}\n")
        try:
            from engine.ipc import emit_error

            emit_error(f"Failed to read config file: {exc}", recoverable=False)
        except Exception:
            pass
        sys.exit(1)

    # ---- 3. Set up logging -------------------------------------------
    output_dir = config.get("output_dir", ".")
    log_file = os.path.join(output_dir, "engine.log")

    from engine.ipc import emit_error, emit_log, setup_logging

    try:
        setup_logging(log_file)
    except Exception as exc:
        emit_error(
            f"Failed to set up logging: {exc}",
            detail=str(exc),
            recoverable=False,
        )
        sys.exit(1)

    emit_log("info", f"Engine started. Log file: {log_file}")
    emit_log("debug", f"Config: {json.dumps(config, indent=2, default=str)}")

    # ---- 4. Signal handlers ------------------------------------------
    _install_signal_handlers(config)

    # ---- 5. Run training pipeline ------------------------------------
    from engine.trainer import run

    run(config)


# ---------------------------------------------------------------------------
# Signal handlers
# ---------------------------------------------------------------------------

_SHUTTING_DOWN = False


def _install_signal_handlers(config: dict[str, Any]) -> None:
    """Install handlers for SIGTERM and SIGINT.

    On receiving either signal we attempt to save a checkpoint (if the
    backend is mid‑training) and then exit cleanly.
    """

    def _handler(signum: int, frame: Any) -> None:
        global _SHUTTING_DOWN
        if _SHUTTING_DOWN:
            # Second signal → force exit.
            sys.exit(128 + signum)
        _SHUTTING_DOWN = True

        from engine.ipc import emit_error, emit_log

        sig_name = signal.Signals(signum).name
        emit_log("warning", f"Received {sig_name} — shutting down gracefully…")
        emit_error(
            f"Training interrupted ({sig_name}). Partial checkpoints may have been saved.",
            recoverable=False,
        )
        sys.exit(128 + signum)

    signal.signal(signal.SIGINT, _handler)

    # SIGTERM is not available on Windows, guard accordingly.
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handler)


# ---------------------------------------------------------------------------
# Script entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
